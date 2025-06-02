import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import layer_init
import tyro
import random
from tqdm import tqdm
from ttt import TTTConfig, TTTModel

class MLPAgent(nn.Module):
  def __init__(self, envs, intermediate_size=64):
    super().__init__()
    self.critic = nn.Sequential(
      layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), intermediate_size)),
      nn.Tanh(),
      layer_init(nn.Linear(intermediate_size, intermediate_size)),
      nn.Tanh(),
      layer_init(nn.Linear(intermediate_size, 1), std=1.0),
    )
    self.actor = nn.Sequential(
      layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), intermediate_size)),
      nn.Tanh(),
      layer_init(nn.Linear(intermediate_size, intermediate_size)),
      nn.Tanh(),
      layer_init(nn.Linear(intermediate_size, envs.single_action_space.n), std=0.01),
    )

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, action=None):
    logits = self.actor(x)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class TTTActor(nn.Module):
  def __init__(self, envs, intermediate_size=64, obs_mask=None):
    super().__init__()

    self.config = TTTConfig(
            vocab_size=envs.single_observation_space.shape[0],
            hidden_size=intermediate_size,
            max_position_embeddings=500,
            intermediate_size=intermediate_size * 2,
            num_attention_heads=2,
            num_hidden_layers=4,
            dropout=0.1,
            hidden_act="relu",
            max_episode_steps=500,
            ttt_layer_type="linear",
            ttt_base_lr=1e-2,
        )

    self.actor = TTTModel(self.config, 500)

    self.head = nn.Linear(intermediate_size, envs.single_action_space.n)

    self.obs_mask = None
    if obs_mask is not None:
      self.obs_mask = obs_mask

  def forward(self, x):
    if self.obs_mask is not None:
      x = x * self.obs_mask
    return self.head(self.actor(x).last_hidden_state)

  def get_action_and_value(self, x, action=None):
    if self.obs_mask is not None:
      x = x * self.obs_mask

    hidden_states = self.actor(x).last_hidden_state
    logits = self.head(hidden_states)

    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), None
  
def make_env(env_id, idx, capture_video, run_name):
  def thunk():
    if capture_video and idx == 0:
      env = gym.make(env_id, render_mode="rgb_array")
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
      env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  return thunk

if __name__ == "__main__":
  # args = tyro.cli(Args)

  # if not args.model_path:
  #     raise ValueError("A --model-path to a pre-trained agent model is required.")
  
  # run_name = f"{args.env_id}__{args.exp_name}__{args.model_type}__{args.seed}__{int(time.time())}__expert_collection"

  # random.seed(args.seed)
  # np.random.seed(args.seed)
  # torch.manual_seed(args.seed)
  # torch.backends.cudnn.deterministic = args.torch_deterministic
  batch_size = 64

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, False, "CartPole-v1") for i in range(batch_size)]
  )

  teacher_path = "/home/waitz/cs224r-proj/cartpole_agent.pth"

  teacher = MLPAgent(envs).to(device)
  teacher.load_state_dict(torch.load(teacher_path, map_location=device))
  teacher.eval()
  print(f"Loaded expert model from {teacher_path}")

  obs_mask = torch.tensor([1, 0, 1, 0], device=device).view(1, 1, 4)
  learner = TTTActor(envs, obs_mask=obs_mask).to(device)

  num_episodes = 2000
  
  # Gradient accumulation settings
  accumulation_steps = 4  # Accumulate gradients over 4 episodes before optimizer step
  
  # Move optimizer outside the loop to maintain optimization state
  optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-4, weight_decay=0.0001)
  
  # Add learning rate scheduler
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes//accumulation_steps, eta_min=1e-6)

  all_rewards = []
  all_seqlens = []
  learning_rates = []  # Track learning rates for plotting
  accumulated_loss = 0.0
  
  for i in tqdm(range(num_episodes)):
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device).unsqueeze(1)
    done = False
    trajectory_obs = None
    rewards = []
    
    # sample a trajectory 
    while not done:
      if trajectory_obs is None:
        trajectory_obs = obs.clone()
      else:
        trajectory_obs = torch.cat((trajectory_obs, obs), dim=1)
      
      with torch.no_grad():
        # unsqueeze is to add singleton batch dim
        action, _, _, _ = learner.get_action_and_value(trajectory_obs)
        action = action[:, -1:]
      
      next_obs, reward, termination, truncation, info = envs.step(action.flatten().cpu().numpy())
      # Fix the IndexError by properly handling the boolean arrays
      done = termination.any() or truncation.any()
      rewards.append(reward)
      obs = torch.Tensor(next_obs).to(device).unsqueeze(1)

    # Convert trajectory observations to a single tensor
    # breakpoint()
    trajectory_obs_tensor = trajectory_obs
    
    # DAgger loss: supervised learning loss between learner and expert actions
    expert_logits = teacher.actor(trajectory_obs_tensor)
    learner_logits = learner.forward(trajectory_obs_tensor)

    # breakpoint()
    expert_probs = torch.softmax(expert_logits, dim=-1)
    expert_action_indices = torch.argmax(expert_probs, dim=-1)

    # TODO perhaps KL divergence might be better here? 
    loss = torch.nn.functional.cross_entropy(learner_logits.view(-1, 2), expert_action_indices.view(-1))
    
    # Scale loss by accumulation steps to maintain consistent gradient magnitude
    loss = loss / accumulation_steps
    accumulated_loss += loss.item()
    
    # Backpropagation (accumulate gradients)
    loss.backward()
    
    # Perform optimizer step only every accumulation_steps episodes
    if (i + 1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
      
      # Update learning rate
      scheduler.step()
      learning_rates.append(scheduler.get_last_lr()[0])
      
      # Reset accumulated loss
      accumulated_loss = 0.0

    all_seqlens.append(len(rewards))
    rewards = np.stack(rewards, axis=1)
    all_rewards.extend(rewards.sum(axis=1).tolist())
    
    if i % 100 == 0:
        current_lr = scheduler.get_last_lr()[0] if learning_rates else 1e-4
        print(f"Episode {i}, DAgger Loss: {accumulated_loss * accumulation_steps:.4f}, LR: {current_lr:.6f}")
        print(f"Learner return: {sum(all_rewards) / len(all_rewards)}")
        # print(f"Learner seqlen: {sum(all_seqlens) / len(all_seqlens)}")

        all_rewards = []
        all_seqlens = []

  