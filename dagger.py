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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, False, "CartPole-v1") for i in range(1)]
  )

  teacher_path = "/home/waitz/cs224r-proj/cartpole_agent.pth"

  teacher = MLPAgent(envs).to(device)
  teacher.load_state_dict(torch.load(teacher_path, map_location=device))
  teacher.eval()
  print(f"Loaded expert model from {teacher_path}")

  learner = MLPAgent(envs).to(device)
  
  # Move optimizer outside the loop to maintain optimization state
  optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)

  num_episodes = 1000

  for i in tqdm(range(num_episodes)):
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    done = False
    trajectory_obs = []
    rewards = []
    
    # sample a trajectory 
    while not done:
      trajectory_obs.append(obs.clone())
      
      with torch.no_grad():
        action, _, _, _ = learner.get_action_and_value(obs)
      
      next_obs, reward, termination, truncation, info = envs.step(action.cpu().numpy())
      # Fix the IndexError by properly handling the boolean arrays
      done = termination[0] or truncation[0]
      rewards.append(reward[0])
      obs = torch.Tensor(next_obs).to(device)

    # Convert trajectory observations to a single tensor
    trajectory_obs_tensor = torch.stack(trajectory_obs)
    
    # DAgger loss: supervised learning loss between learner and expert actions
    expert_logits = teacher.actor(trajectory_obs_tensor)
    learner_logits = learner.actor(trajectory_obs_tensor)

    # breakpoint()
    expert_probs = torch.softmax(expert_logits, dim=-1)
    expert_action_indices = torch.argmax(expert_probs, dim=-1)

    # TODO perhaps KL divergence might be better here? 
    loss = torch.nn.functional.cross_entropy(learner_logits.view(-1, 2), expert_action_indices.view(-1))
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Episode {i}, DAgger Loss: {loss.item():.4f}")
        print(f"Learner return: {sum(rewards)}")
