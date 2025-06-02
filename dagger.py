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
from ttt_custom import TTTConfig, TTTModel
import os
import json

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
  optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-3, weight_decay=0.0001)
  
  # Add learning rate scheduler
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes//accumulation_steps, eta_min=1e-6)

  # Checkpointing setup
  checkpoint_dir = "checkpoints"
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_interval = 500  # Save checkpoint every 500 episodes
  resume_from_checkpoint = None  # Set to checkpoint path to resume training
  
  # Save configuration
  config = {
    'training': {
      'num_episodes': num_episodes,
      'batch_size': batch_size,
      'accumulation_steps': accumulation_steps,
      'checkpoint_interval': checkpoint_interval,
      'expert_trajectory_prob': 0.1,
    },
    'model': {
      'learner_type': 'TTTActor',
      'intermediate_size': 64,
      'obs_mask': [1, 0, 1, 0],  # position and velocity masked
    },
    'ttt_config': {
      'vocab_size': envs.single_observation_space.shape[0],
      'hidden_size': 64,
      'max_position_embeddings': 500,
      'intermediate_size': 128,
      'num_attention_heads': 2,
      'num_hidden_layers': 4,
      'dropout': 0.1,
      'hidden_act': 'relu',
      'max_episode_steps': 500,
      'ttt_layer_type': 'linear',
      'ttt_base_lr': 1e-2,
    },
    'optimizer': {
      'type': 'AdamW',
      'lr': 1e-3,
      'weight_decay': 0.0001,
    },
    'scheduler': {
      'type': 'CosineAnnealingLR',
      'T_max': num_episodes // accumulation_steps,
      'eta_min': 1e-6,
    },
    'environment': {
      'env_id': 'CartPole-v1',
      'observation_space_shape': list(envs.single_observation_space.shape),
      'action_space_n': envs.single_action_space.n,
    },
    'expert': {
      'model_path': teacher_path,
      'model_type': 'MLPAgent',
    },
    'device': str(device),
  }
  
  # Save config to checkpoint directory
  config_path = os.path.join(checkpoint_dir, "config.json")
  with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
  print(f"Configuration saved to {config_path}")
  
  # Initialize training state
  start_episode = 0
  all_rewards = []
  all_seqlens = []
  learning_rates = []  # Track learning rates for plotting
  accumulated_loss = 0.0
  
  # Function to save checkpoint
  def save_checkpoint(episode, model, optimizer, scheduler, all_rewards, all_seqlens, learning_rates, accumulated_loss):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth")
    checkpoint = {
      'episode': episode,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'all_rewards': all_rewards,
      'all_seqlens': all_seqlens,
      'learning_rates': learning_rates,
      'accumulated_loss': accumulated_loss,
      'random_state': random.getstate(),
      'numpy_random_state': np.random.get_state(),
      'torch_random_state': torch.get_rng_state(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    
    # Save training metadata
    metadata = {
      'episode': episode,
      'num_episodes': num_episodes,
      'batch_size': batch_size,
      'accumulation_steps': accumulation_steps,
      'checkpoint_interval': checkpoint_interval,
    }
    with open(os.path.join(checkpoint_dir, "training_metadata.json"), 'w') as f:
      json.dump(metadata, f, indent=2)
  
  # Function to load checkpoint
  def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore random states for reproducibility
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    
    return (checkpoint['episode'], checkpoint['all_rewards'], checkpoint['all_seqlens'], 
            checkpoint['learning_rates'], checkpoint['accumulated_loss'])
  
  # Resume from checkpoint if specified
  if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
      print(f"Resuming training from {resume_from_checkpoint}")
      start_episode, all_rewards, all_seqlens, learning_rates, accumulated_loss = load_checkpoint(
        resume_from_checkpoint, learner, optimizer, scheduler
      )
      print(f"Resumed from episode {start_episode}")
    else:
      print(f"Checkpoint file {resume_from_checkpoint} not found. Starting from scratch.")
  
  # Check for latest checkpoint if no specific checkpoint specified
  elif os.path.exists(os.path.join(checkpoint_dir, "latest_checkpoint.pth")):
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    print(f"Found latest checkpoint: {latest_checkpoint}")
    response = input("Resume from latest checkpoint? (y/n): ").lower().strip()
    if response == 'y':
      start_episode, all_rewards, all_seqlens, learning_rates, accumulated_loss = load_checkpoint(
        latest_checkpoint, learner, optimizer, scheduler
      )
      print(f"Resumed from episode {start_episode}")
  
  for i in tqdm(range(start_episode, num_episodes)):
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device).unsqueeze(1)
    done = False
    trajectory_obs = None
    rewards = []

    expert_trajectory = False
    # sometimes sample expert trajectories
    if np.random.random() < 0.1:
        expert_trajectory = True
    
    # sample a trajectory 
    while not done:
      if trajectory_obs is None:
        trajectory_obs = obs.clone()
      else:
        trajectory_obs = torch.cat((trajectory_obs, obs), dim=1)
      
      with torch.no_grad():
        # unsqueeze is to add singleton batch dim
        if expert_trajectory:
          action, _, _, _ = teacher.get_action_and_value(trajectory_obs)
        else:
          action, _, _, _ = learner.get_action_and_value(trajectory_obs)
        action = action[:, -1:]
      
      next_obs, reward, termination, truncation, info = envs.step(action.flatten().cpu().numpy())

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
    
    # Save checkpoint periodically
    if (i + 1) % checkpoint_interval == 0:
      save_checkpoint(i + 1, learner, optimizer, scheduler, all_rewards, all_seqlens, learning_rates, accumulated_loss)
    
    if i % 100 == 0:
        current_lr = scheduler.get_last_lr()[0] if learning_rates else 1e-4
        print(f"Episode {i}, DAgger Loss: {accumulated_loss * accumulation_steps:.4f}, LR: {current_lr:.6f}")
        print(f"Learner return: {sum(all_rewards) / len(all_rewards)}")
        # print(f"Learner seqlen: {sum(all_seqlens) / len(all_seqlens)}")

        all_rewards = []
        all_seqlens = []

  # Save final checkpoint
  save_checkpoint(num_episodes, learner, optimizer, scheduler, all_rewards, all_seqlens, learning_rates, accumulated_loss)
  print("Training completed. Final checkpoint saved.")
 