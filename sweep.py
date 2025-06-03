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
from dagger import TTTActor
from dagger import make_env
from dataclasses import dataclass
import json
import os

def load_model(model_path):
    # Set inference device and default tensor type
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint dictionaries and direct state dictionaries
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']


        with open(model_path.replace('latest_checkpoint.pth', 'config.json'), 'r') as f:
            config = json.load(f)
    else:
        state_dict = checkpoint
        config = None

    # for key in list(state_dict.keys()):
    #     if key.startswith('actor.'):
    #         new_key = key.replace('actor.', '')
    #         state_dict[new_key] = state_dict.pop(key)

    return state_dict, config

def sample_trajectory(actor):
    done = False
    episode_rewards = []
    t = 0
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device).unsqueeze(1)
    traj_obs = None


    # data = {
    #     "etas": [],
    #     "grads": [],
    #     "grad_time_fracs": [],
    #     "lrs": [],
    #     "total_time": []
    # }

    while not done:
        if traj_obs is None:
            traj_obs = obs.clone()
        else:
            traj_obs = torch.cat((traj_obs, obs), dim=1)
        
        with torch.no_grad():
            # actor itself will throw out (mask) some pieces of the obs to make it a POMDP
            # we don't need to handle that here
            action, _, _, _ = actor.get_action_and_value(traj_obs)
            action = action[:, -1:]
        
        next_obs, reward, termination, truncation, _ = envs.step(action.flatten().cpu().numpy())
        done = termination.any() or truncation.any()
        episode_rewards.append(reward)
        obs = torch.Tensor(next_obs).to(device).unsqueeze(1)
        t += 1

        # data["etas"].append(actor.actor._get_eta().tolist())
        # data["grads"].append(actor.actor._get_gradlists().tolist())
        # data["grad_time_fracs"].append(actor.actor._get_grad_time_fracs())
        # data["lrs"].append(actor.actor._get_lr().tolist())
        # data["total_time"].append(actor.actor._get_time_forward())

    # print(f"Episode rewards: {episode_rewards}")
    # print(f"Episode length: {t}")
    # print(f"Episode reward mean: {np.mean(episode_rewards)}")
    # print(f"Episode reward std: {np.std(episode_rewards)}")

    out = {
        "episode_rewards": [elem[0] for elem in episode_rewards],
        "t": t,
        # these are both regular lists (not np arrays)
        "grad_time_frac": actor.actor._get_grad_time_fracs(),
        "time_forward": actor.actor._get_time_forward()
    }

    return out

def sweep(actor):
    samples_per_config = 64
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data = {}
    for threshold in thresholds:
        data[threshold] = []

    for grad_threshold in thresholds:
      os.environ['TTT_THRESHOLD'] = str(grad_threshold)

      for i in tqdm(range(samples_per_config)):
        out = sample_trajectory(actor)
        data[grad_threshold].append(out)

        # episode_rewards, t, grad_time_frac, time_forward = sample_trajectory(actor)
        # data[grad_threshold].append({
        #     "episode_rewards": episode_rewards.tolist() if hasattr(episode_rewards, 'tolist') else episode_rewards,
        #     "t": int(t) if hasattr(t, 'item') else t,
        #     "mean": float(mean) if hasattr(mean, 'item') else mean,
        #     "std": float(std) if hasattr(std, 'item') else std,
        #     "grad_time_frac": grad_time_frac,
        #     "time_forward": time_forward
        # })
        # print(f"Grad threshold: {grad_threshold}, Episode rewards: {episode_rewards}, Episode length: {t}, Episode reward mean: {mean}, Episode reward std: {std}")

      average_episode_rewards = np.mean([d["t"] for d in data[grad_threshold]])
      average_forward_time = np.mean([np.sum(d["time_forward"]) / d["t"] for d in data[grad_threshold]])
      # average_forward_time_per_layer = np.mean([d["time_forward"] for d in data[grad_threshold]], axis=0)
      average_grad_frac = np.mean([d["grad_time_frac"] for d in data[grad_threshold]])

      print(f"Grad threshold: {grad_threshold}, Average episode length: {average_episode_rewards}, Average forward time: {average_forward_time}, Average grad frac: {average_grad_frac}")

      # average_episode_rewards = np.mean([d["mean"] for d in data[grad_threshold]])
      # average_episode_length = np.mean([d["t"] for d in data[grad_threshold]])
      # average_episode_std = np.mean([d["std"] for d in data[grad_threshold]])
      # average_grad_time = np.mean([d["grad_time_frac"] for d in data[grad_threshold]])
      # average_time_forward = np.mean([d["time_forward"] for d in data[grad_threshold]])
      # print(f"Grad threshold: {grad_threshold}, Average episode length: {average_episode_length}, Average grad time: {average_grad_time}, Average time forward: {average_time_forward}")

    return data

@dataclass
class PleasureConfig:
    model: str
    # batch_size: int = 1

if __name__ == "__main__":
    # parse command line arguments
    config = tyro.cli(PleasureConfig)

    batch_size = 1 # (for test-time)
    state_dict, config = load_model(config.model)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, False, "CartPole-v1") for i in range(batch_size)]
    )

    obs_mask = torch.tensor([1, 0, 1, 0], device=device).view(1, 1, 4)
    matryoshka_actor = TTTActor(envs, intermediate_size=64, obs_mask=obs_mask)
    matryoshka_actor.load_state_dict(state_dict)

    matryoshka_actor = matryoshka_actor.to(device)

    matryoshka_actor.eval()

    data = sweep(matryoshka_actor)

    with open("sweep_pleasure_data.json", "w") as f:
      json.dump(data, f)