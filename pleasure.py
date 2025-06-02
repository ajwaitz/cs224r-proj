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
from dagger import TTTActor
from dagger import make_env
from dataclasses import dataclass


def load_model(model_path):
    # Set inference device and default tensor type
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    
    # Strip 'ttt.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('ttt.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v
    
    return new_state_dict, config


def main(actor):
    done = False
    episode_rewards = []
    t = 0
    obs = envs.reset()
    traj_obs = None
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
    
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode length: {t}")
    print(f"Episode reward mean: {np.mean(episode_rewards)}")
    print(f"Episode reward std: {np.std(episode_rewards)}")


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

    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, False, "CartPole-v1") for i in range(batch_size)]
    )

    matryoshka_actor = TTTActor(envs, intermediate_size=64, obs_mask=None)
    matryoshka_actor.actor.load_state_dict(state_dict)

    matryoshka_actor.eval()

    main(matryoshka_actor)