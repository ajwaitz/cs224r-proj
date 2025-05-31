# source: https://github.com/vwxyzjn/cleanrl/
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# Modified to collect expert trajectories
import os
import random
import time
from dataclasses import dataclass, field
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
# from transformers.transformeragent import TransformerAgent # Keep if TransformerAgent might be used
from tqdm import tqdm
import pickle

# Assuming utils.py with layer_init is in the same directory or accessible
# If not, you might need to define layer_init here or adjust the import.
try:
    from utils import layer_init
except ImportError:
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False # Default to False for trajectory collection
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    
    # Algorithm specific arguments from original PPO
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1228800
    """total timesteps of the experiments (not directly used for collection)"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (not used for collection)"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout (not directly used for collection structure)"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks (not used for collection)"""
    gamma: float = 0.99
    """the discount factor gamma (not used for collection)"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation (not used for collection)"""
    num_minibatches: int = 4
    """the number of mini-batches (not used for collection)"""
    update_epochs: int = 4
    """the K epochs to update the policy (not used for collection)"""
    norm_adv: bool = True
    """Toggles advantages normalization (not used for collection)"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient (not used for collection)"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function (not used for collection)"""
    ent_coef: float = 0.01
    """coefficient of the entropy (not used for collection)"""
    vf_coef: float = 0.5
    """coefficient of the value function (not used for collection)"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping (not used for collection)"""
    target_kl: float = None
    """the target KL divergence threshold (not used for collection)"""
    
    model_type: str = "mlp"
    """the type of model parameterization to use (mlp or transformer)"""
    intermediate_size: int = 64
    """intermediate size for MLP agent layers"""

    # Arguments for expert trajectory collection
    model_path: str = ""
    """path to the pre-trained expert model file"""
    num_expert_episodes: int = 100
    """number of expert episodes to collect"""
    output_file_name: str = ""
    """name for the output pickle file (optional, defaults to a generated name, placed in trajectories/<model_type>/)"""

    # Filled in runtime (from original script, less relevant here)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

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

if __name__ == "__main__":
    args = tyro.cli(Args)

    if not args.model_path:
        raise ValueError("A --model-path to a pre-trained agent model is required.")

    run_name = f"{args.env_id}__{args.exp_name}__{args.model_type}__{args.seed}__{int(time.time())}__expert_collection"

    if args.track: # wandb setup if tracking is enabled
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True, # Monitors videos if available
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.model_type == "mlp":
        agent = MLPAgent(envs, intermediate_size=args.intermediate_size).to(device)
        print(f"Using MLP agent with intermediate_size={args.intermediate_size}.")
    # elif args.model_type == "transformer":
    #     # Ensure TransformerAgent is correctly imported and defined if used
    #     from transformers.transformeragent import TransformerAgent 
    #     agent = TransformerAgent(envs).to(device)
    #     print("Using Transformer agent.")
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Supported types: 'mlp'.") # 'transformer'

    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()
    print(f"Loaded expert model from {args.model_path}")

    expert_trajectories_data = []
    episode_observations = [[] for _ in range(args.num_envs)]
    episode_actions = [[] for _ in range(args.num_envs)]
    episode_rewards = [[] for _ in range(args.num_envs)]
    episode_next_observations = [[] for _ in range(args.num_envs)]
    episode_dones = [[] for _ in range(args.num_envs)] # Stores the done signal for each transition

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.Tensor(next_obs_np).to(device)
    
    episodes_collected_count = 0
    
    progress_bar_disabled = args.capture_video and args.num_envs > 0 # Disable progress bar if capturing video as it might mess with prints
    with tqdm(total=args.num_expert_episodes, desc="Collecting Expert Episodes", disable=progress_bar_disabled) as pbar:
        while episodes_collected_count < args.num_expert_episodes:
            current_obs_np = next_obs_np

            with torch.no_grad():
                action_tensor, _, _, _ = agent.get_action_and_value(next_obs_tensor)
            action_np = action_tensor.cpu().numpy()

            next_obs_after_step_np, reward_np, terminations_np, truncations_np, infos = envs.step(action_np)
            done_np = np.logical_or(terminations_np, truncations_np)

            for i in range(args.num_envs):
                episode_observations[i].append(current_obs_np[i])
                episode_actions[i].append(action_np[i])
                episode_rewards[i].append(reward_np[i])
                episode_next_observations[i].append(next_obs_after_step_np[i])
                episode_dones[i].append(done_np[i])

                if done_np[i]:
                    if episodes_collected_count < args.num_expert_episodes:
                        final_info_env = infos.get("final_info", [None]*args.num_envs)[i]
                        ep_return = None
                        ep_length = None
                        if final_info_env and "episode" in final_info_env:
                            ep_return = final_info_env["episode"]["r"]
                            ep_length = final_info_env["episode"]["l"]
                        
                        trajectory = {
                            "observations": np.array(episode_observations[i], dtype=np.float32),
                            "actions": np.array(episode_actions[i], dtype=np.int64), 
                            "rewards": np.array(episode_rewards[i], dtype=np.float32),
                            "next_observations": np.array(episode_next_observations[i], dtype=np.float32),
                            "dones": np.array(episode_dones[i], dtype=bool), 
                            "terminated": np.array([terminations_np[i]] * len(episode_observations[i]), dtype=bool), 
                            "truncated": np.array([truncations_np[i]] * len(episode_observations[i]), dtype=bool), 
                        }
                        if ep_return is not None: trajectory["ep_return"] = ep_return
                        if ep_length is not None: trajectory["ep_length"] = ep_length
                        
                        expert_trajectories_data.append(trajectory)
                        episodes_collected_count += 1
                        pbar.update(1)
                        
                        if ep_return is not None and ep_length is not None:
                            # Convert to scalar if they are numpy arrays for tqdm
                            scalar_display_return = ep_return.item() if isinstance(ep_return, np.ndarray) else ep_return
                            scalar_display_length = ep_length.item() if isinstance(ep_length, np.ndarray) else ep_length
                            
                            # Ensure they are not None again after .item() (if original was an empty array, though unlikely)
                            # and ensure correct type for formatting
                            if scalar_display_return is not None and scalar_display_length is not None:
                                pbar.set_postfix({
                                    "last_ep_return": f"{float(scalar_display_return):.2f}",
                                    "last_ep_length": int(scalar_display_length)
                                })

                        if args.capture_video and i == 0: 
                            # Also ensure scalar for this print statement
                            return_str_for_video = "N/A"
                            if ep_return is not None:
                                scalar_return_video = ep_return.item() if isinstance(ep_return, np.ndarray) else ep_return
                                if scalar_return_video is not None: # Check again after potential .item()
                                    return_str_for_video = f"{float(scalar_return_video):.2f}"
                            print(f"Collected episode {episodes_collected_count}/{args.num_expert_episodes}. Env 0 Return: {return_str_for_video}")

                    episode_observations[i] = []
                    episode_actions[i] = []
                    episode_rewards[i] = []
                    episode_next_observations[i] = []
                    episode_dones[i] = []
            
            if episodes_collected_count >= args.num_expert_episodes:
                break 
            
            next_obs_np = next_obs_after_step_np
            next_obs_tensor = torch.Tensor(next_obs_np).to(device)

    # Determine output directory based on model type
    base_output_dir = f"trajectories/{args.model_type}"
    os.makedirs(base_output_dir, exist_ok=True) # Ensure the directory exists

    if args.output_file_name:
        # If a specific output file name is given, use it as the file name part
        file_name_part = args.output_file_name
        # Ensure it has a .pkl extension, add if missing
        if not file_name_part.lower().endswith(".pkl"):
            file_name_part += ".pkl"
    else:
        # Generate a default file name
        file_name_part = f"{args.env_id.replace('/', '-')}__{args.exp_name}__{args.model_type}__expert_trajectories_{args.num_expert_episodes}eps.pkl"
    
    output_filename = os.path.join(base_output_dir, file_name_part)
    
    with open(output_filename, "wb") as f:
        pickle.dump(expert_trajectories_data, f)
    print(f"\nSaved {len(expert_trajectories_data)} expert trajectories to {output_filename}")

    envs.close()
    if args.track and "wandb" in locals() and wandb.run: # Check if wandb was imported and run exists
        wandb.finish()