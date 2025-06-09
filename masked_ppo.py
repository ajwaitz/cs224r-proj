# source: https://github.com/vwxyzjn/cleanrl/

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
# from gym_pomdp_wrappers import MuJoCoHistoryEnv

# import popgym # Not used in the final version of your script, can be removed if not needed
# from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction # Not used
# from popgym.core.observability import Observability, STATE # Not used

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
# from transformers.transformeragent import TransformerAgent # Ensure this is available if model_type="transformer"
# from utils import layer_init # Ensure utils.py with layer_init is in the same directory or accessible

# Placeholder for TransformerAgent if it's not defined elsewhere and you want to run with model_type="transformer"
# Replace with actual import if available
class TransformerAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Dummy implementation, replace with actual agent
        self.actor = nn.Linear(np.array(envs.single_observation_space.shape).prod(), envs.single_action_space.n)
        self.critic = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 1)
        print("Using Dummy TransformerAgent. Replace with actual implementation.")

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# Placeholder for layer_init if utils.py is not available
# Replace with actual import if available
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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1228800  # 16 workers * 256 steps * 300 updates
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16  # Match n_workers from transformer config
    """the number of parallel game environments"""
    num_steps: int = 256  # Match worker_steps from transformer config
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    model_type: str = "mlp"
    """the type of model parameterization to use"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    save_path: str = None
    """the path to save the model weights, if specified. if not, the model weights will not be saved."""


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
        obs_shape_prod = np.array(envs.single_observation_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape_prod, intermediate_size)),
            nn.Tanh(),
            layer_init(nn.Linear(intermediate_size, intermediate_size)),
            nn.Tanh(),
            layer_init(nn.Linear(intermediate_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape_prod, intermediate_size)),
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
    total_num_eps = 0
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.model_type}__{args.seed}__{int(time.time())}"
    
    episode_infos = deque(maxlen=100) 
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
    
    if envs.single_observation_space.shape != (4,):
            print(f"Warning: Observation masking is hardcoded for shape (4,). " 
                  f"Current env observation shape: {envs.single_observation_space.shape}. Masking may be incorrect or cause errors.")
    
    if args.model_type == "transformer":
        agent = TransformerAgent(envs).to(device) 
    elif args.model_type == "mlp":
        agent = MLPAgent(envs, intermediate_size=64).to(device)
        print('Using MLP agent...')
    else:
        print(f"Warning: model_type '{args.model_type}' not explicitly handled, defaulting to MLPAgent.")
        agent = MLPAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_storage_shape = (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    actions_storage_shape = (args.num_steps, args.num_envs) + envs.single_action_space.shape if hasattr(envs.single_action_space, 'shape') else (args.num_steps, args.num_envs)

    obs_tensor = torch.zeros(obs_storage_shape).to(device)
    actions_tensor = torch.zeros(actions_storage_shape).to(device) # Renamed to avoid conflict with function
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    
    reset_output = envs.reset(seed=args.seed)
    next_obs_np = reset_output[0] if isinstance(reset_output, tuple) else reset_output

    if isinstance(next_obs_np, np.ndarray) and next_obs_np.ndim > 1 and next_obs_np.shape[1] >= 4:
        next_obs_np[:, 1] = 0 
        next_obs_np[:, 3] = 0  
    elif isinstance(next_obs_np, np.ndarray) and next_obs_np.ndim == 1 and len(next_obs_np) >= 4: # Case for num_envs=1
        next_obs_np[1] = 0
        next_obs_np[3] = 0
    
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_tensor[step] = next_obs # Use renamed storage tensor
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions_tensor[step] = action # Use renamed storage tensor
            logprobs[step] = logprob

            step_results = envs.step(action.cpu().numpy())
            next_obs_np_step, reward_np, terminations_np, truncations_np, infos = step_results
            
            current_next_obs_np = next_obs_np_step # Use a temporary variable for masking
            if isinstance(current_next_obs_np, np.ndarray) and current_next_obs_np.ndim > 1 and current_next_obs_np.shape[1] >= 4:
                current_next_obs_np[:, 1] = 0
                current_next_obs_np[:, 3] = 0
            elif isinstance(current_next_obs_np, np.ndarray) and current_next_obs_np.ndim == 1 and len(current_next_obs_np) >=4: # Case for num_envs=1
                 current_next_obs_np[1] = 0
                 current_next_obs_np[3] = 0


            next_done_np = np.logical_or(terminations_np, truncations_np)
            rewards[step] = torch.tensor(reward_np).to(device).view(-1)
            next_obs, next_done = torch.Tensor(current_next_obs_np).to(device), torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for info_dict in infos["final_info"]:
                    if info_dict is not None and "episode" in info_dict:
                        episode_data = info_dict["episode"]
                        total_num_eps += 1
                        episode_infos.append(episode_data) # Store the whole episode dict

                        # Extract scalar values using .item()
                        episodic_return_scalar = episode_data["r"].item()
                        episodic_length_scalar = episode_data["l"].item()

                        # Calculate means and stds using .item() in list comprehension
                        all_returns = [ep_info["r"].item() for ep_info in episode_infos if "r" in ep_info]
                        all_lengths = [ep_info["l"].item() for ep_info in episode_infos if "l" in ep_info]
                        
                        # Handle cases where deque might be empty or contain partial data initially for std
                        reward_mean = np.mean(all_returns) if all_returns else 0.0
                        reward_std = np.std(all_returns) if len(all_returns) > 1 else 0.0
                        length_mean = np.mean(all_lengths) if all_lengths else 0.0
                        length_std = np.std(all_lengths) if len(all_lengths) > 1 else 0.0
                        
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={episodic_return_scalar:.2f}, "
                            f"reward_mean={reward_mean:.2f}, reward_std={reward_std:.2f}, "
                            f"length_mean={length_mean:.1f}, length_std={length_std:.2f}"
                        )
                        writer.add_scalar("charts/episodic_return", episodic_return_scalar, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length_scalar, global_step)
                        writer.add_scalar("charts/reward_mean", reward_mean, global_step)
                        writer.add_scalar("charts/reward_std", reward_std, global_step)
                        writer.add_scalar("charts/length_mean", length_mean, global_step)
                        writer.add_scalar("charts/length_std", length_std, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs_tensor.reshape((-1,) + envs.single_observation_space.shape) # Use renamed storage tensor
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions_tensor.reshape((-1,) + envs.single_action_space.shape if hasattr(envs.single_action_space, 'shape') else (-1,)) # Use renamed storage tensor
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                mb_b_actions = b_actions[mb_inds]
                # Ensure mb_b_actions is long if discrete, otherwise it might be float from continuous action spaces
                if envs.single_action_space.dtype == np.int64 or isinstance(envs.single_action_space, gym.spaces.Discrete):
                     mb_b_actions = mb_b_actions.long()

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], mb_b_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print('total num eps: ', total_num_eps)
    envs.close()
    writer.close()
    if args.save_path:
        torch.save(agent.state_dict(), args.save_path)
        print(f"Saved model weights to {args.save_path}")