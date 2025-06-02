import torch
import gymnasium as gym
from ttt import TTTModel, TTTConfig
from _transformers import LlamaForCausalLM, LlamaConfig

def build_mlp(input_size: int, hidden_size: int, output_size: int):
  return torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, output_size),
  )

# def build_transformer(input_size: int, hidden_size: int, output_size: int):

def get_model(model_type: str):
  if model_type == "ttt":
    return TTTModel(TTTConfig(
      vocab_size=256,
      hidden_size=128,
      num_hidden_layers=6,
      num_attention_heads=4,
    ))
  if model_type == "mlp":
    return build_mlp(input_size=2, hidden_size=128, output_size=2)
  else:
    raise ValueError(f"Model type {model_type} not supported")

def train():
  # set up env
  env = gym.make("LunarLander-v3", render_mode="human")
  observation, info = env.reset()

  # set up model
  model = get_model(model_type="mlp")

  # set up optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  # set up loss function
  loss_fn = torch.nn.CrossEntropyLoss()

  # set up training loop
  for episode in range(100):
    # reset env
    observation, info = env.reset()

    # set up episode
    episode_reward = 0
    episode_length = 0

    # run episode
    while True:
      # get action
      action = model.get_action(observation)

      # take action
      observation, reward, done, info = env.step(action)

      # update episode reward
      episode_reward += reward
      episode_length += 1

      
