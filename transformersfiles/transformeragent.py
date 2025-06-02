from utils import layer_init
from _transformers.transformer import Transformer
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

class TransformerAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n
        
        # Transformer configuration
        self.d_model = 128  # Transformer model dimension
        self.num_heads = 4  # Number of attention heads
        self.num_layers = 2  # Number of transformer layers
        
        # Transformer for processing observations
        self.transformer = Transformer(
            vocab_size=obs_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.d_model, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.d_model, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Process through transformer
        x = self.transformer(x)
        
        # Take the last sequence output for value prediction
        x = x[:, -1, :]
        
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Process through transformer
        x = self.transformer(x)
        
        # Take the last sequence output for action prediction
        x = x[:, -1, :]
        
        # Get action probabilities
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)