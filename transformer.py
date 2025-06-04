import numpy as np
import torch

from einops import rearrange
from torch import nn

def create_env(config:dict, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        env_name {str}: Name of the to be instantiated environment
        render {bool}: Whether to instantiate the environment in render mode. (default: {False})

    Returns:
        {env}: Returns the selected environment instance.
    """
    if config["type"] == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True, max_episode_steps=32)
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if config["type"] == "Minigrid":
        return Minigrid(config["name"])
    if config["type"] in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
        return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode=render)

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)
    
def batched_index_select(input, dim, index):
    """
    Selects values from the input tensor at the given indices along the given dimension.
    This function is similar to torch.index_select, but it supports batched indices.
    The input tensor is expected to be of shape (batch_size, ...), where ... means any number of additional dimensions.
    The indices tensor is expected to be of shape (batch_size, num_indices), where num_indices is the number of indices to select for each element in the batch.
    The output tensor is of shape (batch_size, num_indices, ...), where ... means any number of additional dimensions that were present in the input tensor.

    Arguments:
        input {torch.tensor} -- Input tensor
        dim {int} -- Dimension along which to select values
        index {torch.tensor} -- Tensor containing the indices to select

    Returns:
        {torch.tensor} -- Output tensor
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def process_episode_info(episode_info:list) -> dict:
    """Extracts the mean and std of completed episode statistics like length and total reward.

    Arguments:
        episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

    Returns:
        {dict} -- Processed episode results (computes the mean and std for most available keys)
    """
    result = {}
    if len(episode_info) > 0:
        for key in episode_info[0].keys():
            if key == "success":
                # This concerns the PocMemoryEnv only
                episode_result = [info[key] for info in episode_info]
                result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
            result[key + "_mean"] = np.mean([info[key] for info in episode_info])
            result[key + "_std"] = np.std([info[key] for info in episode_info])
    return result

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.

        Returns:
            {float}: Returns the norm of the gradients of this model's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.

        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, queries, mask):
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            queries {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = self.values(values)    # (N, value_len, embed_dim)
        keys = self.keys(keys)          # (N, key_len, embed_dim)
        queries = self.queries(queries) # (N, query_len, embed_dim)
        
        values = values.reshape(N, value_len, self.num_heads, self.head_size)   # (N, value_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)         # (N, key_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_size) # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, config):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            config {dict} -- General config
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Setup GTrXL if used
        self.use_gtrxl = config["gtrxl"] if "gtrxl" in config else False
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, config["gtrxl_bias"])
            self.gate2 = GRUGate(embed_dim, config["gtrxl_bias"])

        # LayerNorms
        self.layer_norm = config["layer_norm"]
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            h = self.gate1(query, attention)
        else:
            # Skip connection
            h = attention + query
        
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            out = self.gate2(h, forward)
        else:
            # Skip connection
            out = forward + h
        
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)

        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1., device=self.inv_freqs.device)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, max_episode_steps) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.
        Arguments:
            config {dict} -- Transformer config
            input_dim {int} -- Dimension of the input
            max_episode_steps {int} -- Maximum number of steps in an episode
        """
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.max_episode_steps = max_episode_steps
        self.activation = nn.ReLU()

        # Input embedding layer
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Determine positional encoding
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim))
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])
        
        # Store last hidden state for compatibility with TTTModel
        self.last_hidden_state = None

    def forward(self, x):
        """
        Arguments:
            x {torch.tensor} -- Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
        """
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(x))
        
        # Create attention mask (all ones since we want to attend to all positions)
        batch_size, seq_len = x.shape[0], x.shape[1]
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Add positional encoding if configured
        if self.config["positional_encoding"] == "relative":
            pos_embedding = self.pos_embedding(seq_len).to(x.device)
            h = h + pos_embedding.unsqueeze(0)  # Add batch dimension
        elif self.config["positional_encoding"] == "learned":
            h = h + self.pos_embedding[:seq_len].unsqueeze(0).to(x.device)  # Add batch dimension and move to device
        
        # Forward transformer blocks
        for block in self.transformer_blocks:
            h, _ = block(h, h, h, mask)  # Self-attention: value=key=query=h
        
        # Store last hidden state for compatibility with TTTModel
        self.last_hidden_state = h
        
        return h

class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)
