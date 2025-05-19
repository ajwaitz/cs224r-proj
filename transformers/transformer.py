import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input embedding
        self.embedding = nn.Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, d_model))  # 128 is max sequence length
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x) * (self.d_model ** 0.5)  # Scale embeddings
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Create causal mask to prevent attending to future positions
        if mask is None:
            # Create a mask that allows attention to current and past positions only
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
            mask = mask.unsqueeze(0).expand(x.size(0), -1, -1)

        # Apply transformer
        x = self.transformer(x, mask=mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x

    