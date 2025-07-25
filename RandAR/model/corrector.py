import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class LinearCorrector(nn.Module):
    def __init__(self, 
                 dim: int = 1024,
                 num_ar_layers_for_input: int = 12,
                 *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(dim * num_ar_layers_for_input, 1)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Args:
            x (torch.Tensor): The input tensor of shape `[batch_size, seq_len, dim]`. 
                                This is a concatenation of the class embedding and the interleaved
                                positional and token embeddings. `seq_len` is typically `1 + 2 * block_size`.
            freqs_cis (torch.Tensor): The pre-computed Rotary Positional Embeddings (RoPE)
                                        of shape `[batch_size, seq_len, 1, dim // n_head]`.
        Returns:
            torch.Tensor: The output logits of shape `[batch_size, seq_len, 1]`.
        """
        return self.linear(x)

class MLPCorrector(nn.Module):
    def __init__(self,
                 dim: int = 1024,
                 hidden_dim: Optional[List[int]] = None,
                 num_ar_layers_for_input: int = 12,
                 *args, **kwargs):
        super().__init__()
        if hidden_dim is None:
            self.mlp = nn.Linear(dim * num_ar_layers_for_input, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim * num_ar_layers_for_input, hidden_dim[0]),
                nn.GELU(),
                *[nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                    nn.GELU(),
                ) for i in range(len(hidden_dim) - 1)],
                nn.Linear(hidden_dim[-1], 1)
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Args:
            x (torch.Tensor): The input tensor of shape `[batch_size, seq_len, dim]`. 
                                This is a concatenation of the class embedding and the interleaved
                                positional and token embeddings. `seq_len` is typically `1 + 2 * block_size`.
            freqs_cis (torch.Tensor): The pre-computed Rotary Positional Embeddings (RoPE)
                                        of shape `[batch_size, seq_len, 1, dim // n_head]`.
        Returns:
            torch.Tensor: The output logits of shape `[batch_size, seq_len, 1]`.
        """
        return self.mlp(x)

class TransformerCorrector(nn.Module):
    def __init__(self,
                 dim: int = 1024,
                 num_ar_layers_for_input: int = 12,
                 num_transformer_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 *args, **kwargs):
        super().__init__()
        
        dim = dim * num_ar_layers_for_input
        
        self.dim = dim
        self.num_transformer_layers = num_transformer_layers
        
        # Default feed-forward dimension to 4x the model dimension
        if ff_dim is None:
            ff_dim = 4 * dim
        
        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(dim)
        
        # Final projection layer to output logits
        self.output_projection = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Args:
            x (torch.Tensor): The input tensor of shape `[batch_size, seq_len, dim]`. 
                                This is a concatenation of the class embedding and the interleaved
                                positional and token embeddings. `seq_len` is typically `1 + 2 * block_size`.
            freqs_cis (torch.Tensor): The pre-computed Rotary Positional Embeddings (RoPE)
                                        of shape `[batch_size, seq_len, 1, dim // n_head]`.
        Returns:
            torch.Tensor: The output logits of shape `[batch_size, seq_len, 1]`.
        """
        # Apply transformer encoder (bidirectional attention)
        # No attention mask is provided, so all positions can attend to all other positions
        transformed = self.transformer(x)
        
        # Apply layer normalization
        normalized = self.layer_norm(transformed)
        
        # Project to output dimension
        logits = self.output_projection(normalized)
        
        return logits