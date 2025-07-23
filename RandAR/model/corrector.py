import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class LinearCorrector(nn.Module):
    def __init__(self, 
                 dim: int = 1024):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    
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
                 hidden_dim: Optional[List[int]] = None):
        super().__init__()
        if hidden_dim is None:
            self.mlp = nn.Linear(dim, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim[0]),
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
        # TODO: Implement the forward pass logic
        pass