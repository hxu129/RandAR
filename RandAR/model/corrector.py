import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__():
        super().__init__()

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