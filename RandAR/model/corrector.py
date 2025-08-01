import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Callable

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

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach().clone()
        else:
            b = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`list[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        in_features = out_features = hidden_size
        hidden_features = int(hidden_size * mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)

# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
# with ViT->Dinov2
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Dinov2
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob, qkv_bias: bool = True) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0, f"The hidden size {hidden_size} is not a multiple of the number of attention heads {num_attention_heads}."

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_interface: Callable = eager_attention_forward


        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Dinov2
class SelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, hidden_size, dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class Attention(nn.Module):
    def __init__(self, hidden_size, dropout_prob, num_attention_heads, qkv_bias: bool = True) -> None:
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads, dropout_prob, qkv_bias)
        self.output = SelfOutput(hidden_size, dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class LayerScale(nn.Module):
    def __init__(self, layerscale_value, hidden_size) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1

def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

class CorrectorLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, hidden_size, mlp_ratio, layer_norm_eps, drop_path_rate, layer_scale_value, dropout_prob, num_attention_heads, qkv_bias: bool = True) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = Attention(hidden_size, dropout_prob, num_attention_heads, qkv_bias)
        self.layer_scale1 = LayerScale(layer_scale_value, hidden_size)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.mlp = SwiGLUFFN(hidden_size, mlp_ratio)
        self.layer_scale2 = LayerScale(layer_scale_value, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output

class TransformerCorrector(nn.Module):
    def __init__(self,
                 dim: int = 1024,
                 num_ar_layers_for_input: int = 12,
                 transformer_dim: int = 512,
                 num_transformer_layers: int = 6,
                 mlp_ratio: float = 4.0,
                 num_attention_heads: int = 8,
                 layer_norm_eps: float = 1e-5,
                 drop_path_rate: float = 0.0,
                 layer_scale_value: float = 1.0,
                 dropout: float = 0.1,
                 qkv_bias: bool = True,
                 *args, **kwargs):
        super().__init__()
        
        input_dim = dim * num_ar_layers_for_input

        # create input projection layer
        self.input_projection = nn.Linear(input_dim, transformer_dim)
        
        # Create transformer encoder layers
        self.transformer = nn.Sequential(*[
            CorrectorLayer(
                hidden_size=transformer_dim,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps,
                drop_path_rate=drop_path_rate,
                layer_scale_value=layer_scale_value,
                dropout_prob=dropout,
                num_attention_heads=num_attention_heads,
                qkv_bias=qkv_bias
            )
            for _ in range(num_transformer_layers)
        ])

        # create projection layer
        self.output_projection = nn.Linear(transformer_dim, 1)
        

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
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_projection(x)
        
        return x