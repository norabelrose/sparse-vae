from .Attention import *
from functools import lru_cache


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = False, cross_attention: bool = False,
                 sparse_self_attention: bool = False):
        super(TransformerLayer, self).__init__()

        self.attention = Attention(d_model, num_heads, causal, sparse=sparse_self_attention)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)  # Superfluous since this is followed by layer norm
        )
        self.dropout = nn.Dropout(p=0.1)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.cross_attention = Attention(d_model, num_heads) if cross_attention else None
        self.cross_attn_layer_norm = nn.LayerNorm(d_model) if cross_attention else None

    def forward(self, x: Tensor, context: Tensor = None, mask: Tensor = None, context_mask: Tensor = None) -> Tensor:
        q = k = x + positional_encodings_like(x)
        y = self.attention(q, k, x, mask=mask)
        x = y = self.attn_layer_norm(x + y)

        if self.cross_attention and context is not None:
            context_k = context + positional_encodings_like(context)
            x = self.cross_attention(q, context_k, context, mask=context_mask)
            x = y = y + self.cross_attn_layer_norm(x)

        y = self.ffn(y)
        y = self.dropout(y)
        y = self.ffn_layer_norm(x + y)

        return y


def positional_encodings_like(x: Tensor):
    return get_positional_encodings(x.shape[-2], x.shape[-1], x.device, x.dtype)

@lru_cache(maxsize=10)
def get_positional_encodings(seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    d_model_half = d_model // 2
    frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
    periods = 1 / (10000 ** (frequencies / d_model_half))

    position_ids = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
    angles = position_ids[:, None] * periods[None]  # noqa; Outer product

    encodings = torch.empty(seq_len, d_model, dtype=dtype, device=device)
    encodings[:, ::2] = angles.sin()
    encodings[:, 1::2] = angles.cos()
    return encodings
