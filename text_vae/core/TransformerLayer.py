from .Attention import *


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
        y = self.attention(x, x, x, mask=mask)
        x = y = self.attn_layer_norm(x + y)

        if self.cross_attention and context is not None:
            x = self.cross_attention(x, context, context, mask=context_mask)
            x = y = self.cross_attn_layer_norm(x + y)

        y = self.ffn(y)
        y = self.dropout(y)
        y = self.ffn_layer_norm(x + y)

        return y
