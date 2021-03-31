from copy import deepcopy
from .attention import *


class TransformerLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            causal: bool = False,
            rel_pos_attn: bool = False,
            use_cross_attention: bool = False,
            sparse_self_attention: bool = False
    ):
        super(TransformerLayer, self).__init__()

        self.attention = Attention(d_model, num_heads, causal, rel_pos_attn=rel_pos_attn, sparse=sparse_self_attention)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)  # Superfluous since this is followed by layer norm
        )
        self.dropout = nn.Dropout(p=0.1)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.use_cross_attention = use_cross_attention

    @property
    def use_cross_attention(self):
        return bool(self.cross_attention)

    @use_cross_attention.setter
    def use_cross_attention(self, value: bool):
        self.cross_attention = deepcopy(self.attention) if value else None
        self.cross_attn_layer_norm = nn.LayerNorm(self.attention.d_model) if value else None

    def forward(self, x: PaddedTensor, context: PaddedTensor = None, cache_mask: Tensor = None, pos_enc = None) -> Tensor:
        y = self.attention(x, x, x, cache_mask=cache_mask, pos_enc=pos_enc)
        x = y = self.attn_layer_norm(x + y)

        if self.cross_attention and context is not None:
            x = self.cross_attention(x, context, context, pos_enc=pos_enc)
            x = y = self.cross_attn_layer_norm(x + y)

        y = self.ffn(y)
        y = self.dropout(y)
        y = self.ffn_layer_norm(x + y)

        return y
