from .attention import *


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = False,
        use_cross_attention: bool = False,
        sparse_self_attention: Union[bool, SlidingWindowSparsityConfig] = False,
        learned_queries: int = None,
    ):
        super(TransformerLayer, self).__init__()

        self.attention = Attention(d_model, num_heads, causal, sparse=sparse_self_attention,
                                   learned_queries=learned_queries)
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
        if value:
            self_attn = self.attention
            self.cross_attention = Attention(
                d_model=self_attn.d_model,
                num_heads=self_attn.num_heads
            )
            self.cross_attn_layer_norm = nn.LayerNorm(self_attn.d_model)
            self.context_layer_norm = nn.LayerNorm(self_attn.d_model)
        else:
            self.cross_attention = None

    def forward(self, x: PaddedTensor, context: PaddedTensor = None) -> Tensor:
        # We use the pre-LayerNorm variant of the Transformer architecture because it tends to train more
        # stably across a wider range of hyperparameter configurations
        y = self.attn_layer_norm(x)
        y = self.attention(y, y, y)
        x = x + y if x.shape == y.shape else y

        if self.cross_attention and context is not None:
            context, y = self.context_layer_norm(context), self.cross_attn_layer_norm(x)
            y = self.cross_attention(y, context, context)
            x = x + y

        y = self.ffn_layer_norm(x)
        y = self.ffn(y)
        y = self.dropout(y)
        y = x + y

        return y
