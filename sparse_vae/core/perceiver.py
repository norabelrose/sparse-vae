from .transformer_layer import *


class Perceiver(nn.Module):
    def __init__(
        self, num_layers: int, num_latents: int, d_model: int, bottleneck_width: Optional[int] = None,
        self_attention_layers: int = 1
    ):
        super().__init__()

        assert num_layers > 1
        num_heads = d_model // 64

        self.first_layer = TransformerLayer(d_model, num_heads, learned_queries=num_latents)
        if bottleneck_width:
            self.bottleneck = TransformerLayer(
                d_model, num_heads, learned_queries=bottleneck_width
            )
            num_layers -= 1
        else:
            self.bottleneck = None

        self.middle_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, use_cross_attention=True)
            for _ in range(num_layers - 1)
        ])
        # self.self_attention = nn.Sequential(*[
        #     TransformerLayer(
        #         d_model, num_heads,
        #         sparse_self_attention=SlidingWindowSparsityConfig(
        #             num_heads=num_heads, window_size=4, include_cls=False, causal=False
        #         )
        #     )
        #     for _ in range(self_attention_layers)
        # ])

    def forward(self, x: Tensor):
        # if self.self_attention:
        #     x = self.self_attention(x)

        z = self.first_layer(x)
        for layer in self.middle_layers:
            z = layer(z, context=x)

        if self.bottleneck:
            z = self.bottleneck(z)

        return z
