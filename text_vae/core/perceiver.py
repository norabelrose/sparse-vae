from .transformer_layer import *


class Perceiver(nn.Module):
    def __init__(self, num_layers: int, num_latents: int, d_model: int, bottleneck_width: Optional[int] = None):
        super().__init__()
        assert num_layers > 1

        self.first_layer = TransformerLayer(d_model, d_model // 64, learned_queries=num_latents)
        if bottleneck_width:
            self.bottleneck = TransformerLayer(
                d_model, d_model // 64, learned_queries=bottleneck_width
            )
            num_layers -= 1
        else:
            self.bottleneck = None

        self.middle_layers = nn.ModuleList([
            TransformerLayer(d_model, d_model // 64, use_cross_attention=True)
            for _ in range(num_layers - 1)
        ])

    def forward(self, x: Tensor):
        z = self.first_layer(x)

        for layer in self.middle_layers:
            z = layer(z, context=x)

        if self.bottleneck:
            z = self.bottleneck(z)

        return z
