from .FunnelTransformer import *
import numpy as np


class FunnelWithDecoder(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, OmegaConf], num_decoder_layers: int):
        super().__init__()

        # Make sure the first block's output is returned from the encoder so that we can
        # use it in the residual connection for the decoder
        hparams.return_block_outputs = True

        self.encoder = FunnelTransformer(hparams)
        self.decoder = nn.Sequential(**[  # noqa
            FunnelLayer(hparams)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Dict[str, Any]:
        result = self.encoder({'input': x, 'padding_mask': padding_mask})

        # Residual connection
        total_scaling = np.prod(self.encoder.hparams.scaling_factors)
        scaled_output = result['output'].repeat_interleave(total_scaling, dim=-2)
        decoder_input = scaled_output + result['hidden_states'][0]

        attn_state = self.encoder.attention_state
        attn_state.current_block = 0  # For the decoder, rewind to the stride that we were at in the first block

        result['output'] = self.decoder({'q': decoder_input, 'kv': decoder_input, 'attn_state': attn_state})['q']
        attn_state.reset()
        return result
