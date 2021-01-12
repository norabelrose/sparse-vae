from .FunnelTransformer import *
import numpy as np


class FunnelWithDecoder(nn.Module):
    def __init__(self, hparams: AttributeDict, num_decoder_layers: int):
        super().__init__()

        # Make sure the first block's output is returned from the encoder so that we can
        # use it in the residual connection for the decoder
        if not hparams.return_block_outputs:
            hparams.return_block_outputs = [0]
        elif 0 not in hparams.return_block_outputs:
            hparams.return_block_outputs = [0] + hparams.return_block_outputs

        self.encoder = FunnelTransformer(hparams)
        self.decoder = FunnelBlock(hparams, num_decoder_layers)

    def forward(self, x: Tensor, input_mask: Tensor = None) -> Dict[str, Any]:
        result = self.encoder(x, input_mask)

        # Residual connection
        total_scaling = np.prod(self.encoder.hparams.scaling_factors)
        scaled_output = result['output'].repeat_interleave(total_scaling, dim=-2)
        decoder_input = scaled_output + result['hidden_states'][0]

        attn_state = self.encoder.attention_state
        attn_state.current_block = 0  # For the decoder, rewind to the stride that we were at in the first block

        result['output'] = self.decoder(decoder_input, decoder_input, attn_state)[1]
        attn_state.reset()
        return result
