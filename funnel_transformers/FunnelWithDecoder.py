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

        # AttentionState needs to know that the decoder exists
        self.encoder.attention_state.has_decoder_block = True

    def forward(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None) -> Dict[str, Any]:
        result = self.encoder(x, input_mask, seg_id)

        # Residual connection
        total_scaling = np.prod(self.encoder.hparams.scaling_factors)
        scaled_output = result['output'].repeat_interleave(total_scaling, dim=-2)
        decoder_input = scaled_output + result['hidden_states'][0]

        result['output'] = self.decoder(decoder_input, decoder_input, self.encoder.attention_state)[1]
        return result
