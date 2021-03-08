from .FunnelTransformer import *
import torch.nn.functional as F


class FunnelWithDecoder(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, OmegaConf], num_decoder_layers: int):
        super().__init__()

        # Make sure the first block's output is returned from the encoder so that we can
        # use it in the residual connection for the decoder
        hparams.return_block_outputs = True

        self.encoder = FunnelTransformer(hparams)
        self.decoder = nn.ModuleList([
            FunnelLayer(hparams)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor = None,
                segment_ids: Optional[Tensor] = None) -> FunnelTransformerOutput:
        result = self.encoder(x, padding_mask=padding_mask, segment_ids=segment_ids)

        # Residual connection
        output = rearrange(result.final_state, "... l d -> ... d l")
        scaled_output = F.interpolate(output, size=x.shape[-1])
        scaled_output = rearrange(scaled_output, "... d l -> ... l d")

        decoder_input = scaled_output + result.hidden_states[0]

        attn_state = self.encoder.attention_state
        attn_state.current_block = 0  # For the decoder, rewind to the stride that we were at in the first block

        x = decoder_input
        for layer in self.decoder:
            x = layer(q=x, kv=x, attn_state=attn_state)

        result.final_state = x
        return result
