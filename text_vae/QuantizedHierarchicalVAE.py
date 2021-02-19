from .HierarchicalVAE import *
from .core import Quantizer


@dataclass
class QuantizedHierarchicalVAEHparams(HierarchicalVAEHparams):
    codebook_size: int = 512


class QuantizedHierarchicalVAE(HierarchicalVAE):
    def __init__(self, hparams: DictConfig):
        super(QuantizedHierarchicalVAE, self).__init__(hparams)

        encoder_hparams = hparams.encoder
        num_latent_scales = len(encoder_hparams.scaling_factors)
        self.quantizer = Quantizer(hparams.codebook_size, encoder_hparams.d_model, num_levels=num_latent_scales)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        encoder_out = self.encoder_forward(batch)

        # First epoch: just use the soft codes and train as a continuous, non-variational autoencoder
        should_quantize = self.trainer.current_epoch > 0
        quantizer_out = self.quantizer(encoder_out.final_state, level=0, quantize=should_quantize)

        decoder_in = quantizer_out.hard_codes if should_quantize else quantizer_out.soft_codes
        encoder_out.final_state = self.quantizer.upsample_codes(decoder_in, level=0)

        vae_state = self.decoder_forward(decoder_in, padding_mask=batch['padding_mask'])

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        if not self.hparams.include_padding_positions:
            original_ids = batch.get('labels', batch['token_ids'])
            # Ignore padding by setting its probability to 1.0 (log probability 0.0)
            log_probs = log_probs.where(original_ids.unsqueeze(-1) != 0, 0.0)

        vae_state.stats['nll'] = -log_probs.flatten(start_dim=1).sum(dim=-1)
        return vae_state

    def decoder_forward(self, encoder_output: FunnelTransformerOutput, padding_mask: Tensor = None):
        self.decoder.attention_state.upsampling = True
        coroutine = self.decoder.forward_coroutine(
            encoder_output.final_state if encoder_output else None,
            padding_mask=padding_mask
        )

        for block_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                pass

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None
