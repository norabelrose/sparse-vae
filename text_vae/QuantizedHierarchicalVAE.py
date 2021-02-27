from .HierarchicalAutoencoder import *
from .core import Quantizer
from torch.distributions import Categorical


@dataclass
class QuantizedHierarchicalVAEHparams(HierarchicalAutoencoderHparams):
    codebook_size: int = 512


@dataclass
class QuantizerHierarchicalVAEState(HierarchicalAutoencoderState):
    latents: List[Tensor] = field(default_factory=list)

    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0
    p_of_x_given_z: Optional[Categorical] = None


class QuantizedHierarchicalVAE(HierarchicalAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(QuantizedHierarchicalVAE, self).__init__(hparams)

        encoder_hparams = hparams.encoder
        num_latent_scales = len(encoder_hparams.scaling_factors)
        self.quantizer = Quantizer(hparams.codebook_size, encoder_hparams.d_model, num_levels=num_latent_scales)

    @property
    def quantizing(self) -> bool:
        return not self.training or self.trainer.current_epoch > 0

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        # First epoch: just use the soft codes and train as a continuous, non-variational autoencoder
        encoder_out = self.encoder_forward(batch)
        quantizer_out = self.quantizer(encoder_out.final_state, level=0, quantize=self.quantizing)

        vae_state = QuantizerHierarchicalVAEState()
        vae_state.ground_truth = encoder_out.original_ids
        vae_state.decoder_input = self.quantizer.upsample_codes(quantizer_out.hard_codes, level=0)
        vae_state.encoder_states = encoder_out.hidden_states[-2::-1]
        vae_state = self.decoder_forward(vae_state, padding_mask=batch['padding_mask'])

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        if not self.hparams.include_padding_positions:
            original_ids = batch.get('labels', batch['token_ids'])
            # Ignore padding by setting its probability to 1.0 (log probability 0.0)
            log_probs = log_probs.where(original_ids.unsqueeze(-1) != 0, 0.0)

        vae_state.stats['nll'] = -log_probs.flatten(start_dim=1).sum(dim=-1)
        return vae_state

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        # We're gonna try just adding together the encoder and decoder states here- it might be better
        # to use cross attention
        x = dec_state + enc_state
        quantizer_out = self.quantizer(x, level=block_idx + 1, quantize=self.quantizing)

        vae_state.commitment_loss += quantizer_out.commitment_loss
        vae_state.embedding_loss += quantizer_out.embedding_loss

        return quantizer_out.hard_codes

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None
