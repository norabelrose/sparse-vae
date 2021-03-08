from tqdm import tqdm
from .HierarchicalAutoencoder import *
from .core import Quantizer, QuantizerOutput, QuantizedVAE, QuantizedVAEHparams


@dataclass
class QuantizedHierarchicalVAEHparams(HierarchicalAutoencoderHparams, QuantizedVAEHparams):
    beta: float = 2.0
    grad_clip_threshold: float = 25.0
    lr: float = 1e-4


@dataclass
class QuantizerHierarchicalVAEState(HierarchicalAutoencoderState):
    latents: List[Tensor] = field(default_factory=list)

    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0
    p_of_x_given_z: Optional[Categorical] = None


class QuantizedHierarchicalVAE(HierarchicalAutoencoder, QuantizedVAE):
    def __init__(self, hparams: DictConfig):
        super(QuantizedHierarchicalVAE, self).__init__(hparams)

        encoder_hparams = hparams.encoder
        num_latent_scales = len(encoder_hparams.scaling_factors)
        self.quantizer = Quantizer(
            num_codes=hparams.codebook_size,
            code_depth=hparams.latent_depth,
            d_model=encoder_hparams.d_model,
            num_levels=num_latent_scales
        )

    # quantize = False is used by update_codebook_kmeans()
    def forward(self, batch: Dict[str, Tensor], quantize: bool = True) -> List[QuantizerOutput]:
        encoder_out = self.encoder_forward(batch)
        return [self.quantizer(z, quantize=quantize, level=lvl)
                for lvl, z in enumerate(reversed(encoder_out.hidden_states[1:]))]

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        # First epoch: just use the soft codes and train as a continuous, non-variational autoencoder
        encoder_out = self.encoder_forward(batch)
        quant_out = self.quantizer(encoder_out.final_state, level=0, quantize=self.quantizing)

        vae_state = QuantizerHierarchicalVAEState()
        vae_state.ground_truth = encoder_out.original_ids
        vae_state.decoder_input = self.quantizer.upsample_codes(quant_out.hard_codes, level=0)
        vae_state.encoder_states = encoder_out.hidden_states[-2:0:-1]
        vae_state = self.decoder_forward(vae_state, padding_mask=batch['padding_mask'])

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        if not self.hparams.include_padding_positions:
            original_ids = batch.get('labels', batch['token_ids'])
            # Ignore padding by setting its probability to 1.0 (log probability 0.0)
            log_probs = log_probs.where(original_ids.unsqueeze(-1) != 0, 0.0)

        nll = -log_probs.mean()

        log_prefix = kwargs.get('log_prefix') or 'train_'
        self.log_dict({
            log_prefix + 'nll': nll,
            log_prefix + 'commitment_loss': quant_out.commitment_loss
        })
        return {
            'logits': vae_state.p_of_x_given_z.logits,
            'loss': nll + quant_out.embedding_loss + self.hparams.beta * quant_out.commitment_loss
        }

    def on_after_backward(self):
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_index, log_prefix='val_')

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        # We're gonna try just adding together the encoder and decoder states here- it might be better
        # to use cross attention
        x = dec_state + enc_state
        quantizer_out = self.quantizer(x, level=block_idx + 1, quantize=self.quantizing)

        vae_state.commitment_loss += quantizer_out.commitment_loss
        vae_state.embedding_loss += quantizer_out.embedding_loss

        return self.quantizer.upsample_codes(quantizer_out.hard_codes, level=block_idx + 1)

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None

    @torch.cuda.amp.autocast()  # Manually activate AMP since PL won't do it for us here
    @torch.no_grad()
    def update_codebook_kmeans(self):
        self.print("\nPerforming K means codebook update...")

        # Do encoder forward passes through the entire training dataset in order to gather the soft codes
        loader = self.trainer.train_dataloader

        min_code_count = self.hparams.codebook_size * 1000  # Arbitrary
        observed_codes = []
        code_counts = []

        pbar = tqdm(desc='Gathering encoder outputs', total=min_code_count * self.quantizer.num_levels, unit=' codes')
        for batch in islice(loader, len(loader)):
            outputs = self.forward(
                # Annoyingly we have to do this device transfer manually since we're using the dataloader outside
                # of the usual PyTorch Lightning training hooks
                {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()},
                quantize=False
            )
            soft_codes = [output.soft_codes.flatten(end_dim=-2) for output in outputs]
            counts = [codes.shape[0] for codes in soft_codes]

            # Keep track of the total number of soft codes we've collected for each latent scale so far
            if code_counts:
                code_counts = [running_total + count for running_total, count in zip(code_counts, counts)]
            else:
                code_counts = counts

            codes_added = 0
            if not observed_codes:
                # Wrap each soft code tensor in a list so we can append to it
                observed_codes = [[codes] for codes in soft_codes]
                codes_added = sum(counts)
            else:
                # For each latent scale, check if we already hit our minimum number of latent code vectors. If we have,
                # we don't add any more to the list for that particular scale.
                for i, (count, codes) in enumerate(zip(code_counts, soft_codes)):
                    if count < min_code_count:
                        observed_codes[i] += [codes]
                        codes_added += codes.shape[0]

                # We hit the minimum number of codes we need for all the latent scales, so we don't need
                # to gather any more. Doing so will probably just make us OOM when we try to run K means anyway.
                if not codes_added:
                    break

            pbar.update(codes_added)
        pbar.close()

        # List of list of tensors -> list of tensors
        observed_codes = [torch.cat(codes, dim=0) for codes in observed_codes]
        self.quantizer.perform_kmeans_update(observed_codes)
