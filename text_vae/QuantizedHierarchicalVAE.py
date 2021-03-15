from .HierarchicalAutoencoder import *
from .core import Quantizer, QuantizedVAE, QuantizedVAEHparams
from itertools import islice
from tqdm import tqdm


@dataclass
class QuantizedHierarchicalVAEHparams(HierarchicalAutoencoderHparams, QuantizedVAEHparams):
    # beta_annealing: bool = True
    grad_clip_threshold: float = 25.0
    ema_decay: float = 0.99


@dataclass
class QuantizerHierarchicalVAEState(HierarchicalAutoencoderState):
    soft_codes: List[Tensor] = field(default_factory=list)

    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0
    p_of_x_given_z: Optional[Categorical] = None


class QuantizedHierarchicalVAE(HierarchicalAutoencoder, QuantizedVAE):
    def __init__(self, hparams: DictConfig):
        super(QuantizedHierarchicalVAE, self).__init__(hparams)

        encoder_hparams = hparams.encoder
        d_model = encoder_hparams.d_model
        num_latent_scales = len(encoder_hparams.scaling_factors)

        # These combine the encoder and decoder states together right before quantization
        self.combiners = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(num_latent_scales - 1)
        ])
        self.quantizer = Quantizer(
            num_codes=hparams.codebook_size,
            code_depth=hparams.latent_depth,
            d_model=d_model,
            num_levels=num_latent_scales,
            ema_decay=hparams.ema_decay
        )

    # quantize = False is used by update_codebook_kmeans()
    def forward(self, batch: Dict[str, Tensor], quantize: bool = True) -> QuantizerHierarchicalVAEState:
        encoder_out = self.encoder_forward(batch)
        quant_out = self.quantizer(encoder_out.final_state, level=0, quantize=self.quantizing)

        vae_state = QuantizerHierarchicalVAEState()
        vae_state.commitment_loss = quant_out.commitment_loss
        vae_state.embedding_loss = quant_out.embedding_loss

        vae_state.ground_truth = encoder_out.original_ids
        vae_state.decoder_input = self.quantizer.upsample_codes(quant_out.hard_codes, level=0)
        vae_state.encoder_states = encoder_out.hidden_states[-2:0:-1]
        vae_state.soft_codes = [quant_out.soft_codes]
        vae_state = self.decoder_forward(vae_state, padding_mask=batch['padding_mask'])

        return vae_state

    def on_train_epoch_start(self):
        if self.hparams.use_kmeans_codebook_updates:
            self.update_codebook_kmeans()

        self.quantizer.reset_code_usage_info()

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        vae_state = self.forward(batch)

        # Convert these sums into means over codebooks. Within each codebook we compute the commitment and
        # embedding losses as the *mean* squared distance between the encoder output and the corresponding code in the
        # codebook, and so in order to be consistent we need to mean across codebooks as well.
        vae_state.commitment_loss /= self.quantizer.num_levels
        vae_state.embedding_loss /= self.quantizer.num_levels

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        if not self.hparams.include_padding_positions:
            original_ids = batch.get('labels', batch['token_ids'])
            # Ignore padding by setting its probability to 1.0 (log probability 0.0)
            log_probs = log_probs.where(original_ids.unsqueeze(-1) != 0, 0.0)

        nll = -log_probs.mean()

        log_prefix = kwargs.get('log_prefix') or 'train_'
        self.log_dict({
            log_prefix + 'nll': nll,
            log_prefix + 'commitment_loss': vae_state.commitment_loss,
            log_prefix + 'embedding_loss': vae_state.embedding_loss
        })
        self.log_dict(self.quantizer.used_code_info_dict())

        return {
            'logits': vae_state.p_of_x_given_z.logits.detach(),
            'loss': nll + vae_state.embedding_loss + self.hparams.beta * vae_state.commitment_loss
        }

    def on_after_backward(self):
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_index, log_prefix='val_')

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        # It might be better to use cross attention here
        x = self.combiners[block_idx](torch.cat([dec_state, enc_state], dim=-1))
        quantizer_out = self.quantizer(x, level=block_idx + 1, quantize=self.quantizing)
        vae_state.soft_codes.append(quantizer_out.soft_codes)

        vae_state.commitment_loss += quantizer_out.commitment_loss
        vae_state.embedding_loss += quantizer_out.embedding_loss

        return self.quantizer.upsample_codes(quantizer_out.hard_codes, level=block_idx + 1)

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None

    # @torch.cuda.amp.autocast()  # Manually activate AMP since PL won't do it for us here
    @torch.no_grad()
    def update_codebook_kmeans(self):
        self.print("\nPerforming K means codebook update...")

        # This turns off EMA updates in the quantizer
        was_training = self.quantizer.training
        if was_training:
            self.quantizer.eval()

        # Do encoder forward passes through the entire training dataset in order to gather the soft codes
        loader = self.trainer.train_dataloader

        min_code_count = self.hparams.codebook_size * 50  # Arbitrary
        observed_codes = []
        code_counts = []

        pbar = tqdm(desc='Gathering encoder outputs', total=min_code_count * self.quantizer.num_levels, unit=' codes')
        for batch in islice(loader, len(loader)):
            outputs = self.forward(
                # Annoyingly we have to do this device transfer manually since we're using the dataloader outside
                # of the usual PyTorch Lightning training hooks
                {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()},
                quantize=False
            ).soft_codes
            soft_codes = [output.half().flatten(end_dim=-2) for output in outputs]
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

        if was_training:
            self.quantizer.train()
