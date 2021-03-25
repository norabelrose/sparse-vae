from .CategoricalMixture import *
from .HierarchicalAutoencoder import *
from .core import PaddedTensor, Quantizer
from .train_callbacks import ReconstructionSampler
from itertools import islice
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.decorators import auto_move_data
from torchtext.data.metrics import bleu_score
from tqdm import tqdm


@dataclass
class QuantizedVAEHparams(FunnelAutoencoderHparams):
    beta: float = 0.25
    codebook_size: int = 8192
    ema_decay: float = 0.99
    use_categorical_mixture: bool = False
    use_kmeans_codebook_updates: bool = False

    grad_clip_threshold: float = 25.0
    latent_depth: int = 64
    lr: float = 3e-4  # From the OpenAI Jukebox VQ-VAE hyperparameters


@dataclass
class QuantizedVAEState:
    code_indices: List[Tensor] = field(default_factory=list)
    soft_codes: List[Tensor] = field(default_factory=list)

    ground_truth: Optional[Tensor] = None
    decoder_input: Optional[Tensor] = None
    encoder_states: List[PaddedTensor] = field(default_factory=list)

    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0
    logits: Optional[Tensor] = None


class QuantizedVAE(FunnelAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(QuantizedVAE, self).__init__(hparams)

        encoder_hparams = hparams.funnel
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
        self.gathering_latents = False

        for block, length in enumerate(encoder_hparams.block_sizes[-2:0:-1], start=1):
            self.decoder.layer_with_indices(block, length - 2).use_cross_attention = True
            self.decoder.layer_with_indices(block, length - 1).use_cross_attention = True

        self.initialize_weights()

    def setup_output_layer(self, hparams: DictConfig):
        # last_scaling_factor = hparams.funnel.scaling_factors[0]
        if not hparams.use_categorical_mixture:
            super(QuantizedVAE, self).setup_output_layer(hparams)
        else:
            self.output_layer = ConditionalCategoricalMixture(
                num_mixtures=2,
                num_features=hparams.funnel.d_model,
                num_classes=hparams.vocab_size
            )
            self.output_layer.embedding = self.encoder.input_layer[0].weight

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [EarlyStopping(monitor='val_nll', mode='min'), ReconstructionSampler()]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        pbar_dict = super().get_progress_bar_dict()
        del pbar_dict['loss']  # We don't want to include the commitment and embedding losses in the progress bar
        return pbar_dict

    def configure_optimizers(self, **kwargs):
        noncodebook_params = list(filter(lambda x: x is not self.quantizer.codebook, self.parameters()))

        # Use a higher learning rate for the codebook so that 'dead' codes move quickly toward the
        # encoder outputs and get revived
        return super().configure_optimizers(params=[
            {'params': [self.quantizer.codebook], 'lr': 10.0 * self.hparams.lr},
            {'params': noncodebook_params}
        ])

    def on_train_epoch_start(self):
        if self.hparams.use_kmeans_codebook_updates:
            self.update_codebook_kmeans()

        self.quantizer.reset_code_usage_info()

    # quantize = False is used by update_codebook_kmeans()
    def forward(self, batch: Dict[str, Tensor], quantize: bool = True) -> QuantizedVAEState:
        encoder_out = self.encoder_forward(batch)
        mask = self.encoder.attention_state.padding_mask_with_length(encoder_out.final_state.shape[-2])
        quant_out = self.quantizer(encoder_out.final_state, level=0, quantize=quantize, mask=mask)

        vae_state = QuantizedVAEState()
        vae_state.commitment_loss = quant_out.commitment_loss
        vae_state.embedding_loss = quant_out.embedding_loss

        vae_state.ground_truth = encoder_out.original_ids
        vae_state.decoder_input = self.quantizer.upsample_codes(quant_out.hard_codes, level=0)
        vae_state.encoder_states = encoder_out.hidden_states[-2:0:-1]
        vae_state.code_indices = [quant_out.code_indices]
        vae_state.soft_codes = [quant_out.soft_codes]

        return self.decoder_forward(vae_state)

    def decoder_forward(self, vae_state: QuantizedVAEState) -> QuantizedVAEState:
        self.decoder.attention_state.upsampling = True

        coroutine = self.decoder.forward_coroutine(vae_state.decoder_input)
        enc_state_iter = iter(vae_state.encoder_states)
        last_dec_state = None

        for block_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                vae_state.decoder_output = decoder_state
                vae_state.logits = self.output_layer(decoder_state.final_state).logits

                return vae_state

            encoder_state = next(enc_state_iter, None)
            if encoder_state is None:
                continue

            dec_state = self.decoder_block_end(vae_state, decoder_state, encoder_state, block_idx)
            if dec_state is not None:
                if block_idx == 0:
                    coroutine.send(dec_state)
                elif block_idx == 1:
                    coroutine.send((dec_state, vae_state.decoder_input))
                else:
                    coroutine.send((dec_state, last_dec_state))

                last_dec_state = dec_state

            # Abort the forward pass. Used when gathering latent codes for training the prior model. After we get
            # the bottom latent feature map we don't actually need to run the last few layers.
            else:
                return vae_state

    def decoder_block_end(self, vae_state: QuantizedVAEState, dec_state: Tensor, enc_state: PaddedTensor, block_idx: int):
        enc_data = enc_state.data
        if dec_state.shape != enc_data.shape:
            return self.quantizer.upsample_codes(enc_data, level=block_idx + 1)

        # It might be better to use cross attention here
        x = self.combiners[block_idx](torch.cat([dec_state, enc_data], dim=-1))
        quantizer_out = self.quantizer(x, level=block_idx + 1, quantize=True, mask=enc_state.padding)
        vae_state.code_indices.append(quantizer_out.code_indices)

        # End the forward pass early if we're gathering latent codes and we just computed the bottom latent feature map
        if self.gathering_latents and block_idx + 2 == self.quantizer.num_levels:
            return None

        # Don't use += operator to avoid autograd errors from doing an in-place operation- weirdly
        # this only seems to happen when we train with full precision
        old_commit_loss, old_embed_loss = vae_state.commitment_loss, vae_state.embedding_loss
        vae_state.commitment_loss = old_commit_loss + quantizer_out.commitment_loss
        vae_state.embedding_loss = old_embed_loss + quantizer_out.embedding_loss
        vae_state.soft_codes.append(quantizer_out.soft_codes)

        return self.quantizer.upsample_codes(quantizer_out.hard_codes, level=block_idx + 1)

    def train_or_val_step(self, batch: Dict[str, Tensor], stage: str) -> Dict[str, Tensor]:
        vae_state = self.forward(batch)

        # Convert these sums into means over codebooks. Within each codebook we compute the commitment and
        # embedding losses as the *mean* squared distance between the encoder output and the corresponding code in the
        # codebook, and so in order to be consistent we need to mean across codebooks as well.
        commitment_loss = vae_state.commitment_loss / self.quantizer.num_levels
        embedding_loss = vae_state.embedding_loss / self.quantizer.num_levels

        nll = F.cross_entropy(vae_state.logits.flatten(0, 1), vae_state.ground_truth.flatten(), ignore_index=0)
        self.log('nll', nll, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        log_prefix = stage + '_'
        self.log_dict({
            log_prefix + 'nll': nll,
            log_prefix + 'commitment_loss': commitment_loss,
            log_prefix + 'embedding_loss': embedding_loss
        })
        self.log_dict(self.quantizer.used_code_info_dict())

        logits = vae_state.logits.detach()
        if stage == 'val':
            # BLEU is expensive to compute so we only do it during validation
            reconstructions = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
            reconstructed_words = [sample.split() for sample in reconstructions]

            ground_truth = self.tokenizer.decode_batch(vae_state.ground_truth.tolist())
            real_words = [[sample.split()] for sample in ground_truth]

            self.log('val_bleu', bleu_score(reconstructed_words, real_words))

        elif stage == 'train':
            return {
                'logits': logits,
                'loss': nll + embedding_loss + self.hparams.beta * commitment_loss
            }

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        return self.train_or_val_step(batch, stage='train')

    def on_after_backward(self):
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        self.train_or_val_step(batch, stage='val')

    # Return the quantized latent feature maps only
    @auto_move_data
    def predict(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: Optional[int] = None):
        latent_codes = self.forward(batch).code_indices
        attn_state = self.decoder.attention_state

        result = {}
        for i, z in enumerate(latent_codes):
            level_name = self.name_for_latent_level(i, len(latent_codes))
            padding = attn_state.padding_mask_with_length(z.shape[-1])
            result[level_name] = {'data': z, 'padding': padding}

        return result

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None

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

    @staticmethod
    def name_for_latent_level(level: int, num_levels: int):
        assert num_levels > 1

        if num_levels > 3:
            return f'level {level}'
        if level == 0:
            return 'top'
        if level == num_levels - 1:
            return 'bottom'
        if level == 1:
            return 'middle'
