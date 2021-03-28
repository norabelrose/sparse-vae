from .FunnelAutoencoder import *
from .core import PaddedTensor, Quantizer
from .train_callbacks import ReconstructionSampler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.decorators import auto_move_data
from torchtext.data.metrics import bleu_score


@dataclass
class QuantizedVAEHparams(FunnelAutoencoderHparams):
    beta: float = 0.25
    codebook_size: int = 1024
    ema_decay: float = 0.99
    include_full_res_latents: bool = False
    num_style_codes: int = 0

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

        num_latent_scales = len(hparams.funnel.scaling_factors)
        if hparams.include_full_res_latents:
            num_latent_scales += 1

        # These combine the encoder and decoder states together right before quantization
        d_model = hparams.funnel.d_model
        self.combiners = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(num_latent_scales - 1)
        ])
        self.quantizers = nn.ModuleList([
            Quantizer(
                num_codes=hparams.codebook_size,
                code_depth=hparams.latent_depth,
                d_model=d_model,
                ema_decay=hparams.ema_decay
            )
            for _ in range(num_latent_scales)
        ])
        self.gathering_latents = False

        if hparams.num_style_codes > 0:
            self.style_query = nn.Parameter(torch.randn(hparams.latent_depth))
            self.style_quantizer = Quantizer(
                num_codes=hparams.num_style_codes,
                code_depth=hparams.latent_depth,
                d_model=d_model,
                ema_decay=hparams.ema_decay
            )
        else:
            self.style_quantizer = None

        blocks = self.decoder.hparams.block_sizes[1:1 + num_latent_scales]
        self.decoder.configure_cross_attention([(block, 0) for block, length in enumerate(blocks, start=1)])
        self.initialize_weights()

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [EarlyStopping(monitor='val_nll', mode='min'), ReconstructionSampler()]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        pbar_dict = super().get_progress_bar_dict()
        del pbar_dict['loss']  # We don't want to include the commitment and embedding losses in the progress bar
        return pbar_dict

    def configure_optimizers(self, **kwargs):
        codebooks = set(x.codebook for x in self.quantizers)
        noncodebook_params = list(filter(lambda x: x not in codebooks, self.parameters()))

        # Use a higher learning rate for the codebook so that 'dead' codes move quickly toward the
        # encoder outputs and get revived
        return super().configure_optimizers(params=[
            {'params': list(codebooks), 'lr': 10.0 * self.hparams.lr},
            {'params': noncodebook_params}
        ])

    def on_train_epoch_start(self):
        for quantizer in self.quantizers:
            quantizer.reset_code_usage_info()

    # quantize = False is used by update_codebook_kmeans()
    def forward(self, batch: Dict[str, Tensor], quantize: bool = True) -> QuantizedVAEState:
        encoder_out = self.encoder_forward(batch)
        mask = self.encoder.attention_state.padding_mask_with_length(encoder_out.final_state.shape[-2])
        quant_out = self.quantizers[0](encoder_out.final_state, quantize=quantize, mask=mask)

        vae_state = QuantizedVAEState()
        vae_state.commitment_loss = quant_out.commitment_loss
        vae_state.embedding_loss = quant_out.embedding_loss

        encoder_states = encoder_out.hidden_states[:-1]
        encoder_states.reverse()
        if not self.hparams.include_full_res_latents:
            del encoder_states[-1]

        vae_state.ground_truth = encoder_out.original_ids
        vae_state.decoder_input = self.quantizers[0].upsample_codes(quant_out.hard_codes)
        vae_state.encoder_states = encoder_states
        vae_state.code_indices = [quant_out.code_indices]
        vae_state.soft_codes = [quant_out.soft_codes]

        return self.decoder_forward(vae_state)

    def decoder_forward(self, vae_state: QuantizedVAEState, mask: Tensor = None) -> QuantizedVAEState:
        self.decoder.attention_state.upsampling = True

        coroutine = self.decoder.forward_coroutine(vae_state.decoder_input, padding_mask=mask)
        enc_state_iter = iter(vae_state.encoder_states)
        last_dec_state = None

        for block_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                vae_state.decoder_output = decoder_state
                vae_state.logits = self.output_layer(decoder_state.final_state)

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
            return self.quantizers[block_idx + 1].upsample_codes(enc_data)

        # It might be better to use cross attention here
        x = self.combiners[block_idx](torch.cat([dec_state, enc_data], dim=-1))
        quantizer_out = self.quantizers[block_idx + 1](x, quantize=True, mask=enc_state.padding)
        vae_state.code_indices.append(quantizer_out.code_indices)

        # End the forward pass early if we're gathering latent codes and we just computed the bottom latent feature map
        if self.gathering_latents and block_idx + 2 == len(self.quantizers):
            return None

        # Don't use += operator to avoid autograd errors from doing an in-place operation- weirdly
        # this only seems to happen when we train with full precision
        old_commit_loss, old_embed_loss = vae_state.commitment_loss, vae_state.embedding_loss
        vae_state.commitment_loss = old_commit_loss + quantizer_out.commitment_loss
        vae_state.embedding_loss = old_embed_loss + quantizer_out.embedding_loss
        vae_state.soft_codes.append(quantizer_out.soft_codes)

        return self.quantizers[block_idx + 1].upsample_codes(quantizer_out.hard_codes)

    def train_or_val_step(self, batch: Dict[str, Tensor], stage: str) -> Dict[str, Tensor]:
        vae_state = self.forward(batch)

        # Convert these sums into means over codebooks. Within each codebook we compute the commitment and
        # embedding losses as the *mean* squared distance between the encoder output and the corresponding code in the
        # codebook, and so in order to be consistent we need to mean across codebooks as well.
        commitment_loss = vae_state.commitment_loss / len(self.quantizers)
        embedding_loss = vae_state.embedding_loss / len(self.quantizers)

        nll, ppl, entropy = self.stats_from_logits(vae_state.logits, batch, autoregressive=False)
        self.log('nll', nll, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        log_prefix = stage + '_'
        self.log_dict({
            log_prefix + 'nll': nll,
            log_prefix + 'ppl': ppl,
            log_prefix + 'entropy': entropy,
            log_prefix + 'commitment_loss': commitment_loss,
            log_prefix + 'embedding_loss': embedding_loss
        })
        self.log_dict(self.used_code_info_dict())

        logits = vae_state.logits.detach()
        if stage == 'val':
            # BLEU is expensive to compute so we only do it during validation
            reconstructions = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
            reconstructed_words = [sample.split() for sample in reconstructions]

            ground_truth = self.tokenizer.decode_batch(vae_state.ground_truth.tolist())
            real_words = [[sample.split()] for sample in ground_truth]

            self.log('val_bleu', bleu_score(reconstructed_words, real_words))

        elif stage == 'train':
            loss = nll + embedding_loss + self.hparams.beta * commitment_loss
            if not loss.isfinite():
                self.print("Warning: Got NaN loss. Skipping parameter update.")
                return None

            return {'logits': logits, 'loss': loss}

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

    # Used for logging to TensorBoard, mainly
    def used_code_info_dict(self):
        if len(self.quantizers) == 1:
            return {'num_used_codes': self.quantizers[0].num_used_codes()}

        info_dict = {}
        for level, quantizer in enumerate(self.quantizers):
            info_dict[f'code_entropy_{level}'] = quantizer.entropy
            info_dict[f'num_used_codes_{level}'] = quantizer.num_used_codes()

        return info_dict

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
