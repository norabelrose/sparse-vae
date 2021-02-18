from dataclasses import dataclass  # noqa; here to silence PyCharm linter bug
from tqdm import tqdm
from text_vae.core.Autoencoder import *
from . import TransformerLayer, positional_encodings_like
from text_vae.core.TransformerLanguageModel import *


@dataclass
class QuantizedAutoencoderHparams(TransformerLanguageModelHparams, AutoencoderHparams):
    codebook_size: int = 512
    beta: float = 0.5
    ema_decay: float = 0.99
    use_kmeans_codebook_updates: bool = True
    input_dropout: float = 0.5
    output_dropout: float = 0.5


@dataclass
class QuantizedEncoderOutput:
    input_data: Tensor
    soft_codes: Tensor
    hard_codes: Optional[Tensor] = None
    hard_code_indices: Optional[Tensor] = None   # Indices of the hard code in the codebook


class QuantizedAutoencoder(TransformerLanguageModel):
    def __init__(self, hparams: DictConfig):
        super(QuantizedAutoencoder, self).__init__(hparams)

        num_codes = hparams.codebook_size
        self.codebook = nn.Embedding(num_codes, hparams.latent_depth)
        self.register_buffer('code_frequencies', torch.zeros(num_codes, device=self.device))

        self.code_downsample = nn.Linear(hparams.d_model, hparams.latent_depth)
        self.code_upsample = nn.Linear(hparams.latent_depth, hparams.d_model)

        self.encoder = nn.Sequential(*[
            TransformerLayer(d_model=hparams.d_model, num_heads=hparams.num_heads)
            for _ in range(self.hparams.num_layers)
        ])

    # Returns a tuple containing the encoder output and the quantized codes
    def forward(self, batch: Dict[str, Any], **kwargs) -> QuantizedEncoderOutput:
        x = batch['token_ids']
        x = self.input_layer(x)
        x = x + positional_encodings_like(x)

        for layer in self.encoder:
            x = layer(x, padding_mask=batch['padding_mask'])

        soft_codes = self.code_downsample(x.mean(dim=1))
        output = QuantizedEncoderOutput(input_data=x, soft_codes=soft_codes)

        # soft_codes_only = True is used by update_codebook_kmeans
        if not kwargs.get('soft_codes_only'):
            # Find the codes in the codebook closest to the output of the encoder in terms of Euclidean distance
            code_indices = torch.cdist(soft_codes, self.codebook.weight).argmin(dim=-1)
            output.hard_codes = self.codebook(code_indices)
            output.hard_code_indices = code_indices

            # Update our prior
            num_codes = self.hparams.codebook_size
            code_freqs = code_indices.flatten().bincount(minlength=num_codes)[:num_codes]
            self.code_frequencies *= self.hparams.ema_decay  # Exponential decay
            self.code_frequencies += (1 - self.hparams.ema_decay) * code_freqs

        return output

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        # First epoch: just use the soft codes and train as a continuous, non-variational autoencoder
        should_quantize = self.trainer.current_epoch > 0
        encoder_out = self.forward(batch, soft_codes_only=not should_quantize)
        encoder_input, soft_codes, hard_codes = encoder_out.input_data, encoder_out.soft_codes, encoder_out.hard_codes

        decoder_input = encoder_input
        log_prefix = kwargs.get('log_prefix') or 'train_'
        if not should_quantize:
            decoder_input[:, 0] = self.code_upsample(encoder_out.soft_codes)
            commitment_loss = embedding_loss = torch.tensor(0.0, device=soft_codes.device)

        # After the first epoch, ease into using the hard codes
        else:
            commitment_loss = F.mse_loss(hard_codes.detach(), soft_codes)
            embedding_loss = F.mse_loss(hard_codes, soft_codes.detach())

            soft_codes, hard_codes = self.code_upsample(soft_codes), self.code_upsample(hard_codes)
            decoder_input[:, 0] = soft_codes + (hard_codes - soft_codes).detach()

            # These things are only relevant to log after the first epoch
            self.log_dict({
                'active_codes': self.code_frequencies.count_nonzero(),
                log_prefix + 'commitment_loss': commitment_loss,
                log_prefix + 'embedding_loss': embedding_loss
            })

        decoder_output = self.decoder_forward(decoder_input, padding_mask=batch['padding_mask'])
        logits = self.output_layer(decoder_output)

        reconstruction_loss = F.cross_entropy(
            input=logits[:, :-1].flatten(0, 1),          # Remove final [SEP] token
            target=batch['token_ids'][:, 1:].flatten(),  # Remove initial [CLS] token
            ignore_index=0
        )

        # Log the entropy of the model's probability distribution over words to see how confident it is
        self.log('logit_entropy', (logits * F.softmax(logits, dim=-1)).sum(dim=-1).mean())

        total_loss = reconstruction_loss + embedding_loss + self.hparams.beta * commitment_loss
        self.log_dict({
            log_prefix + 'loss': total_loss,
            log_prefix + 'reconstruction_loss': reconstruction_loss
        })
        result = {'loss': total_loss}
        if not kwargs.get('loss_only'):
            result['logits'] = logits

        return result

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_index, log_prefix='val_', loss_only=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.trainer.running_sanity_check or not self.hparams.use_kmeans_codebook_updates:
            return

        self.update_codebook_kmeans()

    def compute_posteriors(self, batch: Dict[str, Tensor]) -> List[Tensor]:
        batched_codes = self.forward(batch).hard_code_indices
        return batched_codes.split(1, dim=0)

    def sample(self, max_length: int, count: int = 1, **kwargs):
        denom = self.code_frequencies.sum()
        if not denom:
            code_indices = torch.randint(self.hparams.codebook_size, size=[count], device=self.device)
        else:
            probs = self.code_frequencies / denom
            code_indices = probs.multinomial(num_samples=count, replacement=True)

        codes = self.codebook(code_indices)
        return super().sample(max_length, count, start_embedding=codes, strategy=GenerationStrategy.Greedy, **kwargs)

    @torch.no_grad()
    def update_codebook_kmeans(self, num_restarts: int = 3):
        self.print("\nPerforming K means codebook update...")

        # Do encoder forward passes through the entire training dataset in order to gather the soft codes
        loader = self.trainer.train_dataloader
        loader = tqdm(islice(loader, len(loader)), desc='Gathering encoder outputs', total=len(loader))
        observed_codes = [self.forward(
            {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()},
            soft_codes_only=True
        ).hard_codes for batch in loader]

        soft_codes = torch.cat(observed_codes, dim=0)

        raw_codebook = self.codebook.weight.data
        cur_codebook = torch.empty_like(raw_codebook)
        num_codes, code_depth = raw_codebook.shape

        soft_codes = soft_codes.flatten(end_dim=-2)
        codes_per_cluster = soft_codes.shape[0] / num_codes
        assert codes_per_cluster > 1.0, "Not enough soft codes to perform K means codebook update"

        best_loss = torch.tensor(float('inf'), device=raw_codebook.device)
        max_iter = min(100, int(ceil(codes_per_cluster)))
        pbar = tqdm(total=max_iter * num_restarts, postfix=dict(best_loss=float('inf')))

        for i in range(num_restarts):
            # Initialize centroids with actually observed values
            cur_codebook.copy_(soft_codes[num_codes * i:num_codes * (i + 1)])
            pbar.desc = f'Computing centroids (restart {i + 1} of {num_restarts})'

            for _ in range(max_iter):
                hard_code_indices = torch.cdist(soft_codes, cur_codebook).argmin(dim=-1)
                cur_codebook.zero_()

                # Sum together all the soft codes assigned to a cluster, then divide by the number in that cluster
                counts = hard_code_indices.bincount(minlength=num_codes)
                cur_codebook.scatter_add_(dim=0, index=hard_code_indices[:, None].repeat(1, code_depth), src=soft_codes)
                cur_codebook /= counts.unsqueeze(-1)

                pbar.update()

            cur_loss = torch.cdist(soft_codes, cur_codebook).min(dim=-1).values.pow(2.0).mean() / code_depth  # MSE
            if cur_loss < best_loss:
                raw_codebook.copy_(cur_codebook)

                best_loss = cur_loss
                pbar.set_postfix(best_loss=best_loss.item())

        self.code_frequencies.zero_()
        pbar.close()

    def should_unconditionally_sample(self) -> bool:
        return self.trainer.current_epoch > 0
