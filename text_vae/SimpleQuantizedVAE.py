from dataclasses import dataclass  # noqa; here to silence PyCharm linter bug
from .core import (
    TransformerLayer, positional_encodings_like,
    Quantizer, QuantizerOutput, QuantizedVAE, QuantizedVAEHparams
)
from .core.TransformerLanguageModel import *


@dataclass
class SimpleQuantizedVAEHparams(TransformerLanguageModelHparams, QuantizedVAEHparams):
    ema_decay: float = 0.99


class SimpleQuantizedVAE(TransformerLanguageModel, QuantizedVAE):
    def __init__(self, hparams: DictConfig):
        super(SimpleQuantizedVAE, self).__init__(hparams)

        num_codes = hparams.codebook_size
        self.quantizer = Quantizer(num_codes, hparams.latent_depth, hparams.d_model)
        self.register_buffer('code_frequencies', torch.zeros(num_codes, device=self.device))

        self.encoder = nn.Sequential(*[
            TransformerLayer(d_model=hparams.d_model, num_heads=hparams.num_heads)
            for _ in range(self.hparams.num_layers)
        ])

    # Returns a dataclass containing the encoder output and the quantized codes
    def forward(self, batch: Dict[str, Any], quantize: bool = False) -> QuantizerOutput:
        x = self.preprocess(batch)
        return self.encoder_forward(x, batch['padding_mask'], quantize=quantize)

    def preprocess(self, batch: Dict[str, Any]) -> Tensor:
        x = batch['token_ids']
        x = self.input_layer(x)
        return x + positional_encodings_like(x)

    def encoder_forward(self, x: Tensor, padding_mask: Tensor, quantize: bool = True):
        for layer in self.encoder:
            x = layer(x, padding_mask=padding_mask)

        # soft_codes_only = True is used by update_codebook_kmeans
        output = self.quantizer(x.mean(dim=1), quantize=quantize)

        if quantize:
            # Update our prior
            num_codes = self.hparams.codebook_size
            code_freqs = output.code_indices.flatten().bincount(minlength=num_codes)[:num_codes]
            self.code_frequencies *= self.hparams.ema_decay  # Exponential decay
            self.code_frequencies += (1 - self.hparams.ema_decay) * code_freqs

        return output

    def training_step(self, batch: Dict[str, Any], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        encoder_input = self.preprocess(batch)

        # First epoch: just use the soft codes and train as a continuous, non-variational autoencoder
        should_quantize = self.trainer.current_epoch > 0
        quantized = self.encoder_forward(encoder_input, padding_mask=batch['padding_mask'], quantize=should_quantize)

        decoder_input = encoder_input
        log_prefix = kwargs.get('log_prefix') or 'train_'
        if not should_quantize:
            decoder_input[:, 0] = self.quantizer.upsample_codes(quantized.soft_codes)

        # After the first epoch, ease into using the hard codes
        else:
            decoder_input[:, 0] = self.quantizer.upsample_codes(quantized.hard_codes)

            # These things are only relevant to log after the first epoch
            self.log_dict({
                'active_codes': self.code_frequencies.count_nonzero(),
                log_prefix + 'commitment_loss': quantized.commitment_loss,
                log_prefix + 'embedding_loss': quantized.embedding_loss
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

        total_loss = reconstruction_loss + quantized.embedding_loss + self.hparams.beta * quantized.commitment_loss
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

    def compute_posteriors(self, batch: Dict[str, Tensor]) -> List[Tensor]:
        batched_codes = self.forward(batch).code_indices
        return batched_codes.split(1, dim=0)

    def sample(self, max_length: int, count: int = 1, **kwargs):
        if self.training and self.trainer.current_epoch == 0:
            return None
        
        denom = self.code_frequencies.sum()
        if not denom:
            code_indices = torch.randint(self.hparams.codebook_size, size=[count], device=self.device)
        else:
            probs = self.code_frequencies / denom
            code_indices = probs.multinomial(num_samples=count, replacement=True)

        codes = self.quantizer.lookup_codes(code_indices)
        return super().sample(max_length, count, start_embedding=codes, strategy=GenerationStrategy.Greedy, **kwargs)
