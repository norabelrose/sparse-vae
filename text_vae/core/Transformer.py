from ..core import PaddedTensor
from .GenerationUtils import GenerationStrategy, decode_next_token_from_logits
from .LanguageModel import *
from .TransformerLayer import TransformerLayer, get_positional_encodings
from torch import nn
import torch


@dataclass
class TransformerHparams(LanguageModelHparams):
    d_model: int = 512
    max_seq_length: int = 512
    num_heads: int = 8
    num_layers: int = 6
    input_dropout: float = 0.1
    output_dropout: float = 0.0

    cross_attention: bool = False
    sparse_self_attention: bool = False


class Transformer(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(Transformer, self).__init__(hparams)

        vocab_size, d_model = hparams.vocab_size, hparams.d_model
        input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=hparams.padding_idx)
        self.input_layer = nn.Sequential(
            input_embedding,
            nn.LayerNorm(d_model),
            nn.Dropout(p=hparams.input_dropout)
        )

        output_embedding = nn.Linear(d_model, vocab_size)
        output_embedding.weight = input_embedding.weight
        self.output_layer = nn.Sequential(
            nn.Dropout(p=hparams.output_dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding
        )

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, hparams.num_heads, causal=True, cross_attention=hparams.cross_attention,
                             sparse_self_attention=hparams.sparse_self_attention)
            for _ in range(hparams.num_layers)
        ])
        self.initialize_weights()

    def forward(self, batch: Dict[str, Tensor]):
        x, padding = batch['token_ids'], batch['padding_mask']
        ctx, padding_ctx = batch.get('context'), batch.get('padding_mask_ctx')
        if ctx is not None:
            ctx = self.input_layer(ctx)

        x = self.input_layer(x)

        for layer in self.transformer_layers:
            x = layer(x, context=ctx, mask=padding, context_mask=padding_ctx)

        return self.output_layer(x)

    def _raw_forward(self, x: Tensor, context: Tensor = None, mask: Tensor = None, context_mask: Tensor = None):
        context = iter(context if isinstance(context, Iterable) else [context] * len(self.transformer_layers))

        for layer in self.transformer_layers:
            cur_context = next(context, None)
            x = layer(x, context=cur_context, mask=mask, context_mask=context_mask)

        return x

    @torch.no_grad()
    def sample(self, max_length: int, count: int = 1, k: int = 10, context: Tensor = None, context_mask: Tensor = None,
               min_length: int = 10, strategy: GenerationStrategy = GenerationStrategy.SamplingTopK,
               include_padding: bool = False, **kwargs):
        context = self.input_layer(context) if context is not None else None

        device = self.device
        num_samples = count
        if strategy == GenerationStrategy.Beam:
            beam_log_probs = torch.zeros(count, k, device=device)  # 100% likely that a sample will start with [CLS]
            num_samples *= k
        else:
            beam_log_probs = None

        padding_symbol = self.hparams.padding_idx  # May not be equal to 0 for VQ-VAE priors
        output_ids = torch.full([num_samples, max_length], padding_symbol, device=device, dtype=torch.long)

        start_symbol = torch.tensor([self.start_token], device=device)
        stop_symbol = torch.tensor([self.end_token], device=device)

        live_sample_mask = torch.ones(num_samples, device=device, dtype=torch.bool)

        d_model = self.hparams.d_model
        cur_query = self.input_layer(start_symbol).view(1, 1, -1).expand(num_samples, 1, d_model)
        output_ids[:, 0] = start_symbol

        for layer in self.transformer_layers:
            layer.attention.prepare_kv_cache(batch_size=num_samples, max_length=max_length)
            if context is not None:
                layer.cross_attention.prepare_kv_cache(batch_size=num_samples, max_length=context.shape[-2])

        for current_idx in range(1, max_length):
            next_output = self._raw_forward(cur_query, context=context, context_mask=context_mask)[:, -1]
            next_logits = self.output_layer(next_output)

            # Make the end symbol infinitely unlikely if we're not at the min length yet
            if current_idx < min_length:
                next_logits[..., self.end_token] = -float('inf')

            next_ids = decode_next_token_from_logits(next_logits, strategy, k, beam_log_probs=beam_log_probs)
            next_ids[~live_sample_mask].fill_(padding_symbol)

            output_ids[:, current_idx] = next_ids.squeeze(-1)
            cur_query = self.input_layer(next_ids).unsqueeze(-2)

            live_sample_mask &= (next_ids != stop_symbol)
            if current_idx > min_length and not live_sample_mask.any():
                # If we're doing beam search, get rid of all the sub-optimal hypotheses
                if strategy == GenerationStrategy.Beam:
                    best_indices = beam_log_probs.argmax(dim=-1)

                    output_ids = output_ids.view(count, k, -1)  # (batch_size * k, len) -> (batch_size, k, len)
                    output_ids = output_ids[:, best_indices]  # (batch_size, len)

                output_ids = output_ids[:, :current_idx + 1]  # Get rid of any excess padding
                break

        for layer in self.transformer_layers:
            layer.attention.reset_kv_cache()

        return PaddedTensor(data=output_ids, padding=output_ids.eq(padding_symbol)) if include_padding else output_ids


# Convenience class for learned absolute positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        initial_encodings = get_positional_encodings(max_length, d_model, device=None, dtype=torch.float32)
        self.weight = nn.Parameter(initial_encodings)

    def forward(self, x: Tensor) -> Tensor:
        x_len = x.shape[-2]
        return x + self.weight[:x_len]
