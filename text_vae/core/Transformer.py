from .GenerationUtils import GenerationStrategy, decode_next_token_from_logits
from .LanguageModel import *
from .TransformerLayer import TransformerLayer
from copy import deepcopy
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
import torch


@dataclass
class TransformerHparams(LanguageModelHparams):
    d_model: int = 512
    max_seq_length: int = 512
    num_heads: int = 8
    num_layers: int = 6
    input_dropout: float = 0.1
    tie_embedding_weights: bool = True

    cross_attention: bool = False
    separate_context_embedding: bool = True
    sparse_self_attention: bool = True


class Transformer(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(Transformer, self).__init__(hparams)

        vocab_size, d_model = hparams.vocab_size, hparams.d_model
        input_embedding = nn.Embedding(vocab_size, d_model)
        self.input_layer = nn.Sequential(
            input_embedding,
            nn.Dropout(p=hparams.input_dropout)
        )

        if hparams.cross_attention and hparams.separate_context_embedding:
            self.context_layer = deepcopy(self.input_layer)
        else:
            self.context_layer = None

        output_embedding = nn.Linear(d_model, vocab_size)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding
        )
        if hparams.tie_embedding_weights:
            output_embedding.weight = input_embedding.weight

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, hparams.num_heads, causal=True, cross_attention=hparams.cross_attention,
                             sparse_self_attention=hparams.sparse_self_attention)
            for _ in range(hparams.num_layers)
        ])
        self.initialize_weights()

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [EarlyStopping(monitor='val_nll', mode='min')]

    def setup(self, stage: str):
        super().setup(stage)

        if self.hparams.sparse_self_attention:
            datamodule = self.trainer.datamodule if stage == 'fit' else self.datamodule
            datamodule.hparams.pad_to_multiple_of = 16

    def embed_context(self, context: Tensor):
        return self.context_layer(context) if self.context_layer else self.input_layer(context)

    def forward(self, batch: Dict[str, Tensor], embed_context: bool = True):
        x, padding = batch['token_ids'], batch.get('padding_mask')
        ctx, padding_ctx = batch.get('context'), batch.get('padding_mask_ctx')
        if ctx is not None and embed_context:
            ctx = self.embed_context(ctx)

        x = self.input_layer(x)

        for layer in self.transformer_layers:
            x = layer(x, context=ctx, mask=padding, context_mask=padding_ctx)

        return self.output_layer(x)

    @torch.no_grad()
    def sample(self, max_length: int, count: int = 1, k: int = 10, context: Tensor = None, context_mask: Tensor = None,
               min_length: int = 10, strategy: GenerationStrategy = GenerationStrategy.SamplingTopP, **kwargs):
        context = self.embed_context(context) if context is not None else None

        device = self.device
        num_samples = count
        if strategy == GenerationStrategy.Beam:
            beam_log_probs = torch.zeros(count, k, device=device)  # 100% likely that a sample will start with [CLS]
            num_samples *= k
        else:
            beam_log_probs = None

        padding_symbol = 0
        output_ids = torch.full([num_samples, max_length], padding_symbol, device=device, dtype=torch.long)

        start_symbol = torch.tensor([self.start_token], device=device)
        stop_symbol = torch.tensor([self.end_token], device=device)

        live_sample_mask = torch.ones(num_samples, device=device, dtype=torch.bool)
        output_ids[:, 0] = start_symbol

        for layer in self.transformer_layers:
            layer.attention.prepare_kv_cache(batch_size=num_samples, max_length=max_length, dtype=torch.float16)
            if context is not None:
                layer.cross_attention.prepare_kv_cache(batch_size=num_samples, max_length=context.shape[-2],
                                                       dtype=torch.float16)

        for current_idx in range(1, max_length):
            next_logits = self.forward({
                'token_ids': output_ids[:, current_idx - 1, None],
                'context': context,
                'padding_mask_ctx': context_mask
            }, embed_context=False).squeeze(-2)

            next_ids = decode_next_token_from_logits(next_logits, strategy, k, beam_log_probs=beam_log_probs)
            next_ids[~live_sample_mask].fill_(padding_symbol)
            output_ids[:, current_idx] = next_ids.squeeze(-1)

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

        return output_ids
