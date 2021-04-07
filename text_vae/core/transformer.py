from .generation import GenerationState
from .language_model import *
from .padded_tensor import PaddedTensor
from .transformer_layer import TransformerLayer
from copy import deepcopy
from torch import nn
import torch


@dataclass
class TransformerHparams(LanguageModelHparams):
    d_embedding: Optional[int] = None   # Set to d_model if None
    d_model: int = 512
    max_seq_length: int = 512
    num_heads: int = 8
    num_layers: int = 6
    input_dropout: float = 0.1
    tie_embedding_weights: bool = True

    cross_attention: bool = False
    separate_context_embedding: bool = True
    sparse_self_attention: bool = False


class Transformer(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(Transformer, self).__init__(hparams)

        vocab_size, d_model, d_embedding = hparams.vocab_size, hparams.d_model, hparams.d_embedding
        d_embedding = d_embedding or d_model

        input_embedding = nn.Embedding(vocab_size, d_embedding)
        input_layers = [
            input_embedding,
            nn.Dropout(p=hparams.input_dropout)
        ]
        if d_embedding != d_model:
            input_layers.insert(1, nn.Linear(d_embedding, d_model))

        self.input_layer = nn.Sequential(*input_layers)

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
        if hparams.tie_embedding_weights and d_embedding == d_model:
            output_embedding.weight = input_embedding.weight

        self.decoder_layers = nn.ModuleList([
            TransformerLayer(d_model, hparams.num_heads, causal=True, use_cross_attention=hparams.cross_attention,
                             sparse_self_attention=hparams.sparse_self_attention)
            for _ in range(hparams.num_layers)
        ])

    def setup(self, stage: str):
        super().setup(stage)

        if self.hparams.sparse_self_attention:
            datamodule = self.trainer.datamodule if stage == 'fit' else self.datamodule
            datamodule.hparams.pad_to_multiple_of = 16

        self.initialize_weights()

    def embed_context(self, context: Tensor):
        return self.context_layer(context) if self.context_layer else self.input_layer(context)

    def forward(self, batch: Dict[str, PaddedTensor]):
        x, context = batch['token_ids'], batch.get('context')
        if context is not None:
            context = self.embed_context(context)

        x = self.input_layer(x)
        for layer in self.decoder_layers:
            x = layer(x, context=context)

        return self.output_layer(x)

    @torch.no_grad()
    def sample(self, max_length: int, batch_size: int = 1, context: Tensor = None, initial_embedding: Tensor = None,
               beam_size: int = 1, top_k: int = 0, top_p: float = 0.92, temperature: float = 1.0,
               length_penalty: float = 1.0, dtype: torch.dtype = torch.long):
        context = self.embed_context(context) if context is not None else None

        state = GenerationState(
            max_length, batch_size, beam_size, self.device, dtype, self.end_token,
            top_k, top_p, temperature, length_penalty
        )
        num_samples = batch_size * beam_size
        state.output_ids[:, 0] = self.start_token
        has_created_cache = False

        while not state.should_stop():
            if initial_embedding is not None:
                x = initial_embedding
                initial_embedding = None
            else:
                x = self.input_layer(state.prev_tokens())

            # We need to know what dtype to make the key-value caches, and it has to be the same dtype as
            # whatever the embedding layer outputs. We might be in a torch.cuda.amp.autocast() context, so
            # can't hard-code this. This is why we lazily create the caches after the first token gets embedded
            if not has_created_cache:
                has_created_cache = True
                dtype = x.dtype

                for layer in self.decoder_layers:
                    layer.attention.prepare_kv_cache(batch_size=num_samples, max_length=max_length, dtype=dtype)
                    if layer.cross_attention is not None:
                        layer.cross_attention.prepare_kv_cache(batch_size=num_samples, max_length=context.shape[-2],
                                                               dtype=dtype)

            for layer in self.decoder_layers:
                x = layer(x, context=context, cache_mask=state.live_sample_mask)

            next_logits = self.output_layer(x).squeeze(-2)
            state.process_logits(next_logits)

        for layer in self.decoder_layers:
            layer.attention.reset_kv_cache()
            if layer.cross_attention is not None:
                layer.cross_attention.reset_kv_cache()

        return state.final_output()
