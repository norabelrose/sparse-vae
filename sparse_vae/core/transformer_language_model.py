from .attention import Attention
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
    input_dropout: float = 0.0

    tie_embedding_weights: bool = True

    cross_attention: bool = False
    grad_checkpointing: bool = False
    separate_context_embedding: bool = True

    attn_window_size: int = 4
    sparse_self_attention: bool = True


class TransformerLanguageModel(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(TransformerLanguageModel, self).__init__(hparams)

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
            TransformerLayer(
                d_model, hparams.num_heads, causal=True, use_cross_attention=hparams.cross_attention,
                sparse_self_attention=False if not hparams.sparse_self_attention else hparams.attn_window_size
            )
            for _ in range(hparams.num_layers)
        ])

    def on_fit_start(self):
        self.initialize_weights()

    def embed_context(self, context: Tensor):
        return self.context_layer(context) if self.context_layer else self.input_layer(context)

    def forward(self, batch: Dict[str, PaddedTensor]):
        x, context = batch['token_ids'].long(), batch.get('context')
        if context is not None:
            raise NotImplementedError
            # context = self.embed_context(context)

        x = self.input_layer(x)

        # should_checkpoint = self.hparams.grad_checkpointing and x.requires_grad
        for layer in self.decoder_layers:
            x = layer(x)

        return self.output_layer(x)

    @torch.no_grad()
    def sample(self, max_length: int, batch_size: int = 1, context: Tensor = None, z: Tensor = None, **kwargs):
        context = self.embed_context(context) if context is not None else None

        state = GenerationState(
            max_length, batch_size, self.start_token, self.end_token, device=self.device, **kwargs
        )
        state.output_ids[:, 0] = self.start_token

        with Attention.kv_cache(max_length):
            while not state.should_stop():
                inputs = state.prev_tokens()
                x = self.input_layer(inputs)
                if z is not None:
                    x = x + z[state.live_sample_mask, :]

                for layer in self.decoder_layers:
                    # if z is not None and state.current_index == 1:
                    #     x += z

                    x = layer(x, context=context)

                next_logits = self.output_layer(x.squeeze(1))
                continuing_mask = state.process_logits(next_logits)

                Attention.update_kv_cache(continuing_mask)

        return state.final_output()
