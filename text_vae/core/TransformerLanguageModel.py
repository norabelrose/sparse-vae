from .GenerationUtils import GenerationStrategy
from .LanguageModel import *
from .Transformers import Transformer, autoregressive_decode_transformer
from torch import nn
import torch


@dataclass
class TransformerLanguageModelHparams(LanguageModelHparams):
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    input_dropout: float = 0.1
    output_dropout: float = 0.0


class TransformerLanguageModel(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(TransformerLanguageModel, self).__init__(hparams)

        vocab_size, d_model = hparams.vocab_size, hparams.d_model
        input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        input_embedding.weight.data *= d_model ** -0.5
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

        self.decoder = Transformer(num_layers=hparams.num_layers, d_model=hparams.d_model, num_heads=hparams.num_heads,
                                   causal=True)

    def forward(self, x: Dict[str, Tensor]):
        x, padding = x['token_ids'], x['padding_mask']
        x = self.input_layer(x)
        x = self.decoder_forward(x, padding_mask=padding)

        return self.output_layer(x)

    def decoder_forward(self, x: Tensor, context: Tensor = None, padding_mask: Tensor = None):
        return self.decoder(x, cross_attn_target=context, padding_mask=padding_mask)

    @torch.no_grad()
    def sample(self, max_length: int, count: int = 1, start_embedding: Tensor = None, **kwargs):
        return autoregressive_decode_transformer(
            strategy=kwargs.get('strategy', GenerationStrategy.SamplingTopK),
            transformer_callable=self.decoder_forward,
            embedding=self.input_layer,
            logit_callable=self.output_layer,
            context=kwargs.get('context'),
            max_length=max_length,
            num_samples=count,
            d_model=self.hparams.d_model,
            start_symbol=self.start_token,
            end_symbol=self.end_token
        )
