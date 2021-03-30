from .core.Autoencoder import *
from .funnel_transformers.FunnelTransformer import *
from .TextDataModule import *
from copy import deepcopy
from numpy import prod


@dataclass
class FunnelAutoencoderHparams(AutoencoderHparams):
    funnel: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(4, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2,),  # How much the hidden state is downsampled between each encoder block
    )
    max_seq_length: Optional[int] = None
    tie_embedding_weights: bool = True
    use_pretrained_encoder: bool = False


class FunnelAutoencoder(LanguageModel, ABC):
    # Set include_encoder to False if we're strictly using the model for generation
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        funnel_hparams = hparams.funnel
        funnel_hparams.max_seq_length = hparams.max_seq_length

        decoder_hparams = deepcopy(funnel_hparams)
        decoder_hparams.update(
            block_sizes=list(reversed(funnel_hparams.block_sizes)),
            scaling_factors=list(reversed(funnel_hparams.scaling_factors)),
            upsampling=True
        )
        self.encoder = FunnelTransformer(funnel_hparams)

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

        self.decoder = FunnelTransformer(decoder_hparams)

        d_model = funnel_hparams.d_model
        output_embedding = nn.Linear(d_model, hparams.vocab_size)
        output_layers = [
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding
        ]
        self.output_layer = nn.Sequential(*output_layers)

        if self.encoder and hparams.tie_embedding_weights:
            self.encoder.input_layer[0].weight = output_embedding.weight

    def setup(self, stage: str):
        super().setup(stage)

        data: TextDataModule = self.trainer.datamodule if stage == 'fit' else self.datamodule
        data.hparams.pad_to_multiple_of = int(prod(self.decoder.hparams.scaling_factors))
