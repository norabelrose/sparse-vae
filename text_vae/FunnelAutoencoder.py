from .core.Autoencoder import *
from .funnel_transformers.FunnelTransformer import *
from .TextDataModule import *
from copy import deepcopy
from numpy import prod
from torch.distributions import Categorical


@dataclass
class FunnelAutoencoderHparams(AutoencoderHparams):
    funnel: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(2, 2, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2),  # How much the hidden state is downsampled between each encoder block
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

        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(funnel_hparams)
        attn_state.shared = True
        decoder_hparams = deepcopy(funnel_hparams)
        decoder_hparams.update(
            block_sizes=list(reversed(funnel_hparams.block_sizes)),
            scaling_factors=list(reversed(funnel_hparams.scaling_factors)),
            upsampling=True
        )
        self.encoder = FunnelTransformer(funnel_hparams)
        self.encoder.attention_state = attn_state

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

        self.decoder = FunnelTransformer(decoder_hparams)
        self.decoder.attention_state = attn_state
        self.setup_output_layer(hparams)

    def setup_output_layer(self, hparams: DictConfig):
        d_model = hparams.funnel.d_model
        output_embedding = nn.Linear(d_model, hparams.vocab_size)
        output_layers = [
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding,
            LambdaLayer(lambda x: Categorical(logits=x))
        ]
        self.output_layer = nn.Sequential(*output_layers)

        if self.encoder and hparams.tie_embedding_weights:
            self.encoder.input_layer[0].weight = output_embedding.weight

    def setup(self, stage: str):
        super().setup(stage)

        data: TextDataModule = self.trainer.datamodule if stage == 'fit' else self.datamodule
        data.hparams.pad_to_multiple_of = int(prod(self.decoder.hparams.scaling_factors))

    def encoder_forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        x, padding = batch['token_ids'], batch['padding_mask']

        self.encoder.attention_state.configure_for_input(x.shape[-1], padding)
        return self.encoder(x, padding_mask=padding, **kwargs)
