from .core.Autoencoder import *
from .funnel_transformers.FunnelTransformer import *
from .TextDataModule import *
from copy import deepcopy
from numpy import prod


@dataclass
class FunnelAutoencoderHparams(AutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(2, 2, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2),  # How much the hidden state is downsampled between each encoder block
    )
    max_seq_length: Optional[int] = None
    tie_embedding_weights: bool = True
    use_pretrained_encoder: bool = False
    logit_scale: Optional[float] = None


class FunnelAutoencoder(Autoencoder, ABC):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        encoder_hparams = hparams.encoder
        encoder_hparams.max_seq_length = hparams.max_seq_length

        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(encoder_hparams)
        attn_state.shared = True
        decoder_hparams = deepcopy(encoder_hparams)
        decoder_hparams.update(
            block_sizes=list(reversed(encoder_hparams.block_sizes)),
            scaling_factors=list(reversed(encoder_hparams.scaling_factors)),
            upsampling=True,
            use_transpose_convs=True
        )
        self.encoder = FunnelTransformer(encoder_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)
        self.encoder.attention_state = attn_state
        self.decoder.attention_state = attn_state

        logit_scale = hparams.logit_scale or encoder_hparams.d_model ** -0.5
        output_embedding = nn.Linear(encoder_hparams.d_model, hparams.vocab_size)
        output_layers = [
            nn.Linear(encoder_hparams.d_model, encoder_hparams.d_model),
            nn.GELU(),
            nn.LayerNorm(encoder_hparams.d_model),
            output_embedding,
            LambdaLayer(lambda logits: logits * logit_scale)
        ]
        self.output_layer = nn.Sequential(*output_layers)

        if hparams.tie_embedding_weights:
            output_embedding.weight = self.encoder.input_layer[0].weight

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

    def on_train_start(self):
        super(FunnelAutoencoder, self).on_train_start()

        data: TextDataModule = self.trainer.datamodule
        data.hparams.pad_to_multiple_of = int(prod(self.encoder.hparams.scaling_factors))

    def encoder_forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        x, padding = batch['token_ids'], batch['padding_mask']

        self.encoder.attention_state.configure_for_input(x.shape[-1], x.dtype, x.device, padding)
        return self.encoder(x, padding_mask=padding, **kwargs)
