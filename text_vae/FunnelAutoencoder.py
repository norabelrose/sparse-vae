from .core.Autoencoder import *
from .funnel_transformers.FunnelTransformer import *
from copy import deepcopy


@dataclass
class FunnelAutoencoderHparams(AutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(2, 2, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(4, 2),  # How much the hidden state is downsampled between each encoder block
    )
    logit_scale_factor: Optional[float] = None  # If None defaults to 1/sqrt(d_model)
    tie_embedding_weights: bool = True
    use_pretrained_encoder: bool = False


class FunnelAutoencoder(Autoencoder, ABC):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        encoder_hparams = hparams.encoder

        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(encoder_hparams)
        attn_state.shared = True
        decoder_hparams = deepcopy(encoder_hparams)
        decoder_hparams.update(
            block_sizes=list(reversed(encoder_hparams.block_sizes)),
            scaling_factors=list(reversed(encoder_hparams.scaling_factors)),
            upsampling=True
        )
        self.encoder = FunnelTransformer(encoder_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)
        self.encoder.attention_state = attn_state
        self.decoder.attention_state = attn_state

        logit_scale = hparams.logit_scale_factor or encoder_hparams.d_model ** -0.5
        output_embedding = nn.Linear(encoder_hparams.d_model, hparams.vocab_size)
        self.output_layer = nn.Sequential(
            nn.Linear(encoder_hparams.d_model, encoder_hparams.d_model),
            nn.GELU(),
            nn.LayerNorm(encoder_hparams.d_model),
            output_embedding,
            LambdaLayer(lambda logits: logits * logit_scale)
        )

        if hparams.encoder.positional_attention_type == 'learned':
            self.positional_encodings = nn.Embedding(192, encoder_hparams.d_model)
            attn_state.learned_pos_encodings = self.positional_encodings.weight

        elif hparams.tie_embedding_weights:
            output_embedding.weight = self.encoder.input_layer[0].weight

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()
        # else:
        #     # Scale the initializations of each layer by 1/sqrt(N) where N is the depth
        #     num_layers = sum(encoder_hparams.block_sizes)
        #     # self.encoder.scale_parameters(depth=num_layers * 2)
        #     self.decoder.scale_parameters(depth=num_layers * 2)

    def encoder_forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        x, padding = batch['token_ids'], batch['padding_mask']

        self.encoder.attention_state.configure_for_input(x.shape[-1], x.dtype, x.device, padding)
        return self.encoder(x, padding_mask=padding, **kwargs)
