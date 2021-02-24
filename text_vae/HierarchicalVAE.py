from .core.VAE import *
from .funnel_transformers.FunnelTransformer import *
from .KNNLookupTable import *
from torch.distributions import Categorical


@dataclass
class HierarchicalVAEHparams(VAEHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(2, 2, 2),   # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(4, 2),  # How much the hidden state is downsampled between each encoder block
    )
    decoder_input_dropout: float = 0.5
    tie_embedding_weights: bool = True
    use_encoder_residual_connections: bool = True
    use_long_latents: bool = True
    use_length_encodings: bool = False
    use_pretrained_encoder: bool = False

    include_padding_positions: bool = True


@dataclass
class HierarchicalVAEState:
    ground_truth: Optional[Tensor] = None
    decoder_input: Optional[Tensor] = None
    encoder_states: List[Tensor] = field(default_factory=list)


class HierarchicalVAE(VAE, ABC):
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
        output_embedding = nn.Linear(encoder_hparams.d_model, hparams.vocab_size)
        self.output_layer = nn.Sequential(
            nn.Linear(encoder_hparams.d_model, encoder_hparams.d_model),
            nn.GELU(),
            nn.LayerNorm(encoder_hparams.d_model),
            output_embedding
        )

        if hparams.encoder.positional_encoding_type == 'learned':
            self.positional_encodings = nn.Embedding(192, encoder_hparams.d_model)
            attn_state.learned_pos_encodings = self.positional_encodings.weight

        elif hparams.tie_embedding_weights:
            self.encoder.input_layer[0].weight.data *= encoder_hparams.d_model * -0.5
            output_embedding.weight = self.encoder.input_layer[0].weight

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()
        else:
            # Scale the initializations of each layer by 1/sqrt(N) where N is the depth
            num_layers = sum(encoder_hparams.block_sizes)
            self.encoder.scale_parameters(depth=num_layers * 2)
            self.decoder.scale_parameters(depth=num_layers * 2)

    def encoder_forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        x, padding = batch['token_ids'], batch['padding_mask']

        self.encoder.attention_state.configure_for_input(x.shape[-1], x.dtype, x.device, padding)
        return self.encoder(x, padding_mask=padding, **kwargs)

    def decoder_forward(self, vae_state: HierarchicalVAEState, padding_mask: Tensor = None, **kwargs):
        self.decoder.attention_state.upsampling = True
        coroutine = self.decoder.forward_coroutine(
            vae_state.decoder_input,
            padding_mask=padding_mask
        )
        encoder_state_iter = iter(vae_state.encoder_states)

        for block_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                vae_state.decoder_output = decoder_state
                raw_output = decoder_state.final_state

                logits = self.output_layer(raw_output)
                vae_state.p_of_x_given_z = Categorical(logits=logits)

                return vae_state

            if encoder_state_iter:
                encoder_state = next(encoder_state_iter, None)
                if encoder_state is None:  # We ran out of encoder states to use
                    continue
            else:
                encoder_state = None

            coroutine.send(self.decoder_block_end(vae_state, decoder_state, encoder_state, block_idx, **kwargs))

    # Returns an optional Tensor which, if not None, is passed as input to the next decoder block
    @abstractmethod
    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        raise NotImplementedError

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'train')

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'test')

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)
