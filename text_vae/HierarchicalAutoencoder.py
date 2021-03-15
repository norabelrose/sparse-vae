from .FunnelAutoencoder import *
from torch.distributions import Categorical


@dataclass
class HierarchicalAutoencoderHparams(FunnelAutoencoderHparams):
    use_long_latents: bool = True
    use_length_encodings: bool = False
    include_padding_positions: bool = True


@dataclass
class HierarchicalAutoencoderState:
    ground_truth: Optional[Tensor] = None
    decoder_input: Optional[Tensor] = None
    encoder_states: List[Tensor] = field(default_factory=list)
    p_of_x_given_z: Optional[Categorical] = None


class HierarchicalAutoencoder(FunnelAutoencoder, ABC):
    def decoder_forward(self, vae_state: HierarchicalAutoencoderState, padding_mask: Tensor = None, **kwargs):
        attn_state = self.decoder.attention_state
        attn_state.upsampling = True

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
