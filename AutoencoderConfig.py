from __future__ import annotations
from dataclasses import dataclass

from .Utilities import *
from .funnel_transformers.FunnelTransformer import FunnelConfig


@dataclass
class AutoencoderConfig(SerializableObject):
    # The number of layers in each block in the encoder; sequence is reversed for the decoder
    block_sizes: Tuple[int, ...]

    # The (default) scaling factor used between each block
    scaling_factors: Tuple[int, ...]

    overt_depth: int        # Depth of the (overt) token embeddings (e.g. 768, 1024)
    latent_depth: int = 16  # Depth of the latent tensors (dimensionality per token)
    attention_head_depth: int = 64  # The dimensionality of each attention head

    use_pretrained_encoder: bool = True
    copy_encoder_weights_to_decoder: bool = True

    # Whether to condition z0 on the document title
    condition_on_title: bool = True

    # Whether to use an autoregressive Transformer decoder as the last layer of our decoder
    use_autoregressive_decoding: bool = False

    # Whether to use O(n) attention mechanism from "Rethinking Attention with Performers"
    use_performer_attention: bool = False

    # Determines the size of the positional embeddings we use
    max_sequence_length: int = 512

    attention_dropout: float = 0.1

    block_size_to_name: ClassVar[dict] = {
        (4, 4, 4): "small",
        (6, 6, 6): "intermediate",
        (8, 8, 8): "large",
        (10, 10, 10): "xlarge"
    }
    name_to_block_size: ClassVar[dict] = invert(block_size_to_name)

    @classmethod
    def default(cls) -> AutoencoderConfig:
        return cls.default_with_size("xlarge")

    @classmethod
    def default_with_size(cls, size: str) -> AutoencoderConfig:
        try:
            funnel_layout = cls.name_to_block_size[size]
        except KeyError:
            raise ValueError(f"AutoencoderConfig: No such named size '{size}'.")

        # xlarge and large use 1024-D embeddings, everything else uses 768
        embedding_dim = 1024 if "large" in size else 768

        # Default is to add 2 extra low-resolution layers after the funnel
        return cls(block_sizes=funnel_layout + (2, 2), scaling_factors=(2, 2, 2, 4, 4), overt_depth=embedding_dim)

    def __post_init__(self):
        assert all(x > 0 for x in self.block_sizes) and\
               all(x > 0 for x in self.scaling_factors) and\
               self.latent_depth > 0
        assert self.overt_depth % self.attention_head_depth == 0
        assert len(self.block_sizes) == len(self.scaling_factors) + 1, \
            "AutoencoderConfig: block_sizes must be exactly one element longer than scaling_factors."

    # Returns the name of the Huggingface Funnel Transformer tokenizer that should work best for this latent structure.
    # This method is less "picky" than pretrained_funnel_transformer_name in that the block sizes do not have to
    # exactly line up with those of the pretrained models. It just picks the tokenizer for the model which has the
    # most similar block structure in terms of L1 distance.
    def pretrained_tokenizer_name(self) -> str:
        num_funnel_blocks = min(3, len(self.block_sizes))  # Conceivably we might have less than 3 blocks
        funnel_layout = self.block_sizes[0:num_funnel_blocks]

        pretrained_block_layouts = [sizes[0:num_funnel_blocks] for sizes in self.block_size_to_name.keys()]
        closest_pretrained_layout = min(pretrained_block_layouts, key=lambda x: sum(abs(x - funnel_layout)))
        return self.block_size_to_name[closest_pretrained_layout]

    def get_funnel_config(self) -> FunnelConfig:
        return FunnelConfig(
            attention_type="factorized" if self.use_performer_attention else "rel_shift",
            block_sizes=self.block_sizes[0:3],
            max_position_embeddings=self.max_sequence_length,
            return_block_outputs=True,
            use_performer_attention=self.use_performer_attention,
            hiddens_to_return='per_block'
        )
