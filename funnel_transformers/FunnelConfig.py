from dataclasses import *
from ..Utilities import *

@dataclass
class FunnelConfig(SerializableObject):
    block_sizes: Tuple[int, ...]
    d_model: int
    num_heads: int

    # **Default values taken from pretrained config files & flags- do not change**
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    pooling_type: str = 'mean'
    scaling_factors: Union[int, Tuple[int, ...]] = 2
    separate_cls: bool = True
    pool_q_only: bool = True
    truncate_seq: bool = True
    seg_id_cls: int = 2  # Segment ID of the [CLS] token
    pad_id: int = None
    num_classes: int = 0
    use_classification_head: bool = False

    # Fine to change these
    attention_type: str = 'rel_shift'
    rezero_blocks: Optional[Iterable[int]] = field(default_factory=tuple)  # Blocks for which to use ReZero
    max_position_embeddings: int = 512
    return_attention_state: bool = False  # Useful for a VAE encoder; can reuse the state in the decoder
    return_block_outputs: bool = False  # Whether to return the pre-pooling output of each block on forward()
    use_performer_attention: bool = False
    upsampling: bool = False  # True for the "reverse" funnel transformer; e.g. a VAE decoder

    # 'decoder' in the Funnel-Transformer sense, not the VAE decoder sense. Used for pretraining.
    num_decoder_layers: int = 0

    def __post_init__(self):
        # Turn a single floating point scaling factor x into (x, x, x...) of the appropriate length
        if isinstance(self.scaling_factors, int):
            factor = self.scaling_factors
            self.scaling_factors = tuple(factor for _ in range(len(self.block_sizes) - 1))

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1 (no scaling)
        if len(self.scaling_factors) < len(self.block_sizes):
            self.scaling_factors += (1,)

    # Get a dictionary compatible with the old ModelConfig class from Funnel-Transformers
    def get_backward_compatible_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "d_embed": self.d_model,
            "d_model": self.d_model,
            "n_head": self.num_heads,
            "d_head": self.d_model // self.num_heads,
            "d_inner": self.d_model * 4,
            "dropout": self.dropout,
            "dropatt": self.attention_dropout,
            "dropact": self.ffn_dropout,
            "block_size": '_'.join([str(x) for x in self.block_sizes]),
            "pooling_type": self.pooling_type,

            # We lose info here since Funnel-Transformers doesn't support different scaling factors for each block
            "pooling_size": self.scaling_factors[0],
            "separate_cls": self.separate_cls,
            "pool_q_only": self.pool_q_only
        }

    # For the "args" parameter in the old FunnelTFM.__init__()
    def get_backward_compatible_args(self) -> Dict:
        return DynamicDict(
            pad_id=self.pad_id,
            num_class=self.num_classes,
            seg_id_cls=self.seg_id_cls,
            truncate_seq=self.truncate_seq,
            attn_type=self.attention_type
        )
