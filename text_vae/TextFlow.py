from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms import (
    AffineCouplingBijection,
    LogisticMixtureCouplingBijection,
    SequentialTransform,
    Shuffle
)
from .AttentionState import AttentionState
from text_vae.core.LanguageModel import *
from .EmbeddingSurjection import *
from .ops import AttentionHparams, RelativePositionalAttention
from .Stride import *


@dataclass
class TextFlowHparams(AttentionHparams, LanguageModelHparams):
    block_sizes: int = (2, 2, 2, 2, 2)
    scaling_factors: int = (1, 1, 1, 1)

    d_model: int = 512
    num_heads: int = 8


class TextFlow(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(TextFlow, self).__init__(hparams)

        d_model = hparams.d_model
        self.attention_state = AttentionState(hparams)

        self.flow = Flow(
            base_dist=StandardNormal([d_model * 2]),
            transforms=[
                EmbeddingSurjection(vocab_size=self.tokenizer.get_vocab_size(), out_features=d_model * 2,
                                    use_knn_inverse=True),
                SequentialTransform([
                    # Each Transformer layer
                    SequentialTransform([
                        LogisticMixtureCouplingBijection(
                            MultiHeadTransformerLayer(hparams, self.attention_state, num_ffns=12, depth=i),
                            num_mixtures=4,
                            split_dim=2
                        ),
                        Shuffle(d_model * 2, dim=2),
                        AffineCouplingBijection(
                            MultiHeadTransformerLayer(hparams, self.attention_state, num_ffns=2, depth=i),
                            split_dim=2
                        ),
                        Shuffle(d_model * 2, dim=2)
                    ])
                    for i in range(len(hparams.block_sizes))
                ])
            ])

    # Get the latent representation for some input
    def forward(self, batch: Dict[str, Any], **kwargs) -> Tensor:
        x = self._prepare_batch(batch)
        for transform in self.flow.transforms:
            x, _ = transform(x)  # Discard the log Jacobian determinants

        return x

    def sample(self, max_length: int, count: int = 1, **kwargs):
        z = torch.randn(count, max_length, self.hparams.d_model * 2, device=self.device)
        if temp := kwargs.get('temperature'):
            z *= temp

        self.attention_state.configure_for_input(
            seq_len=max_length,
            dtype=z.dtype,
            device=self.device,
            padding_mask=None
        )
        for transform in reversed(self.flow.transforms):
            z = transform.inverse(z)

        return z

    def training_step(self, batch: Dict[str, Any], batch_index: int, **kwargs) -> Optional[Tensor]:
        x = self._prepare_batch(batch)

        log_prob = self.flow.log_prob(x)
        loss = -log_prob.mean() / (x.shape[1] * self.hparams.d_model * 2 * log(2))  # Bits per dimension

        log_prefix = kwargs.get('log_prefix') or 'train_'
        self.log(log_prefix + 'bits_per_dim', loss)

        # Skip the training step if the loss is NaN or inf
        if not loss.isfinite():
            return None

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_index: int, **kwargs):
        self.training_step(batch, batch_index, log_prefix='val_')

    def _prepare_batch(self, batch: Dict[str, Any]):
        original_tokens = batch['token_ids']
        self.attention_state.configure_for_input(
            seq_len=original_tokens.shape[1],
            dtype=original_tokens.dtype,
            device=original_tokens.device,
            padding_mask=batch['padding_mask']
        )
        return original_tokens


# A Transformer Layer with multiple FFN "heads"
class MultiHeadTransformerLayer(nn.Module):
    def __init__(self, hparams: DictConfig, attn_state: AttentionState, num_ffns: int, depth: int = 0):
        super(MultiHeadTransformerLayer, self).__init__()
        d_model = hparams.d_model

        self.attention = RelativePositionalAttention(hparams)
        self.attn_state = attn_state
        self.dropout = nn.Dropout(p=0.1)
        self.attn_layer_norm = nn.LayerNorm(d_model)

        def get_linear(in_features: int, out_features: int, zero_initialized: bool = True):
            linear = nn.Linear(in_features, out_features)
            if zero_initialized:
                linear.weight.data.zero_()
                linear.bias.data.zero_()

            return linear

        self.ffn_layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_ffns)])
        self.inner_ffn = get_linear(d_model, d_model * 4)
        self.outer_ffns = nn.ModuleList([
            nn.Sequential(nn.GELU(), get_linear(d_model * 4, d_model))
            for _ in range(num_ffns)
        ])

        if depth > 0:  # Scale parameter initializations by 1/sqrt(N)
            self._scale_parameters(depth)

    # Output has new final dimension
    def forward(self, x: Tensor) -> Tensor:
        y = self.attention(x, x, x, attn_state=self.attn_state)

        y = self.dropout(y)
        x = y = self.attn_layer_norm(x + y)
        y = self.inner_ffn(y)

        outputs = [self.dropout(ffn(y)) for ffn in self.outer_ffns]
        outputs = [layer_norm(x + y) for layer_norm, y in zip(self.ffn_layer_norms, outputs)]

        return torch.stack(outputs, dim=-1)

    @torch.no_grad()
    def _scale_parameters(self, depth: int):
        for param in self.parameters():
            param.data *= depth ** -0.5
