from text_vae.core.TransformerLanguageModel import *
from text_vae.core.Autoencoder import *
from .FunnelTransformer import *
from collections import defaultdict
from einops import rearrange
from torch.distributions import kl_divergence

from text_vae.core.Transformers import TransformerLayer, positional_encodings_like, autoregressive_decode_transformer
from .core.Distributions import ConditionalGaussian


@dataclass
class AutoregressiveAutoencoderHparams(TransformerLanguageModelHparams, ContinuousAutoencoderHparams):
    num_scales: int = 3

@dataclass
class AutoregressiveAutoencoderState:
    ground_truth: Optional[Tensor] = None
    encoder_output: Optional[FunnelTransformerOutput] = None
    decoder_output: Optional[FunnelTransformerOutput] = None

    priors: List[Normal] = field(default_factory=list)      # p(z)
    posteriors: List[Normal] = field(default_factory=list)  # q(z|x)
    latents: List[Tensor] = field(default_factory=list)
    stats: Dict[str, Tensor] = field(default_factory=lambda: defaultdict(float))


class AutoregressiveAutoencoder(ContinuousAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(AutoregressiveAutoencoder, self).__init__(hparams)

        encoder_depth = hparams.d_model
        self.encoder = FunnelTransformer(
            FunnelTransformerHparams(
                block_sizes=[2] * hparams.num_scales,
                scaling_factors=[2] * (hparams.num_scales - 1),
                d_model=encoder_depth,
                num_heads=hparams.num_heads
            )
        )
        self.encoder.input_layer[-1].p = 0.5
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(d_model=encoder_depth, num_heads=hparams.num_heads, causal=True, use_cross_attention=True)
            for _ in range(hparams.num_scales * 2)
        ])
        self.posteriors = nn.ModuleList([
            ConditionalGaussian(encoder_depth, hparams.latent_depth)
            for _ in range(hparams.num_scales)
        ])
        self.priors = nn.ModuleList([
            ConditionalGaussian(hparams.latent_depth, hparams.latent_depth, zero_initialized=True)
            for _ in range(hparams.num_scales - 1)
        ])
        self.latent_upsamplers = nn.ModuleList([
            nn.Linear(hparams.latent_depth, hparams.d_model)
            for _ in range(hparams.num_scales)
        ])
        input_embedding = self.encoder.input_layer[0].codebook
        input_embedding.data *= encoder_depth ** -0.5

        output_embedding = nn.Linear(encoder_depth, input_embedding.shape[0])
        output_embedding.weight = input_embedding

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.35),  # Incentivizes model to output a higher entropy distribution
            nn.Linear(encoder_depth, encoder_depth),
            nn.GELU(),
            nn.LayerNorm(encoder_depth),
            output_embedding
        )

    # Returns posterior distribution over the latent space
    def forward(self, batch: Dict[str, Any], **kwargs) -> AutoregressiveAutoencoderState:
        vae_state = AutoregressiveAutoencoderState()
        vae_state.ground_truth = batch['token_ids']
        vae_state.encoder_output = self.encoder(vae_state.ground_truth, batch['padding_mask'])

        for i, posterior_getter in enumerate(self.posteriors):
            encoder_state = vae_state.encoder_output.hidden_states[i]

            posterior = posterior_getter(encoder_state[:, 0])
            vae_state.latents += [posterior.rsample()]
            vae_state.posteriors += [posterior]

        vae_state.latents.reverse()
        vae_state.posteriors.reverse()
        return vae_state

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        vae_state = self.forward(batch)

        # Decoder uses absolute positional encodings, unlike the encoder
        x = vae_state.encoder_output.embedded_input
        x = x + positional_encodings_like(x)
        context = torch.stack([upsampler(z) for z, upsampler in zip(vae_state.latents, self.latent_upsamplers)], dim=-2)

        for i, (layer, upsampler) in enumerate(zip(self.decoder_layers, self.latent_upsamplers)):
            x = layer(x, cross_attn_kv=context, padding_mask=batch['padding_mask'])

        logits = self.output_layer(x)

        # Compute the KL divergences
        for i, (z, posterior) in enumerate(zip(vae_state.latents, vae_state.posteriors)):
            prior = self.get_base_prior() if i == 0 else self.priors[i - 1](z)
            vae_state.stats['kl'] += kl_divergence(prior, posterior).sum(dim=-1)

        # Sum NLL over tokens, but don't reduce the batch dimension just yet
        reconstruction_loss = F.cross_entropy(
            input=rearrange(logits[:, :-1], 'b l v -> b v l'),  # Remove final [SEP] token
            target=batch['token_ids'][:, 1:],                   # Remove initial [CLS] token
            ignore_index=0,
            reduction='none'
        ).sum(dim=-1)

        # Log the entropy of the model's probability distribution over words to see how confident it is
        self.log('logit_entropy', (logits * F.softmax(logits, dim=-1)).sum(dim=-1).mean())

        # We want to log the total, summed, KL and NLL for comparison with other text-based VAEs
        token_counts = batch['padding_mask'].logical_not().sum(dim=-1)
        self.log('total_kl', vae_state.stats['kl'].mean())
        self.log('total_nll', reconstruction_loss.mean())

        # But for the loss we divide by the sequence lengths
        nll_per_token = (reconstruction_loss / token_counts).mean()
        nll_per_word = (reconstruction_loss / batch['word_count']).mean()
        vae_state.stats['kl'] = (vae_state.stats['kl'] / token_counts).mean()
        loss = nll_per_token + self.hparams.kl_weight * vae_state.stats['kl']

        prefix = kwargs.get('log_prefix') or 'train_'
        self.log_dict({
            'kl': vae_state.stats['kl'],
            'nll': nll_per_token,
            'ppl': nll_per_word.exp(),
            prefix + 'loss': loss
        })
        result = {'loss': loss}
        if not kwargs.get('loss_only'):
            result['logits'] = logits

        return result

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        return self.training_step(batch, batch_index, log_prefix='val_', loss_only=True)

    # Returns an outer list where each inner list corresponds to one data sample, and each element in the inner list
    # is a Normal distribution object for a given level in the latent variable hierarchy
    def compute_posteriors(self, batch: Dict[str, Any]) -> List[List[Normal]]:
        batched_dists = self.forward(batch).posteriors  # Posteriors whose parameters are tensors with batch dims

        # List of lists of Normal distribution objects
        dists = [
            [
                torch.cat([mu, sigma], dim=-1)
                for mu, sigma, in zip(dist.mean.split(1, dim=0), dist.stddev.split(1, dim=0))
            ]
            for dist in batched_dists
        ]
        return list(map(list, zip(*dists)))  # Transpose the inner and outer lists

    def decoder_forward(self, x: Tensor, context: Tensor, padding_mask: Tensor = None):
        for layer in self.decoder_layers:
            x = layer(x, cross_attn_kv=context, padding_mask=padding_mask)

        return x

    def sample(self, max_length: int, count: int = 1, **kwargs):
        latents = [self.get_base_prior().rsample([count, max_length])]
        for prior_layer in self.priors:
            prior = prior_layer(latents[-1])
            latents += [prior.rsample()]

        context = torch.stack([upsampler(z) for z, upsampler in zip(latents, self.latent_upsamplers)], dim=-2)
        return autoregressive_decode_transformer(
            strategy=GenerationStrategy.SamplingTopK,
            transformer_callable=self.decoder_forward,
            logit_callable=self.output_layer,
            embedding=self.encoder.input_layer,
            context=context,
            d_model=self.hparams.d_model,
            start_symbol=self.start_token,
            end_symbol=self.end_token
        )

    def decoder_requires_grad_(self, value: bool):
        self.decoder.requires_grad_(value)
