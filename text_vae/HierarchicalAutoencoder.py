from collections import defaultdict
from torch.distributions import Categorical, kl_divergence
from numpy import cumsum, prod
from .core.Autoencoder import *
from .FunnelTransformer import *
from .GenerationUtils import *
from .KNNLookupTable import *
from . import autoregressive_decode_transformer, ConditionalGaussian
import math
import torch


@dataclass
class HierarchicalAutoencoderHparams(ContinuousAutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(2, 2,),   # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(4,),  # How much the hidden state is downsampled between each encoder block
    )
    # Indicates how many groups of latent variables should be used. If None, the number will be set equal to
    # len(encoder.block_sizes) - 1
    num_latent_groups: Optional[int] = None

    codebook_size: int = 0
    decoder_input_dropout: float = 0.5
    tie_embedding_weights: bool = True
    use_autoregressive_decoder: bool = False
    use_encoder_residual_connections: bool = True
    use_long_latents: bool = True
    use_length_encodings: bool = False
    use_pretrained_encoder: bool = False

    include_padding_positions: bool = True

@dataclass
class HierarchicalAutoencoderState:
    ground_truth: Optional[Tensor] = None
    encoder_output: Optional[FunnelTransformerOutput] = None
    decoder_output: Optional[FunnelTransformerOutput] = None
    posteriors: Dict[int, Normal] = field(default_factory=dict)  # q(z|x); keys are layer indices
    latents: List[Tensor] = field(default_factory=list)

    p_of_x_given_z: Optional[Categorical] = None
    stats: Dict[str, Tensor] = field(default_factory=lambda: defaultdict(float))


class HierarchicalAutoencoder(ContinuousAutoencoder):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        encoder_hparams = hparams.encoder
        num_layers = sum(encoder_hparams.block_sizes)
        num_latent_groups = hparams.num_latent_groups

        if not num_latent_groups:
            self.latent_groups = [0] + list(cumsum(encoder_hparams.block_sizes[:-2]))
        else:
            assert 1 <= num_latent_groups <= num_layers

            # Evenly space the latent groups to the extent possible, biasing them toward lower resolutions
            stride, leftover = divmod(num_layers, num_latent_groups)
            self.latent_groups = list(range(0, num_layers - leftover, stride))

        # save_hyperparameters() stores the hparams in self.hparams and ensures they are saved to disk during training.
        self.save_hyperparameters(hparams)

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
        output_embedding = nn.Linear(encoder_hparams.d_model, self.tokenizer.get_vocab_size())
        self.output_layer = nn.Sequential(
            nn.Linear(encoder_hparams.d_model, encoder_hparams.d_model),
            nn.GELU(),
            nn.LayerNorm(encoder_hparams.d_model),
            output_embedding
        )

        if hparams.codebook_size:
            self.codebooks = nn.ModuleList([
                nn.Embedding(hparams.codebook_size, encoder_hparams.d_model)
                for _ in range(len(self.latent_groups))
            ])

        if hparams.encoder.positional_encoding_type == 'learned':
            self.positional_encodings = nn.Embedding(192, encoder_hparams.d_model)
            attn_state.learned_pos_encodings = self.positional_encodings.weight

        elif hparams.tie_embedding_weights:
            self.encoder.input_layer[0].weight.data *= encoder_hparams.d_model * -0.5
            output_embedding.weight = self.encoder.input_layer[0].weight

        if hparams.use_autoregressive_decoder:
            self.ar_decoder = FunnelLayer(hparams, use_cross_attention=True)

        def get_distributions(input_dim, output_dim, layers: List[int], zero_initialized: bool = False):
            return nn.ModuleDict({
                str(layer_idx): ConditionalGaussian(input_dim, output_dim, zero_initialized=zero_initialized)
                for layer_idx in layers
            })

        overt_depth, latent_depth = encoder_hparams.d_model, hparams.latent_depth
        # self.decoder_seed = nn.Parameter(torch.zeros())
        self.distributions = nn.ModuleDict({
            # Output is a Normal distribution object. Note initializing the weights of the prior to zero means that at
            # the start of training, it will output 0 mean, 0 log sigma (i.e. 1 sigma) independent of the input
            'p(z)': get_distributions(overt_depth, latent_depth, self.latent_groups[1:], zero_initialized=True),
            'q(z|x)': get_distributions(overt_depth, latent_depth, self.latent_groups)
        })
        self.latent_upsample = get_distributions(latent_depth, overt_depth, self.latent_groups)

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()
        else:
            # Scale the initializations of each layer by 1/sqrt(N) where N is the depth
            self.encoder.scale_parameters(depth=num_layers * 2)
            self.decoder.scale_parameters(depth=num_layers * 2)

    # Runs a forward pass through the encoder and computes the posterior distributions over the latent variables
    # (but does not actually sample the latents- you can do that yourself)
    def forward(self, batch: Dict[str, Any], **kwargs) -> HierarchicalAutoencoderState:
        x = batch['token_ids']
        self.encoder.attention_state.configure_for_input(x.shape[1], x.dtype, x.device, batch['padding_mask'])
        funnel_out = self.encoder(batch['token_ids'], padding_mask=batch['padding_mask'], **kwargs)
        funnel_state_iter = reversed(funnel_out.hidden_states)

        vae_state = HierarchicalAutoencoderState()
        vae_state.encoder_output = funnel_out
        vae_state.posteriors = {
            idx: self.get_distribution_for_tensor('q(z|x)', idx, next(funnel_state_iter), 1.0)
            for idx in self.latent_groups
        }
        return vae_state

    def compute_loss_for_step(self, result: HierarchicalAutoencoderState, step: str):
        # Figure out the correct denominator for the loss
        if self.hparams.include_padding_positions:
            num_tokens = result.decoder_output.final_state.shape[1]
        else:
            # Each sample in the batch will have a different number of nonpadding tokens
            num_tokens = self.decoder.attention_state.padding_mask.sum(dim=-1)

        # Normalize the KL, NLL, and p(x) and q(z|x) terms by dividing them by the number of tokens in the input.
        # Then take the average over all samples in the batch.
        for key, value in result.stats.items():
            result.stats[key] = (value / num_tokens).mean()

        # When the KL weight is at its default 1.0 setting, this is equal to the negative ELBO (evidence lower bound)
        loss = self.hparams.kl_weight * result.stats['kl'] + result.stats['nll']

        self.log_dict({'kl': result.stats['kl'], 'nll': result.stats['nll'], step + '_loss': loss})
        return {'loss': loss, 'output': result}

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'train')

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        self.log('iw_loss', self.get_loss_monte_carlo(batch))
        return self.compute_loss_for_step(result, 'val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'test')

    # Runs a forward pass through the encoder and the decoder and returns a dict with the sum total KL and NLL
    # for each sample in the batch. Note that these tensors must be normalized before computing the loss.
    def reconstruct(self, batch: Dict[str, Any], **kwargs) -> HierarchicalAutoencoderState:
        # Encoder forward pass
        vae_state = self.forward(batch)
        vae_state = self.decoder_forward(vae_state, **kwargs)

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        if not self.hparams.include_padding_positions:
            original_ids = batch.get('labels', batch['token_ids'])
            # Ignore padding by setting its probability to 1.0 (log probability 0.0)
            log_probs = log_probs.where(original_ids.unsqueeze(-1) != 0, 0.0)

        vae_state.stats['nll'] = -log_probs.flatten(start_dim=1).sum(dim=-1)
        return vae_state

    # Called both by reconstruct() and sample()
    def decoder_forward(self, vae_state: Optional[HierarchicalAutoencoderState],
                        padding_mask: Tensor = None,
                        batch_dims: List[int] = None,
                        clamped_latents: Mapping[int, Tensor] = None,
                        temperature: float = 1.0,
                        return_latents: bool = False,
                        return_log_probs: bool = False
                        ) -> HierarchicalAutoencoderState:
        if not vae_state:
            vae_state = HierarchicalAutoencoderState()
        if not clamped_latents:
            clamped_latents = {}

        # We use a coroutine so that we can unobtrusively insert ourselves into Funnel Transformer's forward pass
        # and replace its hidden state with samples from our prior or posterior distributions
        self.decoder.attention_state.upsampling = True
        encoder_output = vae_state.encoder_output
        coroutine = self.decoder.forward_coroutine(
            encoder_output.final_state if encoder_output else None,
            padding_mask=padding_mask,
            states_to_yield=self.latent_groups
        )

        for block_idx, layer_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                vae_state.decoder_output = decoder_state
                raw_output = decoder_state.final_state

                # If we're using autoregressive decoding, now is the time to run the AR Transformer layer
                if self.hparams.use_autoregressive_decoder:
                    raw_output = self.ar_decoder(
                        # Remove [SEP] token to keep the input the same length as the expected output
                        vae_state.encoder_output.embedded_input[:, :-1],
                        cross_attn_keys=raw_output
                    )

                logits = self.output_layer(raw_output)
                vae_state.p_of_x_given_z = Categorical(logits=logits)
                if encoder_output:
                    # Remove [CLS] token, thereby shifting everything to the left
                    if self.hparams.use_autoregressive_decoder:
                        vae_state.ground_truth = encoder_output.original_ids[:, 1:]
                    else:
                        vae_state.ground_truth = encoder_output.original_ids

                return vae_state

            # In the first stochastic layer, the prior is just a standard diagonal Gaussian:
            if layer_idx == 0:
                # Sampling unconditionally
                if decoder_state is not None:
                    batch_dims = list(decoder_state.shape[:2])

                new_shape = batch_dims + [self.hparams.latent_depth]
                prior = self.get_base_prior().expand(new_shape)  # noqa

            # But in subsequent layers, it is conditioned on the state of the previous decoder layer:
            else:
                prior = self.get_distribution_for_tensor('p(z)', layer_idx, decoder_state, temperature=temperature)

            if not vae_state.posteriors:  # Sample unconditionally
                z = clamped_latents.get(layer_idx)
                if z is None:
                    z = prior.rsample()

                posterior = None
            else:
                posterior = vae_state.posteriors[layer_idx]
                z = posterior.rsample()

            self.update_stats_dict(prior, posterior, z, vae_state.stats, return_log_probs=return_log_probs)
            if return_latents:
                vae_state.latents.append(z)

            reconstruction = self.latent_upsample[str(layer_idx)](z)
            if not self.hparams.use_long_latents:
                # Replace the [CLS] token with the latent vector in an out-of-place, autograd-friendly way
                out_state = decoder_state.where(not_cls_mask_like(decoder_state), z.unsqueeze(-2))
            else:
                out_state = decoder_state + reconstruction if decoder_state is not None else reconstruction
            coroutine.send(out_state)

    def update_stats_dict(self, prior: Normal, posterior: Normal, z: Tensor, stats: Dict[str, Any],
                          return_log_probs: bool = False):
        # Marginalize over successive layers of latent variables
        if return_log_probs:
            stats['p(z)'] += prior.log_prob(z).flatten(start_dim=1).sum(dim=1)
            stats['q(z|x)'] += posterior.log_prob(z).flatten(start_dim=1).sum(dim=1)

        # Update the running totals of the KL divergences
        if posterior:
            kl_tensor = kl_divergence(prior, posterior)

            if not self.hparams.include_padding_positions:
                # Gives us the appropriately scaled mask for the current block
                padding_mask = self.decoder.attention_state.get_padding_mask().unsqueeze(-1)
                kl_tensor.masked_fill_(padding_mask, 0.0)

            stats['kl'] += kl_tensor.flatten(start_dim=1).sum(dim=1)  # Sum over seq len and latent dim

    # Used for both the prior 'p(z)' and the posterior 'q(z|x)'
    def get_distribution_for_tensor(self, name: str, layer_idx: int, x: Tensor, temperature: float) -> Normal:
        cond_dist = self.distributions[name][str(layer_idx)]
        return cond_dist(x, reduce_dim=-2 if self.hparams.use_long_latents else None, temperature=temperature)

    # Get a tighter estimate for the KL and NLL of some input using Monte Carlo importance sampling
    @torch.no_grad()
    def get_loss_monte_carlo(self, batch: Dict[str, Any], num_samples: int = 10):
        import gc

        batch_size, seq_len = batch['token_ids'].shape
        marginal_prob_list = []

        for _ in range(num_samples):
            # This tends to be very memory intensive
            gc.collect()
            torch.cuda.empty_cache()

            stats = self.reconstruct(batch, return_log_probs=True).stats

            log_joint_prob = stats['p(z)'] - stats['nll']  # p(z) * p(x|z) = p(x,z)
            log_marginal_prob = log_joint_prob - stats['q(z|x)']  # p(x,z) / p(z|x) = p(x)
            marginal_prob_list.append(log_marginal_prob.detach())

        # Average over the Monte Carlo samples
        log_marginal_prob = torch.stack(marginal_prob_list, dim=0)
        avg_log_prob = log_marginal_prob.logsumexp(dim=0) - math.log(num_samples)
        return (-avg_log_prob / seq_len).mean()

    def compute_posteriors(self, batch: Dict[str, Any]) -> List[List[Normal]]:
        encoder_out = self.forward(batch)
        batched_dists = encoder_out.posteriors  # Posteriors whose parameters are tensors with batch dims

        # List where each element is a list of (mu, sigma) tuples; each inner list is for just one level latent
        dists = [
            [
                Normal(loc=mu, scale=sigma) for mu, sigma, in
                zip(dist.mean.split(1, dim=0), dist.stddev.split(1, dim=0))
            ]
            for layer_idx, dist in batched_dists.items()
        ]
        return [list(x) for x in zip(*dists)]  # Transpose the inner and outer lists

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, return_latents: bool = False,
               clamped_latents: Optional[Mapping[int, Tensor]] = None,
               temperature: float = 1.0) -> Tensor:
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder
        total_scaling = prod(funnel_hparams.scaling_factors)
        seed_length = max_length // total_scaling

        # (batch, seq_len, vocab_size)
        self.decoder.attention_state.configure_for_input(
            seq_len=total_scaling * seed_length,  # May not be the same as max_length if it doesn't divide evenly
            dtype=torch.long,
            device=self.device,
            padding_mask=None
        )
        vae_output = self.decoder_forward(
            vae_state=None,
            padding_mask=None,
            batch_dims=[count, seed_length],
            clamped_latents=clamped_latents or {},  # Allow for style transfer-type experiments
            return_latents=return_latents,
            temperature=temperature
        )

        if self.hparams.use_autoregressive_decoder:
            return autoregressive_decode_transformer(
                strategy=GenerationStrategy.Greedy,
                transformer_callable=self.ar_decoder,
                context=vae_output.decoder_output.final_state,
                embedding=self.encoder.input_layer,
                logit_callable=self.output_layer,
                max_length=max_length,
                d_model=self.hparams.encoder.hparams,
                start_symbol=self.start_token,
                end_symbol=self.end_token
            )
        else:
            return self.decode_logits(vae_output.p_of_x_given_z.logits)

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)
        # self.distributions.requires_grad_(requires_grad)
