from collections import defaultdict, deque
from numpy import prod
from torch import nn
from torch import Tensor
from .Utilities import *
from .AttentionState import AttentionState
from .Autoencoder import *
from .FunnelTransformer import FunnelTransformer, FunnelTransformerHparams
import math
import torch.nn.functional as F
import torch


@dataclass
class HierarchicalAutoencoderHparams(AutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        block_sizes=(4, 4, 4),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2)  # How much the hidden state is downsampled between each encoder block
    )
    # Indicates how many layers of latent variables should be conditioned on one another, and the Transformer layer
    # indices after which each latent variable will be sampled. If None, the stochastic depth will be set equal to the
    # number of Transformer layers in the encoder.
    stochastic_layers: Optional[List[int]] = None
    use_encoder_residual_connections: bool = True
    use_pretrained_encoder: bool = True

    grad_skip_threshold: float = 400.0
    include_padding_positions: bool = True


class HierarchicalAutoencoder(Autoencoder):
    def __init__(self, hparams: OmegaConf):
        super().__init__(hparams)

        # save_hyperparameters() stores the hparams in self.hparams and ensures they are saved to disk during training.
        self.save_hyperparameters(hparams)

        encoder_hparams = hparams.encoder
        num_layers = sum(encoder_hparams.block_sizes)
        stochastic_layers = hparams.stochastic_layers

        if not stochastic_layers:
            stochastic_layers = list(range(num_layers))
        else:
            assert len(stochastic_layers) <= num_layers, \
                "Number of stochastic layers must not exceed the number of Transformer layers in the encoder"

            stochastic_layers.sort()

        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(encoder_hparams)
        attn_state.shared = True
        decoder_hparams = mutate(
            encoder_hparams,
            block_sizes=encoder_hparams.block_sizes[::-1],  # Reverse the order of the blocks
            scaling_factors=encoder_hparams.scaling_factors[::-1],
            upsampling=True
        )
        self.encoder = FunnelTransformer(encoder_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)
        self.encoder.attention_state = attn_state
        self.decoder.attention_state = attn_state

        # Whether we should feed selected encoder states into the decoder to help it along
        overt_depth, latent_depth = encoder_hparams.d_model, hparams.latent_depth
        if self.hparams.use_encoder_residual_connections:
            posterior_input_depth = overt_depth * 2  # Concatenate the encoder and decoder states depthwise
        else:
            posterior_input_depth = overt_depth  # Just the decoder state

        def get_linear_with_gelu(input_dim, output_dim, zero_initialized: bool = False):
            linear = nn.Linear(input_dim, output_dim)
            if zero_initialized:
                linear.weight.data.zero_()

            return nn.Sequential(nn.GELU(), linear)

        def linears_with_gelu(input_dim, output_dim, layers: List[int], zero_initialized: bool = False):
            return nn.ModuleDict({
                str(layer_idx): get_linear_with_gelu(input_dim, output_dim, zero_initialized)
                for layer_idx in layers
            })

        self.distributions = nn.ModuleDict({
            # Output contains mu & log sigma. Note that initializing the weights of the prior to zero means that at
            # the start of training, it will output 0 mean, 0 log sigma (i.e. 1 sigma) independent of the input
            'p(z)': linears_with_gelu(latent_depth, latent_depth * 2, stochastic_layers[1:], zero_initialized=True),
            'q(z|x)': linears_with_gelu(posterior_input_depth, latent_depth * 2, stochastic_layers)
        })
        self.latent_upsample = linears_with_gelu(latent_depth, overt_depth, stochastic_layers)

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

    def forward(self, batch: Dict[str, Any]) -> List[Tensor]:
        self.encoder.attention_state.configure_for_input(batch['token_ids'], batch['padding_mask'])
        return self.encoder(batch['token_ids'], padding_mask=batch['padding_mask'])

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        result = self.reconstruct(batch)

        neg_elbo = result['kl'] + result['nll']  # Negative ELBO (evidence lower bound)
        self.log_dict({'kl': result['kl'], 'nll': result['nll'], 'train_loss': neg_elbo})
        # setattr(self, 'last_loss', neg_elbo.detach())  # Hack
        return neg_elbo

    # Implements gradient skipping to stabilize training as described in 'Very Deep VAEs'
    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        # if grad_norm > self.hparams.grad_skip_threshold:
        #    self.zero_grad()

        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        nll = self.get_loss_monte_carlo(batch)
        self.log('val_loss', nll)
        return nll

    def test_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        nll = self.get_loss_monte_carlo(batch, num_samples=100)
        self.log('test_loss', nll)
        return nll

    # Runs a forward pass through the encoder and the decoder and returns a dict with the KL and NLL per token
    def reconstruct(self, batch: Dict[str, Any], return_logits: bool = False, **kwargs) -> Dict[str, Tensor]:
        # Figure out the correct denominator for the loss
        if self.hparams.include_padding_positions:
            ignore_idx = -100
            num_tokens = batch['token_ids'].shape[-1]
        else:
            # Each sample in the batch will have a different number of nonpadding tokens
            ignore_idx = 0
            num_tokens = batch['padding_mask'].sum(dim=-1)

        # Encoder forward pass
        encoder_out = self(batch)

        # Help out the decoder by giving it some of the states of the encoder
        if self.hparams.use_encoder_residual_connections:
            kwargs['encoder_states'] = encoder_out.pop('hidden_states')

        output = self.decoder_forward(encoder_out['output'], **kwargs)
        if return_logits:
            return output['logits']

        nll_target = output['logits'].transpose(-2, -1)  # nll_loss wants the dimensions permuted for some reason
        raw_nll = F.nll_loss(nll_target, target=batch['token_ids'], ignore_index=ignore_idx, reduction='none')
        output['nll'] = raw_nll.sum(dim=-1)
        del output['logits']

        # Divide by the number of tokens and then average over samples in the batch
        return {name: (stat / num_tokens).mean() for name, stat in output.items()}

    # Called both by reconstruct() and sample()
    def decoder_forward(self, x: Optional[Tensor],
                        padding_mask: Tensor = None,
                        batch_dims: List[int] = None,
                        clamped_latents: Mapping[int, Tensor] = None,
                        encoder_states: Mapping[int, Tensor] = None,
                        temperature: float = 1.0,
                        return_latents: bool = False,
                        return_log_probs: bool = False
                        ) -> Dict[str, Any]:
        # We use a coroutine so that we can unobtrusively insert ourselves into Funnel Transformer's forward pass
        # and replace its hidden state with samples from our prior or posterior distributions
        self.decoder.attention_state.upsampling = True
        coroutine = self.decoder.hidden_state_coroutine(x, padding_mask=padding_mask,
                                                        states_to_yield=self.hparams.stochastic_layers)

        latents = deque(maxlen=None if return_latents else 1)
        num_blocks = len(self.decoder.hparams.block_sizes)
        stats = defaultdict(float)

        for block_idx, layer_idx, hidden_state in coroutine:
            # Final output- hidden_state is a dict here
            if block_idx == -1:
                hidden_state.update(stats)
                return hidden_state

            # In the first stochastic layer, the prior is just a standard diagonal Gaussian:
            if not latents:
                # Sampling unconditionally
                if hidden_state is not None:
                    batch_dims = list(hidden_state.shape[:2])

                new_shape = batch_dims + [self.hparams.latent_depth]
                prior = self.get_base_prior().expand(new_shape)  # noqa

            # But in subsequent layers, it is conditioned on the previous layer of latent variables:
            else:
                prior = self.get_distribution_for_tensor('p(z)', layer_idx, latents[-1], temperature=temperature)

            if encoder_states is None and hidden_state is None:  # Sample unconditionally
                z = clamped_latents.get(layer_idx) if clamped_latents else None
                if z is None:
                    z = prior.rsample()

                posterior = None
            else:
                # Corresponding block in the encoder, counting in reverse
                if encoder_states is not None:
                    encoder_block_idx = num_blocks - block_idx - 1
                    encoder_state = encoder_states[encoder_block_idx]
                    posterior_input = torch.cat([x, encoder_state], dim=-1)

                # We're just given the final output of the encoder
                else:
                    posterior_input = hidden_state

                # Sample conditioned on the input from the encoder
                posterior = self.get_distribution_for_tensor('q(z|x)', layer_idx, posterior_input,
                                                             temperature=temperature)
                z = posterior.rsample()

            self.update_stats_dict(prior, posterior, z, stats, return_log_probs=return_log_probs)

            # Needed by the next stochastic layer up to parameterize its prior distribution
            latents.append(z)

            reconstruction = self.latent_upsample[str(layer_idx)](z)
            out_state = hidden_state + reconstruction if hidden_state is not None else reconstruction

            coroutine.send(out_state)

    def update_stats_dict(self, prior: Normal, posterior: Normal, z: Tensor, stats: Dict[str, Any],
                          return_log_probs: bool = False):
        # Marginalize over successive layers of latent variables
        if return_log_probs:
            stats['p(z)'] += prior.log_prob(z).sum(dim=(1, 2))
            stats['q(z|x)'] += posterior.log_prob(z).sum(dim=(1, 2))

        # Update the running totals of the KL divergences
        if self.training:
            kl_tensor = torch.distributions.kl_divergence(prior, posterior)

            if not self.hparams.include_padding_positions:
                # Gives us the appropriately scaled mask for the current block
                padding_mask = self.decoder.attention_state.get_padding_mask().unsqueeze(-1)
                kl_tensor.masked_fill_(padding_mask, 0.0)

            stats['kl'] += kl_tensor.sum(dim=(1, 2))  # Sum over seq len and latent dim

    # Used for both the prior 'p(z)' and the posterior 'q(z|x)'
    def get_distribution_for_tensor(self, name: str, layer_idx: int, x: Tensor, temperature: float) -> Normal:
        dist_linear = self.distributions[name][str(layer_idx)]
        mu, logsigma = dist_linear(x).chunk(2, dim=-1)
        stddev = logsigma.exp()

        if temperature != 1.0:
            stddev *= temperature

        return Normal(mu, stddev)

    # Get a tighter estimate for the KL and NLL of some input using Monte Carlo importance sampling
    def get_loss_monte_carlo(self, batch: Dict[str, Any], num_samples: int = 10):
        batch_size = batch['token_ids'].shape[0]

        marginal_prob_list = []
        for _ in range(num_samples):
            stats = self.reconstruct(batch, return_log_probs=True)

            log_joint_prob = stats['p(z)'] - stats['nll']  # p(z) * p(x|z) = p(x,z)
            log_marginal_prob = log_joint_prob - stats['q(z|x)']  # p(x,z) / p(z|x) = p(x)
            marginal_prob_list.append(log_marginal_prob)

        # Average over the Monte Carlo samples
        log_marginal_prob = torch.stack(marginal_prob_list, dim=0)
        avg_log_prob = log_marginal_prob.logsumexp(dim=0).sub(math.log(num_samples * batch_size))
        return -avg_log_prob

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, return_latents: bool = False,
               clamped_latents: Optional[Mapping[int, Tensor]] = None,
               temperature: float = 1.0) -> Tensor:
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder_hparams
        seed_length = max_length // prod(funnel_hparams.scaling_factors)

        # (batch, seq_len, vocab_size)
        output_logits: Tensor = self.decoder_forward(
            x=None,
            padding_mask=None,
            batch_dims=[count, seed_length],
            clamped_latents=clamped_latents or {},  # Allow for style transfer-type experiments
            return_latents=return_latents,
            temperature=temperature
        )['logits']
        output_ids = output_logits.topk(top_k, dim=-1)

        return output_ids

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)
        self.distributions.requires_grad_(requires_grad)
