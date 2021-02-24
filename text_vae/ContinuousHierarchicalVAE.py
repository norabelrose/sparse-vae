from .core import ConditionalGaussian
from .HierarchicalVAE import *
from dataclasses import dataclass
from collections import defaultdict
from torch.distributions import Categorical
from numpy import prod
from torch.distributions import kl_divergence
import math
import torch


@dataclass
class ContinuousHierarchicalVAEHparams(HierarchicalVAEHparams, ContinuousVAEHparams):
    pass

@dataclass
class ContinuousHierarchicalVAEState(HierarchicalVAEState):
    latents: List[Tensor] = field(default_factory=list)
    posteriors: List[Normal] = field(default_factory=list)  # q(z|x)

    p_of_x_given_z: Optional[Categorical] = None
    stats: Dict[str, Tensor] = field(default_factory=lambda: defaultdict(float))


class ContinuousHierarchicalVAE(HierarchicalVAE, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super(ContinuousHierarchicalVAE, self).__init__(hparams)

        encoder_hparams = hparams.encoder
        num_latent_scales = len(encoder_hparams.scaling_factors)

        self.samplers = nn.ModuleList([
            ContinuousLatentSampler(
                latent_depth=hparams.latent_depth,
                overt_depth=encoder_hparams.d_model,
                is_base=i == 0,
                reduce_length=not hparams.use_long_latents
            )
            for i in range(num_latent_scales)
        ])

    def forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        return self.encoder_forward(batch)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Tensor]:
        result = self.reconstruct(batch)
        self.log('iw_loss', self.get_loss_monte_carlo(batch))
        return self.compute_loss_for_step(result, 'val')

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        z = self.samplers[block_idx + 1](dec_state, vae_state, enc_state=enc_state)

        if not self.hparams.use_long_latents:
            # Replace the [CLS] token with the latent vector in an out-of-place, autograd-friendly way
            return dec_state.where(not_cls_mask_like(dec_state), z.unsqueeze(-2))
        else:
            return dec_state + z

    def compute_loss_for_step(self, result: ContinuousHierarchicalVAEState, step: str):
        # Figure out the correct denominator for the loss
        if self.hparams.include_padding_positions:
            num_tokens = result.p_of_x_given_z.logits.shape[-2]
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

    def compute_latents(self, batch: Dict[str, Any]) -> List[List[Normal]]:
        vae_state = self.reconstruct(batch)
        batched_dists = vae_state.posteriors  # Posteriors whose parameters are tensors with batch dims

        # List where each element is a list of (mu, sigma) tuples; each inner list is for just one level latent
        dists = [
            [
                Normal(loc=mu, scale=sigma) for mu, sigma, in
                zip(dist.mean.split(1, dim=0), dist.stddev.split(1, dim=0))
            ]
            for dist in batched_dists
        ]
        return [list(x) for x in zip(*dists)]  # Transpose the inner and outer lists

    # Runs a forward pass through the encoder and the decoder and returns a dict with the sum total KL and NLL
    # for each sample in the batch. Note that these tensors must be normalized before computing the loss.
    def reconstruct(self, batch: Dict[str, Any], **kwargs) -> ContinuousHierarchicalVAEState:
        encoder_output = self.encoder_forward(batch)

        vae_state = ContinuousHierarchicalVAEState()
        vae_state.ground_truth = encoder_output.original_ids
        vae_state.decoder_input = self.samplers[0](encoder_output.final_state, vae_state)
        vae_state.encoder_states = encoder_output.hidden_states[:0:-1]

        vae_state = self.decoder_forward(vae_state, **kwargs)

        log_probs = vae_state.p_of_x_given_z.log_prob(vae_state.ground_truth)
        vae_state.stats['nll'] = -log_probs.flatten(start_dim=1).sum(dim=-1)

        return vae_state

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

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, temperature: float = 1.0) -> Tensor:
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

        base_sampler = self.samplers[0]
        base_sampler.eval()

        initial_state = ContinuousHierarchicalVAEState()
        initial_state.decoder_input = base_sampler.forward(
            x=torch.empty([count, seed_length, funnel_hparams.d_model]),
            vae_state=initial_state
        )
        base_sampler.train()

        vae_output = self.decoder_forward(
            vae_state=initial_state,
            padding_mask=None,
            temperature=temperature
        )
        return self.decode_logits(vae_output.p_of_x_given_z.logits)


class ContinuousLatentSampler(nn.Module):
    def __init__(self, latent_depth: int, overt_depth: int, reduce_length: bool = False, is_base: bool = False):
        super(ContinuousLatentSampler, self).__init__()

        reduce_dim = -2 if reduce_length else None
        if is_base:
            # Register these as buffers so they are moved to the correct device automagically by PyTorch Lightning
            self.register_buffer('prior_mu', torch.zeros(latent_depth))
            self.register_buffer('prior_sigma', torch.ones(latent_depth))
        else:
            self.prior = ConditionalGaussian(overt_depth, latent_depth, reduce_dim=reduce_dim, zero_initialized=True)

        # We concatenate the encoder and decoder states depthwise to get the input for the posterior
        posterior_input_depth = overt_depth if is_base else overt_depth * 2
        self.is_base = is_base
        self.posterior = ConditionalGaussian(posterior_input_depth, latent_depth, reduce_dim=reduce_dim)
        self.upscaler = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_depth, overt_depth)
        )

    def get_std_prior(self) -> Normal:
        return Normal(self.prior_mu, self.prior_sigma)

    # TODO: Add support for sampling temperatures
    def forward(self, x: Tensor, vae_state: ContinuousHierarchicalVAEState, enc_state: Optional[Tensor] = None):
        # For fixed Standard Gaussian priors
        if self.is_base:
            new_shape = list(x.shape[:-1]) + [self.prior_mu.shape[-1]]
            prior = self.get_std_prior().expand(new_shape)  # noqa
        else:
            prior = self.prior(x)

        if self.training:
            if not self.is_base:
                assert enc_state is not None, "Need encoder state to parameterize the posterior"
                x = torch.cat([x, enc_state], dim=-1)

            posterior = self.posterior(x)
            z = posterior.rsample()

            # We sum over all dimensions except the batch dimension
            vae_state.stats['p(z)'] += prior.log_prob(z).flatten(start_dim=1).sum(dim=1)
            vae_state.stats['q(z|x)'] += posterior.log_prob(z).flatten(start_dim=1).sum(dim=1)
            vae_state.stats['kl'] += kl_divergence(prior, posterior).flatten(start_dim=1).sum(dim=1)

            vae_state.posteriors += [posterior]
        else:
            z = prior.sample()

        vae_state.latents += [z]
        return self.upscaler(z)
