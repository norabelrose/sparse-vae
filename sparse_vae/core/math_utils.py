from functools import lru_cache
from torch import Tensor
from torch.distributions import Normal, kl_divergence
import math
import torch
import torch.nn.functional as F


def reconstruction_bleu(output: Tensor, originals: Tensor):
    assert output.ndim == originals.ndim == 1  # and output.element_size() == originals.element_size() == 2
    output, originals = output.long(), originals.long()

    def compute_ngram_score(x, ground_truth):
        x, ground_truth = x.sort().values, ground_truth.sort().values
        true_ngrams, true_counts = ground_truth.unique_consecutive(return_counts=True)
        x_ngrams, x_counts = x.unique_consecutive(return_counts=True)

        combined = torch.cat([x_ngrams, true_ngrams], dim=-1).sort().values
        matching_ngrams, matching_counts = combined.unique_consecutive(return_counts=True)
        matching_ngrams = matching_ngrams[matching_counts > 1]

        true_counts = true_counts[torch.searchsorted(true_ngrams, matching_ngrams)]
        x_counts = x_counts[torch.searchsorted(x_ngrams, matching_ngrams)]

        clipped_counts = x_counts.minimum(true_counts)
        return clipped_counts.sum(dim=-1) / x.shape[-1]

    def get_ngrams(x, n = 4):
        x = x.clone()
        yield x     # Unigrams

        for i in range(1, n + 1):
            shifted = x[:-i]
            shifted += x[i:] << (16 * i)
            yield shifted

    scores = torch.stack([compute_ngram_score(x, y) for x, y in zip(get_ngrams(output), get_ngrams(originals))])
    return scores.log().mean().exp()


# Batch dot product across the last dimension
def bdot(a, b):
    return (a[..., None, :] @ b[..., None]).flatten(-3)

# KL(P || N(0, I))
def std_gaussian_kl(p: Normal):
    inv_var = p.variance.reciprocal()
    return 0.5 * (-inv_var.log().sum(dim=-1) + inv_var.sum(dim=-1) + bdot(p.mean ** 2, inv_var) - inv_var.shape[-1])

# Approximates KL(q(z)||p(z)) where p(z) is N(0, I)
def marginal_kl(posteriors: Normal, num_samples: int = 10):
    samples = posteriors.rsample([num_samples])

    cross_probs = posteriors.log_prob(samples[:, :, None]).sum(dim=-1)
    marginal_prob = cross_probs.logsumexp(dim=2) - math.log(samples.shape[1])

    sample_prob = -0.5 * (samples.pow(2.0).sum(dim=-1).mean() + samples.shape[-1] * math.log(2 * math.pi))
    return sample_prob - marginal_prob.mean()

# Convenience function to efficiently compute the entropies of a batch of multivariate Gaussians,
# reducing across the last dimension.
def multivariate_gaussian_entropy(gaussians: Normal) -> Tensor:
    return gaussians.scale.log().sum(dim=-1) + 0.5 * (math.log(2 * math.pi) + 1.0) * gaussians.scale.shape[-1]

# H(P, Q)
def multivariate_gaussian_cross_entropy(p: Normal, q: Normal) -> Tensor:
    var_p = p.variance.flatten(1)
    inv_var_q = q.variance.flatten(1).T.reciprocal()

    mu_p, mu_q = p.mean.view(*var_p.shape), q.mean.view(*var_p.shape)
    weighted_dists = (mu_p ** 2 @ inv_var_q) - 2 * mu_p @ (mu_p.T * inv_var_q) + (mu_q ** 2 @ inv_var_q)

    return q.scale.log().sum(dim=-1) + 0.5 * (var_p @ inv_var_q + weighted_dists + var_p.shape[-1] * math.log(2 * math.pi))

def pairwise_gaussian_cross_entropy(gaussians: Normal) -> Tensor:
    trace_log_sigma = gaussians.scale.flatten(1).log().sum(dim=-1)

    var_p = gaussians.variance.flatten(1)  # Covariance matrix diagonals
    inv_var_q = var_p.T.reciprocal()

    mu_p = gaussians.mean.flatten(1)
    weighted_norms = mu_p ** 2 @ inv_var_q
    weighted_dists = weighted_norms - 2 * mu_p @ (mu_p.T * inv_var_q) + weighted_norms.diagonal()

    return trace_log_sigma + 0.5 * (var_p @ inv_var_q + weighted_dists + var_p.shape[-1] * math.log(2 * math.pi))


# Efficiently compute all the pairwise KL divergences for a batch of multivariate Gaussians. Returns
# a square matrix of the form [P, Q]; i.e. the KL from the first to the second Gaussian is at [0, 1]
def pairwise_gaussian_kl(gaussians: Normal) -> Tensor:
    trace_log_sigma = gaussians.scale.flatten(1).log().sum(dim=-1)
    log_sigma_ratio = trace_log_sigma - trace_log_sigma[:, None]

    var_p = gaussians.variance.flatten(1)     # Covariance matrix diagonals
    inv_var_q = var_p.T.reciprocal()

    mu_p = gaussians.mean.flatten(1)
    weighted_norms = mu_p ** 2 @ inv_var_q
    weighted_dists = weighted_norms - 2 * mu_p @ (mu_p.T * inv_var_q) + weighted_norms.diagonal()

    return log_sigma_ratio + 0.5 * (var_p @ inv_var_q + weighted_dists - var_p.shape[-1])


# Analytically computes an unbiased estimate of the the squared maximum mean discrepancy between the
# distribution from which x was drawn and a standard multivariate Gaussian, using a Gaussian RBF kernel.
# If standardize = True, we divide by the standard error of the estimator under the null hypothesis.
def analytic_gaussian_rbf_mmd_sq(x, standardize = True):
    assert x.ndim == 2
    n, d = x.shape

    kernel_var = 0.125 * d
    normalizer = (kernel_var / (1 + kernel_var)) ** (d / 2)
    first_term = (kernel_var / (2 + kernel_var)) ** (d / 2)
    second_term = torch.exp(-0.5 * x.pow(2.0).sum(dim=-1) / (1 + kernel_var)).mean()
    third_term = torch.exp(-0.5 * F.pdist(x) ** 2 / kernel_var).mean()
    mmd_sq = first_term - 2 * normalizer * second_term + third_term

    if standardize:
        ugly_term = 2 * (kernel_var ** 2 / ((1 + kernel_var) * (3 + kernel_var))) ** (d / 2)
        variance = (2 / (n * (n - 1))) * (first_term ** 2 + (kernel_var / (4 + kernel_var)) ** (d / 2) - ugly_term)
        mmd_sq = mmd_sq / variance ** 0.5

    return mmd_sq

# Analytically computes an unbiased estimate of the the squared maximum mean discrepancy between the
# distribution from which x was drawn and a diagonal Gaussian with the specified mean and variance,
# using a Gaussian RBF kernel.
def custom_gaussian_rbf_mmd_sq(x, mean, var, standardize = True):
    assert x.ndim == 2
    n, d = x.shape

    kernel_var = 0.125 * d        # Python scalar
    var_sum = kernel_var + var    # [d] or [n, d]

    # Log for numerical stability
    kernel_logvar = math.log(kernel_var)
    cov_logdet1 = var_sum.log().sum(dim=-1) * 0.5
    cov_logdet2 = torch.log(2 * var + kernel_var).sum(dim=-1) * 0.5
    normalizer = torch.exp(kernel_logvar * d / 2 - cov_logdet1)
    first_term = torch.exp(kernel_logvar * d / 2 - cov_logdet2)
    second_term = torch.exp(-0.5 * torch.sum((x - mean) ** 2 / var_sum, dim=-1)).mean()
    third_term = torch.exp(-0.5 * F.pdist(x) ** 2 / kernel_var).mean()
    mmd_sq = first_term - 2 * normalizer * second_term + third_term

    if standardize:
        cov_logdet3 = torch.log(3 * var + kernel_var).sum(dim=-1) * 0.5
        cov_logdet4 = torch.log(4 * var + kernel_var).sum(dim=-1) * 0.5
        ugly_term = torch.exp(math.log(2) + kernel_logvar * d - cov_logdet1 - cov_logdet3)

        variance = 2 / (n * (n - 1)) * (first_term ** 2 + torch.exp(kernel_logvar * d / 2 - cov_logdet4) - ugly_term)
        mmd_sq = mmd_sq / variance ** 0.5

    return mmd_sq


def gaussian_imq_mmd_sq(x):
    # We average over 7 IMQ kernels with different scales as in the WAE paper
    c = 2 * x.shape[-1]
    scales = torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], device=x.device).view(-1, 1, 1) * c

    # Helper functions- use the same prior samples every time to reduce the variance of the estimator
    @lru_cache()
    def gaussian_prior_samples(d, device, n=1000):
        raw_samples = torch.randn(n, d, device=device)

        # Artificially ensure the samples have *exactly* unit variance and zero mean
        var, mean = torch.var_mean(raw_samples)
        return (raw_samples - mean) / var.sqrt()

    # The expectation E[k(x, y)] under a Gaussian prior for the IMQ kernel is not analytically computable,
    # but we'll need this quantity every for single MMD computation so we cache a Monte Carlo estimate
    @lru_cache()
    def gaussian_prior_imq_avg_distance(d, device, n=1000):
        samples = gaussian_prior_samples(d, device, n)
        return torch.mean(scales / (scales + F.pdist(samples) ** 2))

    first_term = torch.mean(scales / (scales + F.pdist(x) ** 2))

    prior_samples = gaussian_prior_samples(x.shape[-1], x.device)
    prior_interactions = torch.sum(x[None] * prior_samples[:, None], dim=-1)
    prior_dists = x.pow(2.0).sum(dim=-1)[None] - 2 * prior_interactions + x.shape[-1]   # [batch, prior samples]
    middle_term = 2 * torch.mean(scales / (scales + prior_dists))

    return first_term - middle_term + gaussian_prior_imq_avg_distance(x.shape[-1], x.device)
