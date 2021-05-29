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
        x_ngrams , x_counts = x.unique_consecutive(return_counts=True)

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

# Attempt to compute a not-so-biased estimate of the mutual info by averaging a lower and an upper bound
def mutual_info_monte_carlo(posteriors: Normal, num_samples: int = 10):
    samples = posteriors.rsample([num_samples])
    batch, dim = samples.shape[1], samples.shape[-1]

    # Including the posterior from which each sample came from to estimate q(z) gives a lower bound on the MI
    cross_probs = posteriors.log_prob(samples[:, :, None]).sum(dim=-1)
    full_marginals = cross_probs.logsumexp(dim=2) - math.log(batch)

    # Use a 'leave-one-out' estimator for the marginal distribution, excluding the posterior from which each
    # sample came from, in order to get an upper bound on the MI
    huge = torch.finfo(cross_probs.dtype).max
    loo_probs = cross_probs - torch.eye(batch, device=cross_probs.device).unsqueeze(-1) * huge
    loo_marginals = loo_probs.logsumexp(dim=2) - math.log(batch - 1)
    neg_entropy = -posteriors.entropy().sum(dim=-1).mean()

    return neg_entropy - (full_marginals.mean() + loo_marginals.mean()) / 2

def mutual_info_jackknife(posteriors: Normal, num_samples: int = 10):
    neg_entropy = -posteriors.entropy().sum(dim=-1).mean()
    return neg_entropy - marginal_entropy_jackknife(posteriors, num_samples)

def marginal_entropy_jackknife(posteriors: Normal, num_samples: int = 10):
    samples = posteriors.rsample([num_samples])
    batch = samples.shape[1]

    cross_log_probs = posteriors.log_prob(samples[:, :, None]).sum(dim=-1)
    naive_sums = cross_log_probs.logsumexp(dim=2)
    loo_sums = naive_sums[:, :, None].sub(cross_log_probs).expm1().add(1e-7).log() + cross_log_probs

    naive_marginal = naive_sums.mean() - math.log(batch)
    loo_marginal = loo_sums.mean() - math.log(batch - 1)
    jackknife_bias = (batch - 1) * (loo_marginal - naive_marginal)
    return naive_marginal - jackknife_bias

def mutual_info_variational(posteriors: Normal):
    mu_var, mu_mean = torch.var_mean(posteriors.mean, dim=0, unbiased=False)
    var_mean = posteriors.variance.mean(dim=0)
    q_star = Normal(loc=mu_mean, scale=torch.sqrt(var_mean + mu_var))
    return kl_divergence(posteriors, q_star).sum(dim=-1).mean()

def gaussian_mixture_std_kl_approx(p: Normal) -> Tensor:
    batch, dim = p.scale.flatten(1).shape
    z_a_alpha = pairwise_gaussian_log_product_normalizer(p)
    lf_f = z_a_alpha.logsumexp(dim=-1) - math.log(batch)

    p_neg_entropy = pairwise_gaussian_kl(p).neg().logsumexp(dim=-1) - math.log(batch)
    normalizer = -0.5 * (math.log(2.0 * math.pi) * dim - p.variance.log1p().sum(dim=-1) + p.mean.pow(2.0).sum(dim=-1))
    return (lf_f.mean() + std_gaussian_kl(p).mean() + p_neg_entropy.mean() - normalizer.mean()) / 2

# Upper bound on KL(P || N(0, I)) where P is a Gaussian mixture with equal component weights
def gaussian_mixture_std_kl_upper_bound(p: Normal) -> Tensor:
    z_a_alpha = pairwise_gaussian_log_product_normalizer(p)
    lf_f = z_a_alpha.logsumexp(dim=-1) - math.log(z_a_alpha.shape[0])            # Upper bound
    lf_g = -std_gaussian_kl(p).mean() - multivariate_gaussian_entropy(p).mean()  # Lower bound

    return lf_f.mean() - lf_g.mean()

def gaussian_mixture_std_kl_lower_bound(p: Normal) -> Tensor:
    batch, dim = p.scale.flatten(1).shape
    p_neg_entropy = pairwise_gaussian_kl(p).neg().logsumexp(dim=-1) - math.log(batch)
    std_normalizer = -0.5 * (math.log(2.0 * math.pi) * dim - p.variance.log1p().sum(dim=-1) + p.mean.pow(2.0).sum(dim=-1))
    return p_neg_entropy.mean() - std_normalizer.mean()

# Variational approximation to KL(P || N(0, I)) where P is a Gaussian mixture with equal component weights
def gaussian_mixture_std_kl_variational(p: Normal) -> Tensor:
    neg_kls = -pairwise_gaussian_kl(p)
    p_neg_entropy = neg_kls.logsumexp(dim=-1) - math.log(neg_kls.shape[0])
    return p_neg_entropy.mean() + std_gaussian_kl(p).mean()

# Compute the log of the normalizing constant for all the pairwise products
def pairwise_gaussian_log_product_normalizer(gaussians: Normal) -> Tensor:
    var = gaussians.variance                    # [A, dim]
    var_sums = var[:, None] + var               # [A, B, dim]
    inv_var_sums = var_sums.reciprocal()
    log_cov_det = var_sums.log().sum(dim=-1)    # [A, B]

    mean = gaussians.mean                       # [A, dim]
    mean_diffs = mean[:, None] - mean           # [A, B, dim]
    quadratic_form = bdot(mean_diffs, mean_diffs * inv_var_sums)
    return -0.5 * (var.shape[-1] * math.log(2 * math.pi) + log_cov_det + quadratic_form)

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

def multivariate_gaussian_kl(p: Normal, q: Normal) -> Tensor:
    log_sigma_ratio = q.scale.log().sum(dim=-1) - p.scale.log().sum(dim=-1)

    var_p = p.variance.flatten(1)  # Covariance matrix diagonals
    inv_var_q = q.variance.flatten(1).T.reciprocal()

    mu_p, mu_q = p.mean.view(*var_p.shape), q.mean.view(*var_p.shape)
    weighted_dists = (mu_p ** 2 @ inv_var_q) - 2 * mu_p @ (mu_p.T * inv_var_q) + (mu_q ** 2 @ inv_var_q)

    return log_sigma_ratio + 0.5 * (var_p @ inv_var_q + weighted_dists - var_p.shape[-1])

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
