from typing import *

import torch
from torch import Tensor


# Copy a dictionary and set (possibly new) key value pairs from kwargs. Inspired by the R `dplyr` package.
def mutate(old_dict: dict, **kwargs: Any) -> dict:
    new_dict = old_dict.copy()
    new_dict.update(kwargs)
    return new_dict


# Get a new dictionary with a subset of keys from the old dictionary. If you want a key name to stay the same,
# put it in `args`. To rename a key, use **kwargs. Inspired by the R `dplyr` package.
def transmute(big_dict: dict, *args: str, **kwargs: str) -> dict:
    return {**{k: big_dict[k] for k in args}, **{new_k: big_dict[old_k] for new_k, old_k in kwargs}}


def gaussian_kl_divergence(mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor) -> Tensor:
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor) -> Tensor:
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu
