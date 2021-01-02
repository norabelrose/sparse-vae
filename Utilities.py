from typing import *

import torch
from torch import Tensor


# Get a new dictionary with a subset of the keys
def select(source_dict: dict, *keys: str) -> dict:
    return {source_dict[k] for k in keys}


# Copy a dictionary and set (possibly new) key value pairs from kwargs. Inspired by the R `dplyr` package.
def mutate(old_dict: dict, **kwargs: Any) -> dict:
    new_dict = old_dict.copy()
    new_dict.update(kwargs)
    return new_dict


# Get a new dictionary with a subset of (possibly transformed) values from the old dictionary. If you want a
# key-value pair to simply be copied from the old dict, list the key in `args`. If you want to rename a value,
# or get transformed values using arbitrary Python expressions, use **kwargs. Inspired by the R `dplyr` package.
def transmute(big_dict: dict, *args: str, **kwargs: str) -> dict:
    dict_cls = type(big_dict)  # May be an AttributeDict or other dict subclass
    return dict_cls(
        **{k: big_dict[k] for k in args},
        **{new_k: (eval(expr, big_dict) if isinstance(expr, str) else expr) for new_k, expr in kwargs.items()}
    )


def gaussian_kl_divergence(mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor) -> Tensor:
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor) -> Tensor:
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu
