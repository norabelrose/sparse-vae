from typing import *

import torch
from torch import Tensor

DataType = NewType('DataType', torch.dtype)
Device = NewType('Device', torch.device)


# Get a new dictionary inverting the keys and values from a source dictionary
def invert(source: dict) -> dict:
    return dict((value, key) for key, value in source.items())


# Performs a series of replace operations on a string defined by a list of tuples of strings
def replace_all(string: str, mappings: List[Tuple[str, str]]) -> str:
    for old, new in mappings:
        string = string.replace(old, new)

    return string


def gaussian_kl_divergence(mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor) -> Tensor:
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor) -> Tensor:
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


T = TypeVar('T', bound=Union[List[Tensor], Tensor])

def slice_tensors(x: T, axis: int, start: int = None, stop: int = None, step: int = None) -> T:
    rank = x.dim() if torch.is_tensor(x) else x[0].dim()
    assert axis < rank
    axis = axis % rank  # for negative indices

    indices = []
    for i in range(rank):
        if i == axis:
            indices.append(slice(start, stop, step))
            break
        else:
            indices.append(slice(None))  # Include all elements along this axis; equivalent to : in x[:, 5]

    return x[indices] if torch.is_tensor(x) else [tensor[indices] for tensor in x]

# Used to make 'args' objects for the old FunnelTFM model
class DynamicDict(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        try:
            self[key] = value
        except KeyError:
            super().__setattr__(key, value)
