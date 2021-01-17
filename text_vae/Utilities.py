from copy import deepcopy
from typing import *


class SizedIterator:
    def __init__(self, raw_iter, size: int):
        self.raw_iter = iter(raw_iter)
        self.size = size

    def __iter__(self):
        return self
    
    def __length_hint__(self):
        return self.size
    
    def __next__(self):
        return next(self.raw_iter)


# Get a new dictionary with a subset of the keys
T = TypeVar('T', bound=Mapping)
def select(source_dict: T, *keys: str) -> T:
    return {k: source_dict[k] for k in keys}


T = TypeVar('T', bound=Mapping)
def mutate(old_dict: T, **kwargs: Any) -> T:
    new_dict = deepcopy(old_dict)
    new_dict.update(kwargs)
    return new_dict


# Get a new dictionary with a subset of (possibly transformed) values from the old dictionary. If you want a
# key-value pair to simply be copied from the old dict, list the key in `args`. If you want to rename a value,
# or get transformed values using arbitrary Python expressions, use **kwargs. Inspired by the R `dplyr` package.
T = TypeVar('T', bound=Mapping)
def transmute(big_dict: T, *args: str, **kwargs: str) -> T:
    new_dict = select(big_dict, *args)
    eval_ctx = big_dict if isinstance(big_dict, dict) else dict(**big_dict)
    new_dict.update({new_k: eval(expr, eval_ctx) for new_k, expr in kwargs.items()})
    return new_dict
