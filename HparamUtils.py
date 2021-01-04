from copy import copy
from typing import *


# Get a new dictionary with a subset of the keys
T = TypeVar('T', bound=Mapping)
def select(source_dict: T, *keys: str) -> T:
    dict_cls = type(source_dict)  # May be an AttributeDict or other dict subclass
    return dict_cls(**{k: source_dict[k] for k in keys})


# Copy a dictionary and set (possibly new) key value pairs from kwargs. Inspired by the R `dplyr` package.
T = TypeVar('T', bound=Mapping)
def mutate(old_dict: T, **kwargs: Any) -> T:
    new_dict = copy(old_dict)
    new_dict.update(**kwargs)
    return new_dict


# Get a new dictionary with a subset of (possibly transformed) values from the old dictionary. If you want a
# key-value pair to simply be copied from the old dict, list the key in `args`. If you want to rename a value,
# or get transformed values using arbitrary Python expressions, use **kwargs. Inspired by the R `dplyr` package.
T = TypeVar('T', bound=Mapping)
def transmute(big_dict: T, *args: str, **kwargs: str) -> T:
    new_dict = select(big_dict, *args)
    new_dict.update({new_k: eval(expr, big_dict) for new_k, expr in kwargs.items()})
    return new_dict
