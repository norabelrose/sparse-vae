from __future__ import annotations
from typing import *
import json
import logging
import os
import sys
import torch

Tensor = NewType('Tensor', torch.tensor)
DataType = NewType('DataType', torch.dtype)
Device = NewType('Device', torch.device)


# Get a new dictionary inverting the keys and values from a source dictionary
def invert(source: dict) -> dict:
    return dict((value, key) for key, value in source.items())

N = TypeVar('N', Union[int, float])

def product(numbers: Iterable[N]) -> N:
    result = 1
    for x in numbers:
        result *= x

    return x


# Performs a series of replace operations on a string defined by a list of tuples of strings
def replace_all(string: str, mappings: List[Tuple[str, str]]) -> str:
    for old, new in mappings:
        string = string.replace(old, new)

    return string

T = TypeVar('T', Union[List[Tensor], Tensor])

def slice_tensors(x: T, axis: int, start: int = None, stop: int = None, step: int = None) -> T:
    rank = x.dim() if isinstance(x, torch.tensor) else x[0].dim()
    assert axis < rank

    indices = []
    for i in range(rank):
        if i == axis:
            indices.append(slice(start, stop, step))
            break
        else:
            indices.append(slice(None))  # Include all elements along this axis; equivalent to : in x[:, 5]

    return x[indices] if isinstance(x, torch.tensor) else [tensor[indices] for tensor in x]

# Convenience for saving and loading objects from JSON
class SerializableObject:
    @classmethod
    def from_dict(cls, frozen: dict) -> SerializableObject:
        new_object = cls()
        
        # Recursively de-serialize SerializableObjects in this dict
        for key, value in frozen.items():
            # Replace the dict with the corresponding SerializableObject
            if isinstance(value, dict) and "__classname__" in value:
                class_obj = getattr(sys.modules[__name__], value["__classname__"])
                if not issubclass(class_obj, SerializableObject):
                    continue
                
                value = class_obj.from_dict(value)
            
            # Assuming all attributes should be initialized in the __init__ method- if it doesn't exist
            # now, we won't add it to the object.
            if hasattr(new_object, key):
                setattr(new_object, key, value)
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"from_dict: Ignoring non-existent attribute {key} for {cls.__name__}.")

        return new_object
    
    def to_dict(self) -> dict:
        data = self.__dict__
        data["__classname__"] = self.__class__.__name__
        
        for key, value in data.items():
            # Replace the SerializableObject with the corresponding dict
            if issubclass(value.__class__, SerializableObject):
                data[key] = value.to_dict()
        
        return data

    @classmethod
    def from_json(cls, path: str) -> SerializableObject:
        with open(path, 'r') as handle:
            data = json.load(handle)
            return cls.from_dict(data)

    def save(self, path: str):
        # Make the directory if it doesn't already exist
        json_dir = os.path.dirname(path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open(json_dir, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=4, sort_keys=True)
