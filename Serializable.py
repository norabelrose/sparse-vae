from __future__ import annotations
from pathlib import Path
from typing import *

import json

SerializableClassRegistry: Dict[str, type] = {}  # A dictionary from class names to class objects

# Convenience for saving and loading objects from JSON
class Serializable:
    autosave_dir: ClassVar[str] = '~/.autosave'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        class_name = cls.__name__
        SerializableClassRegistry[class_name] = cls  # No serializable type goes unregistered

    @staticmethod
    def from_dict(frozen: dict) -> Serializable:
        try:
            class_name = frozen['__classname__']
            class_obj = SerializableClassRegistry[class_name]
        except KeyError:
            raise ValueError("Invalid state dictionary")

        new_obj = class_obj()

        for key, value in frozen.items():
            if not hasattr(new_obj, key):
                continue

            if isinstance(value, dict) and '__classname__' in value:
                value = Serializable.from_dict(value)

            setattr(new_obj, key, value)

        return new_obj

    def to_dict(self) -> dict:
        return _serialize(self)

    @staticmethod
    def load(path: Path) -> Serializable:
        with path.open() as handle:
            data = json.load(handle)

        return Serializable.from_dict(data)

    def save(self, path: Path):
        # Make the directory if it doesn't already exist
        json_dir = path.parent
        if not json_dir.exists():
            json_dir.mkdir(parents=True)

        with json_dir.open('w') as handle:
            json.dump(self.to_dict(), handle, indent=4, sort_keys=True)

    # Saves the generator's state in a unique JSON file in /.autosave folder
    def autosave(self):
        autosave_path = Path(self.autosave_dir).expanduser()
        autosave_path = autosave_path / self.__class__.__module__ / f'{type(self).__name__}_{id(self)}.json'

        self.save(autosave_path)


def _serialize(obj):
    kind = type(obj)

    # Atomic JSON primitives
    if kind in (str, int, float, bool, type(None)):
        return obj

    # Recursively serialize the contents of lists, and tuples
    if kind in (list, tuple):
        return kind(filter(None, (_serialize(x) for x in obj)))

    # Same for dicts
    if kind is dict:
        new_dict = dict((_serialize(k), _serialize(v)) for k, v in obj)
        for k, v in new_dict.items():
            if not v:
                del new_dict[k]

    if issubclass(kind, Serializable):
        state_dict = _serialize(obj.__dict__)
        state_dict['__classname__'] = kind.__name__
        return state_dict

    # Skip objects that we can't serialize
    return None
