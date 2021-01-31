from dataclasses import *
from pytorch_lightning.callbacks.base import Callback
from re import sub
from typing import *

# Maps snake case names of AutoencoderCallback classes to the actual class objects
AutoencoderCallbackRegistry = {}

# e.g. KLAnnealing -> kl_annealing
def camel_to_snake(name):
    name = sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


# Abstract superclass
class AutoencoderCallback(Callback):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        snake_case = camel_to_snake(cls.__name__)
        AutoencoderCallbackRegistry[snake_case] = cls

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.__dict__.update(checkpointed_state)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)  # noqa
