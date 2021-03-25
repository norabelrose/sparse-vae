from dataclasses import *
from pytorch_lightning.callbacks.base import Callback
from typing import *

# Abstract superclass
class AutoencoderCallback(Callback):
    def on_load_checkpoint(self, ckpt_state: Dict[str, Any]):
        self.__dict__.update(ckpt_state)

    def on_save_checkpoint(self, trainer, pl_module, ckpt_state) -> Dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)  # noqa
