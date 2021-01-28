from dataclasses import asdict, dataclass
from pytorch_lightning.callbacks.base import Callback
from typing import *
from .Autoencoder import Autoencoder


@dataclass
class KLAnnealing(Callback):
    kl_weight_start: float = 0.1
    kl_weight_end: float = 1.0
    num_annealing_steps: int = 150_000

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.__dict__.update(checkpointed_state)

    def on_save_checkpoint(self, trainer, autoencoder: Autoencoder) -> Dict[str, Any]:
        return asdict(self)

    def on_after_backward(self, trainer, autoencoder: Autoencoder):
        cur_step = autoencoder.global_step
        max_steps = self.num_annealing_steps
        if cur_step > max_steps:
            return

        progress = cur_step / max_steps
        total_distance = self.kl_weight_end - self.kl_weight_start
        autoencoder.hparams.kl_weight = self.kl_weight_start + total_distance * progress
