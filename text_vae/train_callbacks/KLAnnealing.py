from .AutoencoderCallback import *


@dataclass
class KLAnnealing(AutoencoderCallback):
    kl_weight_start: float = 0.1
    kl_weight_end: float = 1.0
    num_annealing_steps: int = 50_000

    def on_train_start(self, trainer, autoencoder):
        autoencoder.hparams.kl_weight = self.kl_weight_start

    def on_after_backward(self, trainer, autoencoder):
        cur_step = autoencoder.global_step
        max_steps = self.num_annealing_steps
        if cur_step > max_steps:
            return

        progress = cur_step / max_steps
        total_distance = self.kl_weight_end - self.kl_weight_start
        autoencoder.hparams.kl_weight = self.kl_weight_start + total_distance * progress
