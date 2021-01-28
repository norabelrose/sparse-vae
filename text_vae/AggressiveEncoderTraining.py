from .Autoencoder import Autoencoder
from dataclasses import asdict, dataclass, field
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor
from typing import *


# Update encoder parameters more frequently than those of the decoder until the mutual information between
# z and x stops going up (from "Lagging Inference Networks and Posterior Collapse in Variational Autoencoders")
@dataclass
class AggressiveEncoderTraining(Callback):
    min_inner_loop_steps: int = 10
    max_inner_loop_steps: int = 100    # Maximum number of encoder updates before we update the decoder

    _aggressive_stage_complete: bool = field(default=False, init=False)
    _last_decoder_update: int = field(default=0, init=False)
    _last_loss: float = field(default=0.0, init=False)
    _last_mutual_info: float = field(default=0.0, init=False)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.__dict__.update(checkpointed_state)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        return asdict(self)

    # Don't check the mutual information metric during the validation sanity check
    def on_sanity_check_start(self, trainer, autoencoder: Autoencoder):
        self._aggressive_stage_complete = True

    def on_sanity_check_end(self, trainer, autoencoder: Autoencoder):
        self._aggressive_stage_complete = False

    def on_train_start(self, trainer, autoencoder: Autoencoder):
        autoencoder.decoder_requires_grad_(False)

    def on_after_backward(self, trainer, autoencoder: Autoencoder):
        if self._aggressive_stage_complete:
            return

        cur_step = autoencoder.global_step

        inner_loop_step = cur_step - self._last_decoder_update
        new_loss = _to_scalar(getattr(autoencoder, 'last_loss'))

        update_decoder = (
            # We've updated the encoder for 10 steps AND
            inner_loop_step > 0 and inner_loop_step % self.min_inner_loop_steps == 0 and
            # The loss hasn't improved, OR we hit the upper limit for aggressive encoder updates
            (new_loss > self._last_loss or inner_loop_step >= self.max_inner_loop_steps)
        )
        autoencoder.decoder_requires_grad_(update_decoder)
        if update_decoder:
            self._last_decoder_update = cur_step
            self._last_loss = new_loss

    def on_validation_end(self, trainer, autoencoder: Autoencoder):
        if self._aggressive_stage_complete:
            return

        raw_mutual_info = trainer.callback_metrics.get('mutual_info')
        if raw_mutual_info is None and trainer.current_epoch >= 4:
            self.end_aggressive_training(autoencoder)
            return

        new_mutual_info = _to_scalar(raw_mutual_info)
        if new_mutual_info < self._last_mutual_info:
            self.end_aggressive_training(autoencoder)
        else:
            self._last_mutual_info = new_mutual_info

    def end_aggressive_training(self, autoencoder: Autoencoder):
        autoencoder.print("Aggressive encoder training complete.")

        self._aggressive_stage_complete = True
        autoencoder.decoder_requires_grad_(True)

# Convenience method
def _to_scalar(x):
    return x.item() if isinstance(x, Tensor) else x
