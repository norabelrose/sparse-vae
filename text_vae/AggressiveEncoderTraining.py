from dataclasses import dataclass, field
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor


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

    # Don't check the mutual information metric during the validation sanity check
    def on_sanity_check_start(self, trainer, autoencoder):
        self._aggressive_stage_complete = True

    def on_sanity_check_end(self, trainer, autoencoder):
        self._aggressive_stage_complete = False

    def on_train_start(self, trainer, autoencoder):
        autoencoder.decoder.requires_grad_(False)

    def on_train_batch_end(self, trainer, autoencoder, outputs, batch, batch_index, dataloader_idx):
        if self._aggressive_stage_complete:
            return

        inner_loop_step = batch_index - self._last_decoder_update
        new_loss = _to_scalar(trainer.logged_metrics['train_loss'])

        update_decoder = (
            # We've updated the encoder for 10 steps AND
            inner_loop_step > 0 and inner_loop_step % self.min_inner_loop_steps == 0 and
            # The loss hasn't improved, OR we hit the upper limit for aggressive encoder updates
            (new_loss > self._last_loss or inner_loop_step >= self.max_inner_loop_steps)
        )
        autoencoder.decoder.requires_grad_(update_decoder)
        if update_decoder:
            self._last_decoder_update = batch_index

    def on_validation_end(self, trainer, autoencoder):
        if self._aggressive_stage_complete:
            return

        new_mutual_info = _to_scalar(trainer.callback_metrics['mutual_info'])
        if new_mutual_info < self._last_mutual_info:
            autoencoder.print("Aggressive encoder training complete.")

            self._aggressive_stage_complete = True
            autoencoder.decoder.requires_grad_(True)
        else:
            self._last_mutual_info = new_mutual_info

# Convenience method
def _to_scalar(x):
    return x.item() if isinstance(x, Tensor) else x
