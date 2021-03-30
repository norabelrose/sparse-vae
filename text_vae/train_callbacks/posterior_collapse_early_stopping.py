from .autoencoder_callback import *
from collections import deque


@dataclass
class PosteriorCollapseEarlyStopping(AutoencoderCallback):
    kl_threshold: float = 1e-2
    kl_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def on_train_batch_end(self, trainer, autoencoder, outputs, batch, batch_idx, dataloader_idx):
        new_kl = outputs[0][0]['extra'].get('kl')
        if new_kl is None:
            return

        self.kl_history.append(new_kl)
        if sum(self.kl_history) / len(self.kl_history) < self.kl_threshold:
            autoencoder.print("Posterior collapse detected. Aborting training.")
            raise KeyboardInterrupt
