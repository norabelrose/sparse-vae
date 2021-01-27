from .Autoencoder import Autoencoder
from dataclasses import asdict, dataclass
from pathlib import Path
from pytorch_lightning.callbacks.base import Callback
from tokenizers import BertWordPieceTokenizer   # noqa
from typing import *


@dataclass
class UnconditionalSampler(Callback):
    num_samples: int = 5
    sample_max_len: int = 192
    sampling_temperature: float = 0.85  # Taken from Very Deep VAEs paper

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.__dict__.update(checkpointed_state)

    def on_save_checkpoint(self, trainer, pl_module) -> Dict[str, Any]:
        return asdict(self)

    def on_train_epoch_end(self, trainer, autoencoder: Autoencoder, outputs):
        samples = autoencoder.sample(self.sample_max_len, self.num_samples, temperature=self.sampling_temperature)
        samples = samples.tolist()  # Tensor -> List of lists of ints

        vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
        tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)
        samples = tokenizer.decode_batch(samples, skip_special_tokens=True)  # List of strings

        logger = trainer.logger.experiment
        for sample in samples:
            logger.add_text("sample_epoch", sample, global_step=trainer.current_epoch)
