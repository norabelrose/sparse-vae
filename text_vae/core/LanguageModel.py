import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from abc import ABC
from dataclasses import dataclass
from functools import partial
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import *


@dataclass
class LanguageModelHparams(ABC):
    batch_size: int = 0  # This is here just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 150.0
    lr: float = 1e-4
    lr_decay_steps: Optional[int] = 150_000
    vocab_size: int = 30522
    warmup_steps: int = 1000
    adam_beta1: float = 0.9
    adam_eps: float = 1e-8
    weight_decay: float = 0.01


# noinspection PyMethodMayBeStatic
class LanguageModel(pl.LightningModule, ABC):
    def __init__(self, hparams: DictConfig):
        super(LanguageModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.example_input_array = {'batch': {
            'token_ids': torch.zeros(1, 192, dtype=torch.long),
            'padding_mask': torch.zeros(1, 192, dtype=torch.float)}
        }
        self.tokenizer = None
        self.start_token = None
        self.end_token = None

    # The callbacks need to have access to a tokenizer, so let's get a reference to the datamodule's tokenizer.
    def on_train_start(self):
        self.tokenizer = self.trainer.datamodule.tokenizer

        if not self.start_token and not self.end_token:
            vocab = self.tokenizer.get_vocab()
            self.start_token = vocab['[CLS]']
            self.end_token = vocab['[SEP]']

    def configure_optimizers(self):
        adam = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.adam_beta1, 0.999),
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps
        )
        lr_lambda = get_cosine_decay_with_warmup_schedule(self.hparams.lr_decay_steps, self.hparams.warmup_steps)

        return [adam], [{
            'scheduler': LambdaLR(adam, lr_lambda),
            'interval': 'step'
        }]

    # These implementations are used by LSTMLanguageModel and TransformerLanguageModel, but are overriden by
    # HierarchicalAutoencoder and TextFlow
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Tensor:
        logits = self.forward(batch)
        loss = F.cross_entropy(
            input=logits[:, :-1].flatten(0, 1),          # Remove final [SEP] token
            target=batch['token_ids'][:, 1:].flatten(),  # Remove initial [CLS] token
            ignore_index=0
        )

        # Log the entropy of the model's probability distribution over words to see how confident it is
        self.log('logit_entropy', (logits * F.softmax(logits, dim=-1)).sum(dim=-1).mean())
        self.log('train_loss' if not val else 'val_loss', loss, on_step=not val, on_epoch=val)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, val=True)

    def decode_logits(self, logits: Tensor, min_length: int = 10, k: int = 1):
        # Make [SEP] infinitely unlikely until we hit the min length
        logits[:, :min_length, self.end_token] = -float('inf')

        output = logits.topk(k).indices
        return output.squeeze(-1) if k == 1 else output

    # Called by UnconditionalSampler callback
    def sample(self, max_length: int, count: int = 1, **kwargs):
        return None

# The actual lambda to pass into LambdaLR
def cosine_decay_with_warmup(decay_steps: int, warmup_steps: int, cur_step: int):
    if cur_step < warmup_steps:  # Warmup phase
        return cur_step / warmup_steps
    elif not decay_steps:  # Just fall back to a constant schedule if we don't know the decay steps
        return 1.0
    else:  # Cosine decay
        progress = (cur_step - warmup_steps) / max(1, decay_steps - warmup_steps)
        if progress >= 1.0:
            print("Learning rate decayed to 0.0. Halting training.")
            raise KeyboardInterrupt

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

# Convenience function
def get_cosine_decay_with_warmup_schedule(decay_steps: int, warmup_steps: int):
    return partial(cosine_decay_with_warmup, decay_steps, warmup_steps)
