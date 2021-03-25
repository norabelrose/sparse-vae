import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from abc import ABC
from dataclasses import dataclass
from functools import partial
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR
from typing import *
from ..train_callbacks import UnconditionalSampler


@dataclass
class LanguageModelHparams(ABC):
    batch_size: int = 0  # This is here just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 150.0
    lr: float = 1e-4
    lr_decay_steps: Optional[int] = 150_000
    warmup_steps: int = 1000
    adam_beta1: float = 0.9
    adam_eps: float = 1e-7  # We need to use 1e-7 and not the default 1e-8 since 1e-8 underflows to 0 on fp16
    weight_decay: float = 0.01

    vocab_size: int = 30522
    padding_idx: int = 0
    start_token: Optional[int] = None  # If None, it's read off the datamodule's Tokenizer object
    end_token: Optional[int] = None
    log_samples: bool = True


# noinspection PyMethodMayBeStatic
class LanguageModel(pl.LightningModule, ABC):
    def __init__(self, hparams: DictConfig):
        super(LanguageModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.example_input_array = {'batch': {
            'token_ids': torch.zeros(1, 256, dtype=torch.long),
            'padding_mask': torch.zeros(1, 256, dtype=torch.bool)}
        }
        self.tokenizer = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

    def configure_callbacks(self):
        return [UnconditionalSampler()] if self.hparams.log_samples else []

    # The callbacks need to have access to a tokenizer, so let's get a reference to the datamodule's tokenizer.
    def setup(self, stage: str):
        datamodule = self.trainer.datamodule if stage == 'fit' else self.datamodule
        self.tokenizer = datamodule.tokenizer

        if not self.start_token and not self.end_token:
            vocab = self.tokenizer.get_vocab()
            self.start_token = vocab['[CLS]']
            self.end_token = vocab['[SEP]']

    def configure_optimizers(self, lr: float = None, params = None):
        try:
            from deepspeed.ops.adam import FusedAdam as Adam
        except ImportError:
            print("Couldn't import fused Adam kernel from DeepSpeed, falling back on PyTorch version.")
            from torch.optim import Adam

        adam = Adam(
            params or self.parameters(),
            lr=lr or self.hparams.lr,
            betas=(self.hparams.adam_beta1, 0.999),
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps
        )
        lr_lambda = get_cosine_decay_with_warmup_schedule(self.hparams.lr_decay_steps, self.hparams.warmup_steps)

        return [adam], [{
            'scheduler': LambdaLR(adam, lr_lambda),
            'interval': 'step'
        }]

    def initialize_weights(self):
        # Default BERT weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(0.0, 0.02)

    # These implementations are used by LSTMLanguageModel and TransformerLanguageModel, but are overriden by
    # HierarchicalAutoencoder and TextFlow
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Tensor:
        logits = self.forward(batch)
        loss = F.cross_entropy(
            input=logits[:, :-1].flatten(0, 1),          # Remove final [SEP] token
            target=batch['token_ids'][:, 1:].flatten(),  # Remove initial [CLS] token
            ignore_index=self.hparams.padding_idx
        )

        # Log the entropy of the model's probability distribution over words to see how confident it is
        self.log('logit_entropy', (logits * F.softmax(logits, dim=-1)).sum(dim=-1).mean())
        self.log('train_nll' if not val else 'val_nll', loss, on_step=not val, on_epoch=val)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, val=True)

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
