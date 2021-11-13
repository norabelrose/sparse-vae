import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from abc import ABC
from dataclasses import dataclass
from functools import partial
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR
from triton import cdiv
from typing import Dict, Optional
from .padded_tensor import PaddedTensor
from .rectified_adam import RAdam
from .text_sampling_callback import TextSamplingCallback


@dataclass
class LanguageModelHparams(ABC):
    grad_clip_threshold: float = 5.0
    init_scale: Optional[float] = 0.02  # Stddev of Gaussian weight init. If None, default PyTorch init. is used

    base_batch_size: int = 100_000      # Base batch size for sqrt learning rate scaling
    lr: float = 2e-4
    lr_decay_steps: Optional[int] = 250_000

    start_token: Optional[int] = None   # If None, it's read off the datamodule's Tokenizer object
    end_token: Optional[int] = None

    early_stopping_metric: str = 'val_nll'
    log_samples: bool = True


class LanguageModel(pl.LightningModule, ABC):
    def __init__(self, hparams: DictConfig):
        super(LanguageModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.example_input_array = {'batch': {
            'token_ids': PaddedTensor.from_raw(torch.zeros(1, 256, dtype=torch.long))}
        }
        self.tokenizer = None
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token

    def configure_callbacks(self):
        metric = self.hparams.early_stopping_metric
        callbacks = [
            EarlyStopping(monitor=metric, mode='min'),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor=metric, mode='min')
        ]
        return callbacks + [TextSamplingCallback()] if self.hparams.log_samples else callbacks

    # The callbacks need to have access to a tokenizer, so let's get a reference to the datamodule's tokenizer.
    def setup(self, stage: str):
        datamodule = self.trainer.datamodule if stage in ('fit', 'test') else self.datamodule
        self.tokenizer = datamodule.tokenizer
        self.register_buffer('token_weights', datamodule.bytes_per_token.half())

        if not self.start_token and not self.end_token:
            vocab = self.tokenizer.get_vocab()
            self.start_token = vocab['[CLS]']
            self.end_token = vocab['[SEP]']

    def configure_optimizers(self):
        batch_size = self.trainer.datamodule.hparams.tokens_per_batch * self.trainer.accumulate_grad_batches
        lr_scale_factor = (batch_size / self.hparams.base_batch_size) ** 0.5    # sqrt lr scaling
        # opt = FusedLamb(self.parameters(), lr=self.hparams.lr * lr_scale_factor, weight_decay=0.01)
        opt = RAdam(self.parameters(), lr=self.hparams.lr * lr_scale_factor, weight_decay=0.01)
        lr_lambda = partial(cosine_decay, self.hparams.lr_decay_steps)
        # lr_lambda = get_cosine_decay_with_warmup_schedule(self.hparams.lr_decay_steps, 4000)
        return [opt], [{
            'scheduler': LambdaLR(opt, lr_lambda),
            'interval': 'step'
        }]

    def initialize_weights(self):
        scale = self.hparams.init_scale
        if scale is None:   # Use default PyTorch initialization
            return

        # Default BERT weight initialization
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                continue

            if isinstance(module, (nn.Embedding, nn.Linear)):
                module.weight.data.normal_(0.0, scale)

            bias = getattr(module, 'bias', None)
            bias = getattr(bias, 'data', None)
            if bias is not None:
                bias.zero_()

    def get_nll(self, logits: Tensor, labels: Tensor, stage: str = 'train', bytes_per_token: Tensor = None):
        # Expand the labels across any extra batch dimensions that might exist in the logits tensor
        if extra_dims := logits.ndim - labels.ndim - 1:
            labels = labels.expand(*logits.shape[:extra_dims], *labels.shape)

        nll = robust_cross_entropy(logits, labels)
        # nll = F.cross_entropy(logits, labels, ignore_index=0)

        if stage == 'val' and bytes_per_token is not None:
            # logits = rearrange(logits, 'batch length vocab -> batch vocab length')
            nats_per_byte = robust_cross_entropy(logits, labels, weight=self.token_weights) * bytes_per_token
            # nats_per_byte = F.cross_entropy(logits, labels, weight=self.token_weights, ignore_index=0)
            self.log('val_bpb', nats_per_byte / math.log(2))

        self.log(stage + '_nll', nll)
        return nll

    # These implementations are used by LSTMLanguageModel and TransformerLanguageModel, but are overriden by others
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        logits = self.forward(batch)[..., :-1, :]
        return self.get_nll(logits, batch['token_ids'][..., 1:].long())

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        logits = self.forward(batch)[..., :-1, :]
        return self.get_nll(logits, batch['token_ids'][..., 1:].long(), stage='val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.validation_step(batch, batch_index)

    # Called by UnconditionalSampler callback
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        return None

def cosine_decay(decay_steps: int, cur_step: int):
    progress = cur_step / max(1, decay_steps)
    if progress >= 1.0:
        print("Learning rate decayed to 0.0. Halting training.")
        raise KeyboardInterrupt

    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

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

def robust_cross_entropy(logits, labels, weight = None):
    # logits, labels = logits.flatten(end_dim=1), labels.flatten()
    chunks = cdiv(logits.numel(), 2 ** 30)
    if chunks == 1:
        return F.cross_entropy(logits.flatten(end_dim=1), labels.flatten(), ignore_index=0, weight=weight)
    else:
        return torch.stack([
            F.cross_entropy(logit_chunk.flatten(end_dim=1), label_chunk.flatten(), ignore_index=0, weight=weight)
            for logit_chunk, label_chunk in zip(logits.chunk(chunks, dim=-2), labels.chunk(chunks, dim=-1))
        ]).mean()
