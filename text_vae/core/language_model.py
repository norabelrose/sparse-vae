import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from abc import ABC
from dataclasses import dataclass
from einops import rearrange
from functools import partial
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import *
from .padded_tensor import PaddedTensor
from ..train_callbacks import UnconditionalSampler


@dataclass
class LanguageModelHparams(ABC):
    batch_size: int = 0                 # Just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 50.0
    init_scale: Optional[float] = 0.02  # Stddev of Gaussian weight initialization. If None, default PyTorch init. is used
    lr: float = 2.5e-4                  # GPT-2 learning rate
    lr_decay_steps: Optional[int] = 250_000
    lr_plateau_patience: Optional[int] = None  # If non-None, ReduceLROnPlateau scheduler is used w/ specified patience
    warmup_steps: int = 1000

    # If beta2 ie set to 0.0, we just use SGD (with momentum set to beta1)- useful for replicating papers that use it
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-6              # We need to use 1e-6 since 1e-8 underflows to 0 on fp16
    lasso_penalty: float = 0.0          # L1 norm penalty
    weight_decay: float = 0.01

    vocab_size: int = 2 ** 15           # Maximum number of tokens representable with signed 16-bit integers
    start_token: Optional[int] = None   # If None, it's read off the datamodule's Tokenizer object
    end_token: Optional[int] = None

    divide_loss_by_length: bool = True
    early_stopping_metric: str = 'val_nll'
    log_samples: bool = True


# noinspection PyMethodMayBeStatic
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
            ModelCheckpoint(monitor=metric, mode='min')
        ]
        return callbacks + [UnconditionalSampler()] if self.hparams.log_samples else callbacks

    # The callbacks need to have access to a tokenizer, so let's get a reference to the datamodule's tokenizer.
    def setup(self, stage: str):
        datamodule = self.trainer.datamodule if stage in ('fit', 'test') else self.datamodule
        self.tokenizer = datamodule.tokenizer

        if not self.start_token and not self.end_token:
            vocab = self.tokenizer.get_vocab()
            self.start_token = vocab['[CLS]']
            self.end_token = vocab['[SEP]']

    def configure_optimizers(self, lr: float = None, params = None):
        beta1, beta2 = self.hparams.adam_beta1, self.hparams.adam_beta2
        if beta2 == 0.0:
            opt = torch.optim.SGD(
                params or self.parameters(),
                lr=lr or self.hparams.lr,
                momentum=beta1
            )
        else:
            try:
                from deepspeed.ops.adam import FusedAdam as Adam
            except ImportError:
                print("Couldn't import fused Adam kernel from DeepSpeed, falling back on PyTorch version.")
                from torch.optim import Adam

            opt = Adam(
                params or self.parameters(),
                lr=lr or self.hparams.lr,
                betas=(beta1, beta2),
                weight_decay=self.hparams.weight_decay,
                eps=self.hparams.adam_eps
            )

        # This is mainly just here to reproduce the Lagging Inference Networks paper (He et al. 2019) which uses a
        # ReduceLROnPlateau-type learning rate schedule
        on_plateau_patience = self.hparams.lr_plateau_patience
        if on_plateau_patience is not None:
            lr_dict = {
                'scheduler': ReduceLROnPlateau(opt, factor=0.5, patience=on_plateau_patience),
                'monitor': self.hparams.early_stopping_metric,
                'interval': 'epoch'
            }
        else:
            lr_lambda = get_cosine_decay_with_warmup_schedule(self.hparams.lr_decay_steps, self.hparams.warmup_steps)
            lr_dict = {
                'scheduler': LambdaLR(opt, lr_lambda),
                'interval': 'step'
            }

        return [opt], [lr_dict]

    def initialize_weights(self):
        scale = self.hparams.init_scale
        if scale is None:   # Use default PyTorch initialization
            return

        # Default BERT weight initialization
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                continue

            if hasattr(module, 'weight'):
                module.weight.data.normal_(0.0, scale)

            bias = getattr(module, 'bias', None)
            bias = getattr(bias, 'data', None)
            if bias is not None:
                bias.zero_()

    def get_nll(self, logits: Tensor, labels: Tensor, lengths: Tensor = None, stage: str = 'train'):
        # Expand the labels across any extra batch dimensions that might exist in the logits tensor
        if extra_dims := logits.ndim - labels.ndim - 1:
            labels = labels.expand(*logits.shape[:extra_dims], *labels.shape)

        log_probs = rearrange(logits, '... n c -> n c ...').log_softmax(dim=1)
        labels = rearrange(labels, '... n -> n ...')

        divide_by_length = self.hparams.divide_loss_by_length
        if not divide_by_length:
            nll = F.nll_loss(log_probs, labels, ignore_index=0, reduction='sum') / labels.shape[1]
        else:
            nll = F.nll_loss(log_probs, labels, ignore_index=0)

        if stage == 'val' and lengths is not None:
            # We have to do this ugly hack because nll_loss() throws an error when reduction='none' and the
            # logits tensor has more than 2 ** 31 elements; see PyTorch issue #24401
            log_probs[..., 0] = 0.0
            nll_sum = -log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1).sum(dim=0)
            bits_per_char = (nll_sum / math.log(2)) / lengths
            self.log('val_bpc', bits_per_char.mean())

        self.log(stage + '_nll', nll)
        return nll

    # These implementations are used by LSTMLanguageModel and TransformerLanguageModel, but are overriden by others
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        logits = self.forward(batch)[..., :-1, :]
        return self.get_nll(logits, batch['token_ids'][..., 1:])

    # def optimizer_step(self, *args, **kwargs):
    #     # We apply the LASSO penalty here and not in on_after_backward to make sure that we don't apply the
    #     # penalty more than once when using gradient accumulation
    #     if penalty := self.hparams.lasso_penalty:
    #         for module in self.modules():
    #             # Don't shrink the LayerNorm parameters (especially the scale parameters) because the
    #             # 'default' configuration is 1.0, not zero. Also, empirically, including these parameters
    #             # tends to reduce performance quite considerably and makes training less stable.
    #             if isinstance(module, nn.LayerNorm):
    #                 continue
#
    #             for param in module.parameters(recurse=False):
    #                 if (grad := param.grad) is not None:
    #                     grad.add_(grad.sign(), alpha=penalty)  # d/dx[abs(x)] = sign(x)
#
    #     super().optimizer_step(*args, **kwargs)

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        logits = self.forward(batch)[..., :-1, :]
        return self.get_nll(logits, batch['token_ids'][..., 1:], lengths=batch.get('num_char'), stage='val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.validation_step(batch, batch_index)

    # Called by UnconditionalSampler callback
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
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
