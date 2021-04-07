import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from abc import ABC
from dataclasses import dataclass
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
    lr: float = 3e-4
    lr_decay_steps: Optional[int] = 150_000
    lr_plateau_patience: Optional[int] = None  # If non-None, ReduceLROnPlateau scheduler is used w/ specified patience
    warmup_steps: int = 1000

    # If beta2 ie set to 0.0, we just use SGD (with momentum set to beta1)- useful for replicating papers that use it
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-6              # We need to use 1e-6 since 1e-8 underflows to 0 on fp16
    weight_decay: float = 0.01

    vocab_size: int = 30522
    start_token: Optional[int] = None   # If None, it's read off the datamodule's Tokenizer object
    end_token: Optional[int] = None

    # Whether to divide the loss by the number of tokens in each sequence. This is pretty much always done
    # for vanilla language models, but often not done for text VAEs.
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
        datamodule = self.trainer.datamodule if stage == 'fit' else self.datamodule
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

    # If reduce_batch == False, then this method will not reduce across any dimensions other than sequence length
    def get_nll(self, logits: Tensor, labels: Tensor, token_counts: Tensor = None, word_counts: Tensor = None,
                reduce_batch: bool = True, stage: str = 'train'):
        nll = F.cross_entropy(input=logits.flatten(end_dim=-2), target=labels.flatten(), ignore_index=0, reduction='none')
        nll_sum = nll.view(*logits.shape[:-1]).sum(dim=-1)  # Add batch dim(s) back; sum across sequence length

        # Divide by the number of non-padding tokens
        if self.hparams.divide_loss_by_length:
            token_counts = labels.ne(0).sum(dim=-1) if token_counts is None else token_counts
            nll = nll_sum / token_counts
        else:
            nll = nll_sum

        if reduce_batch:
            # Perplexity is normalized using the number of words, not the number of tokens, so that it's comparable
            # across different subword vocabularies
            log_prefix = stage + '_'
            if word_counts is not None:
                per_word_nll = (nll_sum / word_counts).mean()
                ppl = 2 ** (per_word_nll / math.log(2))
                self.log(log_prefix + 'ppl', ppl)

            nll = nll.mean()
            self.log(log_prefix + 'nll', nll)

        return nll

    # These implementations are used by LSTMLanguageModel and TransformerLanguageModel, but are overriden by others
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        logits = self.forward(batch)  # Remove initial [CLS] token from the labels; autoregressive by default
        loss = self.get_nll(logits, batch['token_ids'][..., 1:], word_counts=batch['num_words'])
        return loss

    def on_after_backward(self):
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index)

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
