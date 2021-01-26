import math
import pytorch_lightning as pl
import torch
from abc import ABC
from dataclasses import dataclass
from omegaconf import OmegaConf
from torch.distributions import Normal
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import *


@dataclass
class AutoencoderHparams(ABC):
    latent_depth: int = 16  # Depth of the latent tensors/vectors

    batch_size: int = 0  # This is here just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 150.0
    lr: float = 1e-4
    lr_decay_steps: Optional[int] = 150_000
    warmup_steps: int = 1000
    weight_decay: float = 0.01


class Autoencoder(pl.LightningModule, ABC):
    def __init__(self, hparams: OmegaConf):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters(hparams)

        # Create the standard diagonal Gaussian prior for the first layer
        self.register_buffer('prior_mu', torch.zeros(hparams.latent_depth))
        self.register_buffer('prior_sigma', torch.ones(hparams.latent_depth))

    # Called by AggressiveEncoderTraining callback
    def decoder_requires_grad_(self, requires_grad: bool):
        raise NotImplementedError

    # Called by UnconditionalSampler callback
    def sample(self, max_length: int, count: int = 1, **kwargs):
        raise NotImplementedError

    # Workaround for the fact that Distribution objects don't have a .to() method
    def get_base_prior(self) -> Normal:
        return Normal(self.prior_mu, self.prior_sigma)

    def configure_optimizers(self):
        adam = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        decay_steps = self.hparams.lr_decay_steps
        warmups = self.hparams.warmup_steps

        def cosine_decay_with_warmup(cur_step: int):
            if cur_step < warmups:          # Warmup phase
                return cur_step / warmups
            elif not decay_steps:           # Just fall back to a constant schedule if we don't know the decay steps
                return 1.0
            else:                           # Cosine decay
                progress = (cur_step - warmups) / max(1, decay_steps - warmups)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return [adam], [{
            'scheduler': LambdaLR(adam, cosine_decay_with_warmup),
            'interval': 'step'
        }]
