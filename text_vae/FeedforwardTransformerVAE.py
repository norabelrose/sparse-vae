from .core import ContinuousVAE, ContinuousVAEHparams
from omegaconf import DictConfig
from typing import *


class FeedforwardTransformerVAEHparams(ContinuousVAEHparams):
    pass


class FeedforwardTransformerVAE(ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)


