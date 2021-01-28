from dataclasses import dataclass
from omegaconf import OmegaConf
from torch import nn, Tensor
from typing import *
import pytorch_lightning as pl
import torch.nn.functional as F


@dataclass
class LSTMLanguageModelHparams:
    dec_nh: int = 1024  # Dimensionality of the LSTM hidden state
    dec_dropout_in: float = 0.5
    dec_dropout_out: float = 0.5
    ni: int = 512  # Dimensionality of the input embedding vectors

    vocab_size: int = 30522
    cls_id: int = 101
    sep_id: int = 102


class LSTMLanguageModel(pl.LightningModule):
    def __init__(self, hparams: OmegaConf):
        super(LSTMLanguageModel, self).__init__()
        self.save_hyperparameters(hparams)

        self.embed = nn.Embedding(hparams.vocab_size, hparams.ni)
        self.decoder = nn.LSTM(input_size=hparams.ni, hidden_size=hparams.dec_nh, batch_first=True)
        self.logit_linear = nn.Linear(in_features=hparams.dec_nh, out_features=hparams.vocab_size)

    # Returns [batch, seq_len, vocab_size] tensor of logits
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch['token_ids']
        x = self.embed(x)
        x, _ = self.decoder(x)
        return self.logit_linear(x)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Tensor:
        logits = self.forward(batch)
        loss = F.cross_entropy(input=logits, target=batch['token_ids'])
        self.log('train_loss' if not val else 'val_loss', loss)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, val=True)
