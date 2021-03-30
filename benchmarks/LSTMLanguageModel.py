from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn, Tensor
from typing import *
import torch
from text_vae import LanguageModel, LanguageModelHparams
from .LSTMDecode import autoregressive_decode


@dataclass
class LSTMLanguageModelHparams(LanguageModelHparams):
    dec_nh: int = 1024  # Dimensionality of the LSTM hidden state
    dec_dropout_in: float = 0.5
    dec_dropout_out: float = 0.5
    ni: int = 512  # Dimensionality of the input embedding vectors


class LSTMLanguageModel(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(LSTMLanguageModel, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.embedding = nn.Embedding(hparams.vocab_size, hparams.ni)
        self.decoder = nn.LSTM(input_size=hparams.ni, hidden_size=hparams.dec_nh, batch_first=True, num_layers=2)
        self.initial_state = nn.Parameter(torch.randn(2, 1, hparams.dec_nh))

        output_embedding = nn.Linear(hparams.ni, hparams.vocab_size)
        output_embedding.weight = self.embedding.weight
        self.logit_layer = nn.Sequential(
            nn.Linear(hparams.dec_nh, hparams.ni),
            output_embedding
        )

    # Returns [batch, seq_len, vocab_size] tensor of logits
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch['token_ids']
        batch_size = x.shape[0]
        c0 = self.initial_state.repeat(1, batch_size, 1)
        h0 = torch.tanh(c0)

        x = self.embedding(x)
        x, _ = self.decoder(x, (h0, c0))
        return self.logit_layer(x)

    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        return autoregressive_decode(
            rnn=self.decoder,
            z=None,
            embedding=self.embedding,
            logit_callable=self.logit_layer,
            initial_hidden_state=self.initial_state.repeat(1, batch_size, 1),
            max_length=max_length,
            start_symbol=self.start_token,
            end_symbol=self.end_token
        )
