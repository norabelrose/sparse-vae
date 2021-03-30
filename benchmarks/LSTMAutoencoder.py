import torch
from dataclasses import dataclass
from einops import rearrange
from omegaconf import DictConfig
from text_vae import (
    ContinuousVAE, ConditionalGaussian, ContinuousVAEHparams, MutualInformation
)
from torch import nn, Tensor
from torch.distributions import Normal
from typing import *
from .LSTMDecode import autoregressive_decode
from .LSTMLanguageModel import LSTMLanguageModelHparams


@dataclass
class LSTMAutoencoderHparams(ContinuousVAEHparams, LSTMLanguageModelHparams):
    enc_nh: int = 1024  # Dimensionality of the encoder's LSTM hidden state
    latent_depth: int = 32  # Dimensionality of the latent variable vector


class LSTMAutoencoder(ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super(LSTMAutoencoder, self).__init__(hparams)

        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = nn.Embedding(vocab_size, hparams.ni)
        self.encoder = nn.LSTM(input_size=hparams.ni, hidden_size=hparams.enc_nh, batch_first=True)
        self.decoder = nn.LSTM(
            input_size=hparams.ni + hparams.latent_depth,  # concatenate z with input
            hidden_size=hparams.dec_nh,
            batch_first=True
        )
        self.posterior = ConditionalGaussian(hparams.enc_nh, hparams.latent_depth)

        self.dropout_in = nn.Dropout(hparams.dec_dropout_in)
        self.dropout_out = nn.Dropout(hparams.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(hparams.latent_depth, hparams.dec_nh)

        # prediction layer
        output_embedding = nn.Linear(hparams.dec_nh, vocab_size)
        output_embedding.weight = self.embedding.weight
        self.output_layer = nn.Sequential(
            nn.Linear(hparams.dec_nh, hparams.ni),
            nn.GELU(),
            nn.LayerNorm(hparams.ni),
            output_embedding
        )
        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.reset_parameters()
        self.mutual_info = MutualInformation()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.01, 0.01)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)

    # Get the posterior distribution of the latent variable for an input
    def forward(self, x) -> Normal:
        _, (last_state, last_cell) = self.encoder(x)
        return self.posterior(last_state)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False):
        original = batch['token_ids']

        x = self.embedding(original)
        posterior = self.forward(x)
        z = posterior.rsample().movedim(source=1, destination=0)
        kl = torch.distributions.kl_divergence(self.get_base_prior(), posterior).sum(dim=-1).mean()

        logits, reconstruct_err = self.reconstruct(x, z, original[:, 1:])
        reconstruct_err = reconstruct_err.sum(dim=-1).mean()

        self.log('total_kl', kl)
        self.log('total_nll', reconstruct_err)
        loss = (reconstruct_err + self.hparams.kl_weight * kl) / x.shape[1]

        if val:
            self.mutual_info(posterior, z)
            self.log('mutual_info', self.mutual_info, on_epoch=True, on_step=False)
            self.log('val_loss', loss, on_epoch=True, on_step=False)
        else:
            self.log('train_loss', loss)

        if not loss.isfinite():
            return None
        else:
            return {'logits': logits, 'loss': loss}

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.training_step(batch, batch_index, val=True)

    def reconstruct(self, x, z, labels):
        # remove end symbol
        src = x[:, :-1]

        batch_size, seq_len, _ = src.size()

        # (batch_size, seq_len, ni)
        word_embed = self.dropout_in(src)
        z_ = z.unsqueeze(1).expand(batch_size, seq_len, self.hparams.latent_depth)

        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size, self.hparams.latent_depth)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.decoder(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        output_logits = self.output_layer(output)
        loss = self.loss(rearrange(output_logits, 'b l v -> b v l'), labels)

        return output_logits, loss

    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        latents = self.get_base_prior().sample([batch_size])
        return autoregressive_decode(
            rnn=self.decoder,
            z=latents,
            embedding=self.embedding,
            logit_callable=self.output_layer,
            initial_hidden_state=self.trans_linear(latents).unsqueeze(0),
            max_length=max_length,
            start_symbol=self.start_token,
            end_symbol=self.end_token
        )
