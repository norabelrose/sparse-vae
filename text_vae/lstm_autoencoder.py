import torch
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from .core import (
    ContinuousVAE, ConditionalGaussian, ContinuousVAEHparams
)
from .lstm_language_model import LSTMLanguageModel, LSTMLanguageModelHparams
from torch import nn, Tensor
from torch.distributions import Normal
from typing import *
import math
import torch.nn.functional as F


@dataclass
class LSTMAutoencoderHparams(ContinuousVAEHparams, LSTMLanguageModelHparams):
    bidirectional_encoder: bool = False
    decoder_input_dropout: float = 0.5      # Should make decoder pay more attention to the latents
    decoder_output_dropout: float = 0.5     # Sort of like label smoothing?
    tie_embedding_weights: bool = False     # Tie the decoder's embedding weights to the encoder's embedding weights


class LSTMAutoencoder(LSTMLanguageModel, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        vocab_size = self.tokenizer.get_vocab_size()
        self.encoder_embedding = nn.Embedding(vocab_size, hparams.d_model)
        if hparams.tie_embedding_weights:
            self.encoder_embedding.weight = self.decoder_embedding.weight

        self.encoder = nn.LSTM(
            input_size=hparams.d_embedding,
            hidden_size=hparams.d_model,
            bidirectional=hparams.bidirectional_encoder,
            num_layers=hparams.num_layers,
            batch_first=True
        )
        self.automatic_optimization = hparams.mc_samples == 0

        num_directions = 2 if hparams.bidirectional_encoder else 1
        hidden_size = hparams.d_model * num_directions

        # This is the posterior distribution when we're using the traditional VAE training objective
        # (i.e. mc_samples == 0), and the proposal distribution when using DReG (i.e. mc_samples > 0)
        self.q_of_z_given_x = ConditionalGaussian(hidden_size, hparams.latent_depth)
        self.z_to_hidden = nn.Linear(hparams.latent_depth, hidden_size)

        self.dropout_in = nn.Dropout(hparams.decoder_input_dropout)
        self.dropout_out = nn.Dropout(hparams.decoder_output_dropout)

        self.initialize_weights()

    def decoder_input_size(self) -> int:
        return self.hparams.d_embedding + self.hparams.latent_depth     # We concatenate z with the input embeddings

    def initialize_weights(self):
        scale = self.hparams.init_scale
        for param in self.parameters():
            nn.init.uniform_(param, -scale, scale)

        embedding_scale = scale * 10.0
        nn.init.uniform_(self.encoder_embedding.weight, -embedding_scale, embedding_scale)

    def decoder_params(self) -> Iterable[nn.Parameter]:
        return chain(
            self.decoder.parameters(),
            self.decoder_embedding.parameters(),
            self.output_layer.parameters()
        )

    # Get the posterior distribution of the latent variable for an input
    def forward(self, x) -> Normal:
        _, (last_state, last_cell) = self.encoder(x)
        return self.q_of_z_given_x(last_state)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids']

        x = self.encoder_embedding(original)
        _, (last_state, last_cell) = self.encoder(x)

        if self.hparams.train_mc_samples > 0:
            q_of_z = self.q_of_z_given_x(last_state, get_kl=False)
            nll = -self.dreg_backward_pass(q_of_z, x, original)
            ppl = self.ppl_from_nll(nll, batch)

            optimizer = self.optimizers()
            optimizer.step(); optimizer.zero_grad()

        # Single-sample VAE objective
        else:
            q_of_z, kl = self.q_of_z_given_x(last_state, get_kl=True)
            z = q_of_z.rsample()
            kl = kl.sum(dim=-1).mean()
            self.log('kl', kl)

            logits = self.reconstruct(x, z)
            nll, ppl, entropy = self.stats_from_logits(logits, batch, autoregressive=True)

            log_prefix = stage + '_'
            self.log_dict({
                log_prefix + 'nll': nll,
                log_prefix + 'ppl': ppl,
                log_prefix + 'entropy': entropy
            })
            loss = (nll + self.hparams.kl_weight * kl)

            if stage == 'val':
                mi = self.estimate_mutual_info(q_of_z, z)
                self.log('mutual_info', mi, on_epoch=True, on_step=False)

            return {'logits': logits, 'loss': loss}

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def reconstruct(self, x, z):
        # remove end symbol
        src = x[:, :-1]
        batch_size, seq_len, _ = src.size()

        # (batch_size, seq_len, ni)
        word_embed = self.dropout_in(src)
        z_ = z.unsqueeze(1).expand(batch_size, seq_len, self.hparams.latent_depth)

        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size, self.hparams.latent_depth)
        c_init = self.z_to_hidden(z).unsqueeze(0)
        h_init = c_init.tanh()
        output, _ = self.decoder(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        return self.output_layer(output)

    # Analytical formula for the joint log probability density of all z_i under a standard unit variance Gaussian.
    # This is a highly optimized version; equivalently we could have put the -0.5 and -log(sqrt(2pi)) inside the sum.
    @staticmethod
    def prior_log_prob(z: Tensor):
        return -0.5 * z.pow(2.0).sum(dim=-1) - math.log(math.sqrt(2 * math.pi)) * z.shape[-1]

    def log_prob(self, x, z, labels):
        logits = self.reconstruct(x, z)
        log_probs = -F.cross_entropy(logits.flatten(end_dim=-2), labels.flatten(), ignore_index=0, reduction='none')
        log_probs = log_probs.view(*logits[:-2])    # Add extra batch / MC sample dimension(s) if needed
        return log_probs.sum(dim=-1).mean(dim=-1)   # Mean across batch dim, but not MC sample dim if it exists

    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        z = torch.randn(batch_size, self.hparams.latent_depth, device=self.device)
        return super().sample(max_length, batch_size, initial_state=z, context=z)
