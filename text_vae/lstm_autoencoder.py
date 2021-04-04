import torch
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from .core import (
    ContinuousVAE, ConditionalGaussian, ContinuousVAEHparams
)
from .lstm_language_model import LSTMLanguageModel, LSTMLanguageModelHparams
from .train_callbacks import AggressiveEncoderTraining, KLAnnealing
from torch import nn, Tensor
from torch.distributions import Normal
from typing import *
import math


@dataclass
class LSTMAutoencoderHparams(ContinuousVAEHparams, LSTMLanguageModelHparams):
    latent_depth: int = 32
    bidirectional_encoder: bool = False
    decoder_input_dropout: float = 0.1      # Should make decoder pay more attention to the latents
    decoder_output_dropout: float = 0.1     # Sort of like label smoothing?
    tie_embedding_weights: bool = False     # Tie the decoder's embedding weights to the encoder's embedding weights


class LSTMAutoencoder(LSTMLanguageModel, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_embedding = nn.Embedding(hparams.vocab_size, hparams.d_embedding)
        if hparams.tie_embedding_weights:
            self.encoder_embedding.weight = self.decoder_embedding.weight

        self.encoder = nn.LSTM(
            input_size=hparams.d_embedding,
            hidden_size=hparams.d_model,
            bidirectional=hparams.bidirectional_encoder,
            num_layers=hparams.num_layers,
            batch_first=True
        )
        self.automatic_optimization = hparams.train_mc_samples == 0

        num_directions = 2 if hparams.bidirectional_encoder else 1
        hidden_size = hparams.d_model * num_directions

        # This is the posterior distribution when we're using the traditional VAE training objective
        # (i.e. mc_samples == 0), and the proposal distribution when using DReG (i.e. mc_samples > 0)
        self.q_of_z_given_x = ConditionalGaussian(hidden_size, hparams.latent_depth, bias=False)
        self.z_to_hidden = nn.Linear(hparams.latent_depth, hidden_size)

        self.dropout_in = nn.Dropout(hparams.decoder_input_dropout)
        self.dropout_out = nn.Dropout(hparams.decoder_output_dropout)

        self.initialize_weights()

    def initialize_weights(self):
        scale = self.hparams.init_scale
        for param in self.parameters():
            param.data.uniform_(-scale, scale)

        embedding_scale = scale * 10.0
        self.encoder_embedding.weight.data.uniform_(-embedding_scale, embedding_scale)
        self.decoder_embedding.weight.data.uniform_(-embedding_scale, embedding_scale)

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [AggressiveEncoderTraining(), KLAnnealing()]

    def decoder_input_size(self) -> int:
        return self.hparams.d_embedding + self.hparams.latent_depth     # We concatenate z with the input embeddings

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
        _, (last_state, _) = self.encoder(x)
        last_state = last_state[-1]     # TODO: Make this work for directional LSTMs

        if self.hparams.train_mc_samples > 0:
            q_of_z = self.q_of_z_given_x(last_state, get_kl=False)
            nll = -self.dreg_backward_pass(q_of_z, x, original)

            optimizer = self.optimizers()
            optimizer.step(); optimizer.zero_grad()

        # Single-sample VAE objective
        else:
            q_of_z, kl = self.q_of_z_given_x(last_state, get_kl=True)
            z = q_of_z.rsample()

            kl = kl.sum(dim=-1)
            kl = kl / batch['num_tokens'] if self.hparams.divide_loss_by_length else kl
            kl = kl.mean()

            logits = self.reconstruct(x, z)
            nll, ppl = self.stats_from_logits(logits, batch['token_ids'][..., 1:], word_counts=batch['num_words'])

            log_prefix = stage + '_'
            self.log_dict({
                log_prefix + 'kl': kl,
                log_prefix + 'nll': nll,
                log_prefix + 'ppl': ppl,
                log_prefix + 'mutual_info': self.estimate_mutual_info(q_of_z, z)
            })
            loss = (nll + self.hparams.kl_weight * kl)

            return {'logits': logits, 'loss': loss}

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    # x should be a batch of sequences of token embeddings and z a batch of single latent vectors; both
    # tensors may have a leading num_samples dimension if we're using a multi-sample Monte Carlo objective.
    def reconstruct(self, x, z):
        x = x[..., :-1, :]  # Remove [SEP]
        batch_size, seq_len, _ = x.shape[-3:]

        # (num_samples?), batch_size, seq_len, d_model
        x = self.dropout_in(x)

        # Expand z across the sequence length and then concatenate it to each token embedding
        z_long = z.unsqueeze(-2).expand(*x.shape[:-1], self.hparams.latent_depth)
        x = torch.cat([x, z_long], dim=-1)

        # Merge the minibatch and the MC sample dimensions if needed; nn.LSTM doesn't support multiple batch dims
        z = z.flatten(end_dim=-2)
        c_init = self.z_to_hidden(z).unsqueeze(0)
        h_init = c_init.tanh()
        output, _ = self.decoder(x.flatten(end_dim=-3), (h_init, c_init))

        output = output.view(*x.shape[:-1], output.shape[-1])  # Add the MC sample dim again if needed
        output = self.dropout_out(output)
        return self.output_layer(output)

    # Analytical formula for the joint log probability density of all z_i under a standard unit variance Gaussian.
    # This is a highly optimized version; equivalently we could have put the -0.5 and -log(sqrt(2pi)) inside the sum.
    @staticmethod
    def prior_log_prob(z: Tensor):
        return -0.5 * z.pow(2.0).sum(dim=-1) - math.log(math.sqrt(2 * math.pi)) * z.shape[-1]

    def p_of_x_given_z(self, x, z, labels):
        logits = self.reconstruct(x, z)
        return self.stats_from_logits(logits, labels, reduce_batch=False)

    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        z = torch.randn(batch_size, self.hparams.latent_depth, device=self.device)
        return super().sample(max_length, batch_size, initial_state=self.z_to_hidden(z).unsqueeze(0), context=z)

    def context_depth(self) -> int:
        return self.hparams.latent_depth

    @classmethod
    def learned_initial_state(cls) -> bool:
        return False
