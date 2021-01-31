import math
import torch
from dataclasses import dataclass
from omegaconf import DictConfig
from text_vae import Autoencoder, AutoencoderHparams, MutualInformation
from torch import nn, Tensor
from torch.distributions import Normal
from typing import *
from .LSTMDecoder import LSTMDecoder
from .LSTMLanguageModel import LSTMLanguageModelHparams


@dataclass
class LSTMAutoencoderHparams(AutoencoderHparams, LSTMLanguageModelHparams):
    enc_nh: int = 1024  # Dimensionality of the encoder's LSTM hidden state
    latent_depth: int = 32  # Dimensionality of the latent variable vector


class LSTMAutoencoder(Autoencoder):
    """VAE with normal prior"""
    def __init__(self, hparams: DictConfig):
        super(LSTMAutoencoder, self).__init__(hparams)

        self.encoder_embedding = nn.Embedding(hparams.vocab_size, hparams.ni)
        self.encoder_lstm = nn.LSTM(input_size=hparams.ni, hidden_size=hparams.enc_nh, batch_first=True)
        self.posterior = nn.Linear(hparams.enc_nh, 2 * hparams.latent_depth, bias=False)
        self.reset_parameters()

        self.decoder = LSTMDecoder(hparams)
        self.mutual_info = MutualInformation()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.01, 0.01)

        nn.init.uniform_(self.encoder_embedding.weight, -0.1, 0.1)

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)

    # Get the posterior distribution of the latent variable for an input
    def forward(self, x):
        word_embed = self.encoder_embedding(x)

        _, (last_state, last_cell) = self.encoder_lstm(word_embed)

        mean, logvar = self.posterior(last_state).chunk(2, -1)
        return Normal(mean.squeeze(0), logvar.squeeze(0).exp())

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Optional[Tensor]:
        # (batch_size, nz)
        posterior = self.encoder(batch['token_ids'])
        z = posterior.rsample([1]).movedim(source=1, destination=0)
        kl = torch.distributions.kl_divergence(self.get_base_prior(), posterior).mean()

        reconstruct_err = self.decoder.reconstruct_error(batch['token_ids'], z).mean()
        self.log('kl', kl)
        self.log('nll', reconstruct_err)
        loss = reconstruct_err + self.hparams.kl_weight * kl

        if val:
            self.mutual_info(posterior, z)
            self.log('mutual_info', self.mutual_info, on_epoch=True, on_step=False)
            self.log('val_loss', loss, on_epoch=True, on_step=False)
        else:
            self.log('train_loss', loss)

        if loss.isnan():
            self.print(f"Encountered NaN loss at step {batch_index}. Halting training.")
            raise KeyboardInterrupt
        else:
            return loss

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, val=True)

    def sample(self, max_length: int, count: int = 1, **kwargs):
        latents = self.get_base_prior().sample([count])

    def decode(self, z, strategy, k=5):
        return self.decoder.autoregressive_decode(z, strategy=strategy, k=k)

    def nll_iw(self, x, nsamples, ns=100):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10
        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]
            # param is the parameters required to evaluate q(z|x)
            distribution = self.encoder(x)
            z = distribution.rsample([ns])

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x, z)
            log_infer_ll = distribution.log_prob(z)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = torch.cat(tmp, dim=-1).logsumexp(dim=-1) - math.log(nsamples)

        return -ll_iw

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.get_base_prior().log_prob(z).sum(dim=-1)
        log_gen = self.decoder.log_probability(x, z)

        return log_prior + log_gen

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace

        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        batch_size = x[0].size(0) if isinstance(x, tuple) else x.size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_comp.logsumexp(dim=1, keepdim=True)

        return log_posterior
