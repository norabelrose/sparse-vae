import math
import torch
import pytorch_lightning as pl
from dataclasses import dataclass
from omegaconf import OmegaConf
from text_vae import MutualInformation
from torch import Tensor
from typing import *
from .LSTMDecoder import LSTMDecoder
from .LSTMEncoder import LSTMEncoder


@dataclass
class LSTMAutoencoderHparams:
    enc_nh: int = 1024  # Dimensionality of the encoder's LSTM hidden state
    dec_nh: int = 1024  # Dimensionality of the decoder's LSTM hidden state
    dec_dropout_in: float = 0.5
    dec_dropout_out: float = 0.5
    ni: int = 512  # Dimensionality of the input embedding vectors
    nz: int = 32  # Dimensionality of the latent variable vector

    vocab_size: int = 30522
    cls_id: int = 101
    sep_id: int = 102

    batch_size: int = 10  # This is just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 5.0
    max_steps: int = 1000
    momentum: float = 0.9


class LSTMAutoencoder(pl.LightningModule):
    """VAE with normal prior"""
    def __init__(self, hparams: OmegaConf):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = LSTMEncoder(hparams)
        self.decoder = LSTMDecoder(hparams)
        self.save_hyperparameters(hparams)

        self.nz = hparams.nz

        # Makes sure the distribution gets moved to the right devices
        self.register_buffer('prior_mu', torch.zeros(self.nz))
        self.register_buffer('prior_sigma', torch.ones(self.nz))

        self.mutual_info = MutualInformation()

    # Workaround for the fact that Distribution objects don't have a .to() method
    def get_prior(self):
        return torch.distributions.Normal(self.prior_mu, self.prior_sigma)

    def configure_optimizers(self):
        sgd = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=self.hparams.momentum)
        schedule = torch.optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=0.1)
        return [sgd], [schedule]

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, log_mi: bool = False) -> Optional[Tensor]:
        # (batch_size, nz)
        posterior = self.encoder(batch['token_ids'])
        z = posterior.rsample([1]).movedim(source=1, destination=0)
        kl = torch.distributions.kl_divergence(self.get_prior(), posterior).mean()

        reconstruct_err = self.decoder.reconstruct_error(batch['token_ids'], z).mean()
        self.log('kl', kl)
        self.log('nll', reconstruct_err)

        kl_weight = batch.get('kl_weight', 1.0)
        loss = reconstruct_err + kl_weight * kl
        self.log('train_loss', loss)

        if log_mi:
            self.mutual_info(posterior, z)
            self.log('mutual_info', self.mutual_info, on_epoch=True)

        if loss.isnan():
            self.print("Skipping NaN loss at step ", batch_index)
            return None
        else:
            return loss

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, log_mi=True)

    def decode(self, z, strategy, k=5):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            k: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, k)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def reconstruct(self, x, decoding_strategy="greedy", k=5):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            k: the beam width parameter (if applicable)

        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, k)

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
            z, distribution = self.encoder.sample(x, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x, z)
            log_infer_ll = distribution.log_prob(z)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = torch.cat(tmp, dim=-1).logsumexp(dim=-1) - math.log(nsamples)

        return -ll_iw

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.get_prior().log_prob(zrange).sum(dim=-1)

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
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)
        """

        return self.decoder.log_probability(x, z)

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

    def sample_from_prior(self, nsamples):
        """sampling from prior distribution

        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.get_prior().sample((nsamples,))

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z

    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.hparams.mh_burn_in + nsamples * self.hparams.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(mean=cur,
                                std=cur.new_full(size=cur.size(), fill_value=self.hparams.mh_std))
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()

            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if iter_ >= self.hparams.mh_burn_in and (iter_ - self.hparams.mh_burn_in) % self.hparams.mh_thin == 0:
                samples.append(cur.unsqueeze(1))

        return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]

        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]

        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        under distribution q(z|x)

        Args:
            x: [batch_size, *]. The sampled data to estimate mutual info
        """
        mi = 0
        num_examples = 0
        for batch_data in x:
            batch_size = batch_data.size(0)
            num_examples += batch_size
            mutual_info = self.encoder.calc_mi(batch_data)
            mi += mutual_info * batch_size

        return mi / num_examples
