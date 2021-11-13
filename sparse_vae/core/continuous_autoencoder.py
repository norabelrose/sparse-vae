from abc import abstractmethod
from sparse_vae.core.conditional_gaussian import ConditionalGaussian
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.distributions import Normal
from .language_model import *
import math


# Abstract base classes for autoencoders with continuous latent spaces
@dataclass
class ContinuousVAEHparams(LanguageModelHparams, ABC):
    latent_depth: int = 64     # Dimensionality of the latent variable vector(s)

    # Ignored when using multi-sample Monte Carlo objectives
    kl_annealing_steps: int = 0
    kl_weight_start: float = 1.0
    kl_weight_end: float = 1.0
    kl_weight: float = 1.0

    early_stopping_metric: str = 'val_loss'  # For continuous VAEs we should monitor the whole loss, not just the NLL

class ContinuousVAE(LanguageModel, ABC):
    q_of_z_given_x: ConditionalGaussian
    
    def on_train_start(self):
        self.hparams.kl_weight = self.hparams.kl_weight_start

    def on_after_backward(self):
        super().on_after_backward()

        cur_step = self.global_step
        max_steps = self.hparams.kl_annealing_steps
        kl_end = self.hparams.kl_weight_end
        if not max_steps or self.hparams.kl_weight >= kl_end:
            return

        progress = cur_step / max_steps
        total_distance = kl_end - self.hparams.kl_weight_start
        self.hparams.kl_weight = self.hparams.kl_weight_start + total_distance * progress

    # Returns the latent tensor and the KL
    def sample_z(self, encoder_out: Tensor, token_counts: Tensor, stage: str = 'train'):
        q_of_z, kl = self.q_of_z_given_x(encoder_out, get_kl=True)
        z = q_of_z.rsample()

        raw_kl = kl.flatten(1).sum(dim=-1)  # Sum across everything but the batch dimension
        kl = raw_kl.div(token_counts).mean()

        log_prefix = stage + '_'
        self.log(log_prefix + 'kl', raw_kl.mean())
        
        return z, kl, q_of_z

    # Analytical formula for the joint log probability density of all z_i under a standard unit variance Gaussian.
    @staticmethod
    def prior_log_prob(z: Tensor):
        return -0.5 * z.pow(2.0).sum(dim=-1) - math.log(math.sqrt(2 * math.pi)) * z.shape[-1]

    # Uses importance weighted multi-sample Monte Carlo to get a tighter estimate of log p(x).
    # Intended for evaluation purposes. Run the encoder first to get q(z|x). Use num_iter to
    # break apart computation into N sequential steps to save memory when num_samples is large.
    def estimate_log_prob_iw(self, q_of_z: Normal, x: Tensor, labels: Tensor, num_samples: int, num_iter: int = 1):
        assert num_samples % num_iter == 0
        chunk_size = num_samples // num_iter
        x = x.unsqueeze(0)
        x.padding = x._padding.unsqueeze(0)
        log_ws = []

        # Sample K latent vectors from the encoder's proposal distribution q(z|x)
        for _ in range(num_iter):
            z = q_of_z.rsample([chunk_size])  # [sample, batch, latent_depth]
            log_p_of_z = self.prior_log_prob(z)  # [sample, batch], log prob of z_i's under the prior
            log_q_of_z = q_of_z.log_prob(z).sum(dim=-1)  # log q(z_i|x) for each sample z_i
            log_p_of_x_given_z = self.p_of_x_given_z(
                x.expand(chunk_size, *x.shape[1:]), z, labels.expand(chunk_size, *labels.shape)[..., 1:]
            )

            log_ws += [log_p_of_z + log_p_of_x_given_z - log_q_of_z]

        return torch.cat(log_ws).logsumexp(dim=0) - math.log(num_samples)

    # Returns log p(x|z) summed (not averaged!) over the sequence length, for each element in the batch
    def p_of_x_given_z(self, x, z, labels) -> Tensor:
        logits = self.reconstruct(x, z)[..., :-1, :]  # Remove [SEP]
        log_probs = logits.log_softmax(dim=-1)
        log_probs[..., 0] = 0.0     # Ignore padding tokens in the calculation

        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    # Should return logits
    @abstractmethod
    def reconstruct(self, x, z) -> Tensor:
        raise NotImplementedError
