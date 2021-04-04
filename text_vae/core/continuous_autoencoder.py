from contextlib import contextmanager
from torch.distributions import Normal
from .language_model import *
from ..train_callbacks import ReconstructionSampler


# Abstract base classes for autoencoders with continuous latent spaces
@dataclass
class ContinuousVAEHparams(LanguageModelHparams, ABC):
    latent_depth: int = 32  # Dimensionality of the latent variable vector(s)
    kl_weight: float = 1.0  # Ignored when using multi-sample Monte Carlo objectives
    train_mc_samples: int = 0  # Number of Monte Carlo samples used when training. If 0, use single-sample VAE estimator

    # Weight placed placed on the mutual information term in the augmented maximum likelihood objective from the
    # "Mutual Information Constraints for Monte-Carlo Objectives" DeepMind paper (Melis et al. 2020). Note that
    # this must be a value between 0 and 1 since the log p(x) term is scaled by (1 - mutual info weight).
    mutual_info_weight: float = 0.0
    renyi_alpha: float = 1.0

    early_stopping_metric: str = 'val_loss'  # For continuous VAEs we should monitor the whole loss, not just the NLL

class ContinuousVAE(LanguageModel, ABC):
    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [ReconstructionSampler()]

    def setup(self, stage: str):
        super().setup(stage)
        self.decoder_frozen = False

    # Performs a backward pass for both the encoder and decoder networks, using the DReG gradient estimator for the
    # encoder parameters. Returns the IWAE importance-weighted estimate of log p(x).
    # See Tucker et al. 2018 (https://arxiv.org/pdf/1810.04152.pdf) for derivation.
    def dreg_backward_pass(self, proposal_dist: Normal, x: Tensor, labels: Tensor) -> Tensor:
        mc_samples = self.hparams.train_mc_samples
        mutual_info_lambda = self.hparams.mutual_info_weight
        renyi_alpha = self.hparams.renyi_alpha

        # Sample K latent vectors from the encoder's proposal distribution q(z|x)
        z = proposal_dist.rsample([mc_samples])     # [sample, batch, latent_depth]
        log_p_of_z = self.prior_log_prob(z)         # [sample, batch], log probability of the z_i's under the prior

        # The log w_i's (that is, log [p(x|z_i)/q(z_i|x)]) depend on the encoder parameters (phi) in two ways:
        # indirectly through their effect on the z_i's and thereby p(x|z_i), and directly through their effect on
        # the probability q(z_i|x). DReG approximates the sum of these effects, the total derivative, using a
        # weighted sum of the *partial* derivatives of the log w_i's w.r.t. to phi. Here we detach q(z|x) so
        # that autograd doesn't try to double-count the indirect effect of phi on the log w_i's, which we're
        # manually approximating using the weights.
        detached_q = Normal(loc=proposal_dist.loc.detach(), scale=proposal_dist.scale.detach())
        log_q_of_z = detached_q.log_prob(z).sum(dim=-1)  # log q(z_i|x) for each sample z_i; [sample, batch]

        # We call this twice w/ the Rényi objective
        def perform_backward(log_w_value, normalized_w_values, loss_weight = 1.0, retain_graph = False):
            # See https://github.com/google-research/google-research/blob/master/dreg_estimators/model.py
            encoder_loss = -((normalized_w_values.detach() ** 2) * log_w_value).sum(dim=0).mean()
            with self.freeze_decoder():
                self.manual_backward(encoder_loss * loss_weight, retain_graph=True)

            iwae = (log_w_value.logsumexp(dim=0) - math.log(mc_samples)).mean()
            decoder_loss = -iwae
            self.manual_backward(decoder_loss * loss_weight, retain_graph=retain_graph)
            return iwae

        log_cond = self.p_of_x_given_z(x, z, labels)    # log p(x|z_i) for all z_i
        log_joint = log_p_of_z + log_cond               # log p(x, z_i) for all z_i
        log_w = log_joint - log_q_of_z                  # log w_i = log [p(x|z_i)/q(z_i|x)]
        normalized_w = log_w.softmax(dim=0)

        s_hat_weight = 1.0  # The weight placed on the standard DReG loss terms
        if mutual_info_lambda > 0.0:
            # Rényi objective
            if renyi_alpha != 1.0:
                # Equivalent to replacing p(x|z_i) w/ p(x|z_i)^alpha above
                alpha_log_w = log_p_of_z + (renyi_alpha * log_cond) - log_q_of_z
                s_hat_weight = mutual_info_lambda / (renyi_alpha - 1)
                perform_backward(alpha_log_w, alpha_log_w.softmax(dim=0),
                                 loss_weight=renyi_alpha * s_hat_weight - 1, retain_graph=True)
            # KL objective
            else:
                s_hat_weight = 1 - mutual_info_lambda
                u_hat = (normalized_w * log_cond).sum(dim=0).mean()
                self.manual_backward(u_hat * mutual_info_lambda, retain_graph=True)

        return perform_backward(log_w, normalized_w, s_hat_weight)

    # Uses importance weighted multi-sample Monte Carlo to get a tighter estimate of log p(x).
    # Intended for evaluation purposes. Run the encoder first to get q(z|x). Use num_iter to
    # break apart computation into N sequential steps to save memory when num_samples is large.
    def estimate_log_prob_iw(self, q_of_z: Normal, x: Tensor, labels: Tensor, num_samples: int, num_iter: int = 1):
        assert num_samples % num_iter == 0
        chunk_size = num_samples // num_iter

        # Sample K latent vectors from the encoder's proposal distribution q(z|x)
        latents = q_of_z.rsample([num_iter, chunk_size])  # [chunk, sample, batch, latent_depth]
        log_p_of_z = self.prior_log_prob(latents)         # [chunk, sample, batch], log prob of z_i's under the prior
        log_q_of_z = q_of_z.log_prob(latents).sum(dim=-1)   # log q(z_i|x) for each sample z_i
        log_ws = []

        for z, log_p, log_q in zip(latents, log_p_of_z, log_q_of_z):
            log_joint = log_p + self.p_of_x_given_z(x, z, labels)
            log_ws += [log_joint - log_q]

        log_ws = torch.cat(log_ws)  # [chunks * samples, batch]
        return log_ws.logsumexp(dim=0) - math.log(num_samples)

    @staticmethod
    def estimate_mutual_info(conditional_q: Normal, z: Tensor):
        unsqueezed = Normal(loc=conditional_q.loc[:, None], scale=conditional_q.scale[:, None])
        cross_densities = unsqueezed.log_prob(z[None]).sum(dim=-1)     # [batch, batch]

        # Approximate q(z) by averaging over the densities assigned to each z by the other q(z|x)s in the minibatch
        marginal_q = cross_densities.logsumexp(dim=0) - math.log(cross_densities.shape[0])
        return -conditional_q.entropy().sum(dim=-1).mean() - marginal_q.mean()

    # Should return p(x|z)
    # @abstractmethod
    def p_of_x_given_z(self, x, z, labels) -> Tensor:
        raise NotImplementedError

    # Called by AggressiveEncoderTraining callback
    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder_frozen = not requires_grad
        for param in self.decoder_params():
            param.requires_grad = requires_grad

    # @abstractmethod
    def decoder_params(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def encoder_params(self) -> Iterable[nn.Parameter]:
        dec_params = set(self.decoder_params())
        return filter(lambda x: x not in dec_params, self.parameters())

    @contextmanager
    def freeze_decoder(self):
        was_frozen = self.decoder_frozen
        self.decoder_requires_grad_(False)

        yield

        self.decoder_requires_grad_(not was_frozen)
