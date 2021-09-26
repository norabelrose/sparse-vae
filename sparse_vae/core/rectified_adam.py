import torch
from torch.optim import Optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-6, lamb=False):
        assert 0.0 <= lr, "Learning rate must be non-negative"
        assert 0.0 <= eps, "Epsilon must be non-negative"
        assert 0.0 <= betas[0] < 1.0, "Adam beta1 must be between 0.0 and 1.0"
        assert 0.0 <= betas[1] < 1.0, "Adam beta2 must be between 0.0 and 1.0"
        assert 0.0 <= weight_decay, "Weight decay must be non-negative"
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, lamb=lamb)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            step = group.setdefault('step', 1)  # Note we use 1-indexing here
            beta2_t = beta2 ** step
            bias_correction_v = (1 - beta2_t) ** 0.5

            # Try to estimate the variance of the adaptive learning rate and correct for it
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t = rho_inf - 2 * step * beta2_t / (1 - beta2_t)
            if rho_t > 4:
                r_t_numer = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                r_t_denom = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
                r_t = (r_t_numer / r_t_denom) ** 0.5
                lr *= r_t * bias_correction_v

            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                assert not grad.is_sparse, 'RAdam does not support sparse gradients, use SparseAdam instead'

                # Lazy state initialization
                state = self.state[param]
                if not state:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                # Decay the first and second moment running average coefficient
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                def adam_update_(x, step_size):
                    # If the variance of the EMA of the second moment is tractable to compute (rho_t > 4), then
                    # use the adaptive per-parameter learning rates.
                    if rho_t > 4:
                        denom = (exp_avg_sq.sqrt() / bias_correction_v).add_(group['eps'])
                        x.addcdiv_(exp_avg, denom, value=-step_size)

                    # Otherwise, perform a vanilla SGD w/ momentum update.
                    else:
                        x.add_(exp_avg, alpha=-step_size)

                # With LAMB, we compute a separate update tensor r_t that we project to unit norm, scale
                # by the norm of the parameter, and then add to the parameter tensor. With vanilla RAdam,
                # we directly apply the updates to the parameter tensor.
                bias_correction_m = 1 - beta1 ** step
                if group['lamb']:
                    update = param * -group['weight_decay']
                    adam_update_(update, step_size=1.0 / bias_correction_m)

                    # Normalize the update layerwise by L2 norm
                    trust_ratio = param.norm().clamp(min=0.01, max=10.0) / update.norm()
                    param.add_(update, alpha=lr * trust_ratio)
                    state['trust_ratio'] = trust_ratio
                else:
                    param.mul_(1 - lr * group['weight_decay'])
                    adam_update_(param, step_size=lr / bias_correction_m)

            # Don't forget to update the step
            group['step'] += 1

        return loss
