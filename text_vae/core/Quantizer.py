from dataclasses import dataclass
from torch import nn, FloatTensor, LongTensor, Tensor
from typing import *
import torch
import torch.nn.functional as F


@dataclass
class QuantizationInfo:
    soft_codes: FloatTensor

    code_indices: Optional[LongTensor] = None
    hard_codes: Optional[FloatTensor] = None

    # Both these are zero when not training or when the Quantizer is called with quantize=False
    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0


class Quantizer(nn.Module):
    def __init__(self, num_codes: int, code_depth: int, d_model: int, ema_decay: float = 0.99,
                 reverse_ema: bool = False, gumbel: bool = False):
        super(Quantizer, self).__init__()

        # Initialize the codes to have exactly unit norm- we similarly normalize the encoder outputs
        self.codebook = nn.Parameter(torch.randn(num_codes, code_depth))
        self.codebook.data /= self.codebook.data.norm(dim=-1, keepdim=True)
        self.codebook.data[:, 0] = 0.0  # Code at index 0 is always used for padding and set to the zero vector
        self.register_buffer('used_code_mask', torch.zeros(num_codes, dtype=torch.bool))

        self.ema_decay = ema_decay
        self.reverse_ema = reverse_ema
        self.gumbel = gumbel
        if gumbel:
            self.logit_layer = nn.Linear(d_model, num_codes)
            self.downsampler = None
            self.upsampler = None
        else:
            self.downsampler = nn.Linear(d_model, code_depth)

            if ema_decay > 0.0:
                self.register_buffer('cluster_sizes', torch.zeros(num_codes, dtype=torch.float))
                self.register_buffer('cluster_sums', torch.zeros_like(self.codebook.data))

        self.upsampler = nn.Linear(code_depth, d_model)
        self.entropy = 0.0

    @property
    def num_codes(self) -> int:
        return self.codebook.shape[0]

    def forward(self, x: FloatTensor, quantize: bool = True, mask: Tensor = None) -> QuantizationInfo:
        if self.gumbel:
            logits = self.logit_layer(x)

            gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / 0.9  # ~Gumbel(logits,tau)
            soft_one_hot = gumbels.softmax(dim=-1)
            sorta_hard_codes = soft_one_hot @ self.codebook

            prior = torch.tensor([1.0 / self.num_codes], device=x.device).log()
            posterior = F.log_softmax(logits, dim=-1)

            output = QuantizationInfo(soft_codes=x, hard_codes=sorta_hard_codes)
            output.commitment_loss = F.kl_div(prior, posterior, reduction='batchmean', log_target=True)
            return output

        if self.downsampler:
            x = self.downsampler(x)

            # There seems to be a tendency for the norm of the encoder's output to increase early in training.
            # Training can easily diverge at this point with the straight-through gradient estimator getting very biased.
            # We correct for this by normalizing the encoder outputs to have unit norm.
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        output = QuantizationInfo(soft_codes=x, hard_codes=x)

        # Note that the padding code at index 0 is excluded here- it is initialized to the zero vector and never changed
        distances = torch.cdist(x.detach(), self.codebook[1:])
        output.code_indices = distances.argmin(dim=-1) + 1  # Offset by 1 to account for padding code
        if mask is not None:
            output.code_indices = torch.where(mask, 0, output.code_indices)

        output.hard_codes = self.lookup_codes(output.code_indices)

        if self.training:
            output.commitment_loss = F.mse_loss(output.hard_codes.detach(), x)
            onehot = F.one_hot(output.code_indices, num_classes=self.num_codes).type_as(x)

            probs = onehot.mean(dim=0)
            entropy = (probs * torch.log(probs + 1e-6)).sum().neg()
            self.entropy = entropy.detach()

            flat_onehot = onehot.flatten(end_dim=-2)
            bincounts = flat_onehot.sum(dim=0)
            used_code_mask = bincounts.ne(0)

            # Average Euclidean distance between each code in the codebook and the encoder output nearest to it.
            # Importantly this loss allows for gradient to flow to codes that are never actually selected as the
            # nearest neighbor to a given encoder output, gradually pulling 'dead' codes closer to encoder outputs.
            code_distances = distances.min(dim=-2).values
            unused_code_distances = code_distances[~used_code_mask[1:].unsqueeze(0).expand_as(code_distances)]
            unused_code_loss = unused_code_distances.mean()

            # Apply exponential moving average update
            if (decay := self.ema_decay) > 0.0:
                observations = x.flatten(end_dim=-2)

                if self.reverse_ema:
                    distances = distances.flatten(end_dim=-2).t()  # [codebook size, distances]
                    inv_distances = distances.add(0.5).reciprocal()
                    sizes = inv_distances.sum(dim=-1)
                    sums = inv_distances @ observations
                else:
                    sizes = bincounts
                    sums = (observations.t() @ flat_onehot).t()

                # Update EMA state
                self.cluster_sizes *= decay
                self.cluster_sizes += (1 - decay) * sizes.detach()
                self.cluster_sums *= decay
                self.cluster_sums += (1 - decay) * sums.detach()

                if not self.reverse_ema:
                    output.embedding_loss = unused_code_loss

                    used_cluster_sizes = self.cluster_sizes[used_code_mask]
                    used_cluster_sums = self.cluster_sums[used_code_mask]

                    # Laplace smoothing of cluster sizes
                    n = self.cluster_sizes.sum()
                    smoothed_sizes = (used_cluster_sizes + 1e-5) * n / (n + self.num_codes * 1e-5)

                    centroids = used_cluster_sums / smoothed_sizes.unsqueeze(-1)
                    self.codebook.data[used_code_mask] = centroids.detach()
                    self.codebook.data[0] = 0.0  # Padding code stays the zero vector
                else:
                    self.codebook.data = self.cluster_sums / self.cluster_sizes.unsqueeze(-1)
            else:
                used_code_loss = F.mse_loss(output.hard_codes, x.detach())
                hybrid_loss = used_code_loss + unused_code_loss

                output.embedding_loss = hybrid_loss

        # Keep track of the codes we've used
        self.used_code_mask.index_fill_(0, output.code_indices.flatten().detach(), True)

        if not quantize:
            output.hard_codes = x

        elif self.training:
            # Copy the gradients from the soft codes to the hard codes
            output.hard_codes = output.soft_codes + (output.hard_codes - output.soft_codes).detach()

        return output

    def lookup_codes(self, x: LongTensor) -> Tensor:
        return F.embedding(x, self.codebook, padding_idx=0)

    # Prepare the codes for the decoder
    def upsample_codes(self, x: FloatTensor) -> FloatTensor:
        return self.upsampler(x)

    def num_used_codes(self):
        return self.used_code_mask.sum()

    def reset_code_usage_info(self):
        self.used_code_mask.zero_()
