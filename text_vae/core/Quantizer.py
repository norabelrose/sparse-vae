from dataclasses import dataclass
from math import ceil
from torch import nn, FloatTensor, LongTensor, Tensor
from tqdm import tqdm
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


# Essentially a fancier version of nn.Embedding which can combine multiple embedding matrices into one object. This is
# useful for hierarchical VQ-VAEs which have multiple latent scales. This class also provides a nice method for updating
# the codebook with k-means.
class Quantizer(nn.Module):
    def __init__(self, num_codes: int, code_depth: int, d_model: int, num_levels: int = 1,
                 ema_decay: float = 0.99, reverse_ema: bool = False, gumbel: bool = False):
        super(Quantizer, self).__init__()

        # Initialize the codes to have exactly unit norm- we similarly normalize the encoder outputs
        self.codebook = nn.Parameter(torch.randn(num_levels, num_codes, code_depth))
        self.codebook.data /= self.codebook.data.norm(dim=-1, keepdim=True)
        self.codebook.data[:, 0] = 0.0  # Code at index 0 is always used for padding and set to the zero vector
        self.register_buffer('used_code_mask', torch.zeros(num_levels, num_codes, dtype=torch.bool))

        self.ema_decay = ema_decay
        self.reverse_ema = reverse_ema
        self.gumbel = gumbel
        if gumbel:
            self.logit_layer = nn.Linear(d_model, num_codes)
            self.downsamplers = None
            self.upsamplers = None
        else:
            self.downsamplers = nn.ModuleList([
                nn.Linear(d_model, code_depth)
                for _ in range(num_levels)
            ])
            if ema_decay > 0.0:
                self.register_buffer('cluster_sizes', torch.zeros(num_levels, num_codes, dtype=torch.float))
                self.register_buffer('cluster_sums', torch.zeros_like(self.codebook.data))

        self.upsamplers = nn.ModuleList([
            nn.Linear(code_depth, d_model)
            for _ in range(num_levels)
        ])
        self.entropies = [0.0] * num_levels

    @property
    def num_codes(self) -> int:
        return self.codebook.shape[1]

    @property
    def num_levels(self) -> int:
        return self.codebook.shape[0]

    def forward(self, x: FloatTensor, level: int = 0, quantize: bool = True, mask: Tensor = None) -> QuantizationInfo:
        if self.gumbel:
            logits = self.logit_layer(x)

            gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / 0.9  # ~Gumbel(logits,tau)
            soft_one_hot = gumbels.softmax(dim=-1)
            sorta_hard_codes = soft_one_hot @ self.codebook[level]

            prior = torch.tensor([1.0 / self.num_codes], device=x.device).log()
            posterior = F.log_softmax(logits, dim=-1)

            output = QuantizationInfo(soft_codes=x, hard_codes=sorta_hard_codes)
            output.commitment_loss = F.kl_div(prior, posterior, reduction='batchmean', log_target=True)
            return output

        if self.downsamplers:
            x = self.downsamplers[level](x)

            # There seems to be a tendency for the norm of the encoder's output to increase early in training.
            # Training can easily diverge at this point with the straight-through gradient estimator getting very biased.
            # We correct for this by normalizing the encoder outputs to have unit norm.
            x = x / x.norm(dim=-1, keepdim=True)

        output = QuantizationInfo(soft_codes=x, hard_codes=x)

        # Note that the padding code at index 0 is excluded here- it is initialized to the zero vector and never changed
        distances = torch.cdist(x.detach(), self.codebook[level, 1:])
        output.code_indices = distances.argmin(dim=-1) + 1  # Offset by 1 to account for padding code
        if mask is not None:
            output.code_indices = torch.where(mask, 0, output.code_indices)

        output.hard_codes = self.lookup_codes(output.code_indices, level=level)

        if self.training:
            output.commitment_loss = F.mse_loss(output.hard_codes.detach(), x)
            onehot = F.one_hot(output.code_indices, num_classes=self.num_codes).type_as(x)

            # We need to use 1e-7 and not any smaller stabilization constant here since otherwise it will underflow
            # with 16 bit floats
            probs = onehot.mean(dim=0)
            entropy = (probs * torch.log(probs + 1e-7)).sum().neg()
            self.entropies[level] = entropy.detach()

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
                self.cluster_sizes[level] *= decay
                self.cluster_sizes[level] += (1 - decay) * sizes.detach()
                self.cluster_sums[level] *= decay
                self.cluster_sums[level] += (1 - decay) * sums.detach()

                if not self.reverse_ema:
                    output.embedding_loss = unused_code_loss

                    used_cluster_sizes = self.cluster_sizes[level][used_code_mask]
                    used_cluster_sums = self.cluster_sums[level][used_code_mask]

                    # Laplace smoothing of cluster sizes
                    n = self.cluster_sizes[level].sum()
                    smoothed_sizes = (used_cluster_sizes + 1e-5) * n / (n + self.num_codes * 1e-5)

                    centroids = used_cluster_sums / smoothed_sizes.unsqueeze(-1)
                    self.codebook.data[level][used_code_mask] = centroids.detach()
                    self.codebook.data[level][0] = 0.0  # Padding code stays the zero vector
                else:
                    self.codebook.data[level] = self.cluster_sums[level] / self.cluster_sizes[level].unsqueeze(-1)
            else:
                used_code_loss = F.mse_loss(output.hard_codes, x.detach())
                hybrid_loss = used_code_loss + unused_code_loss

                output.embedding_loss = hybrid_loss

        # Keep track of the codes we've used
        self.used_code_mask[level].index_fill_(0, output.code_indices.flatten().detach(), True)

        if not quantize:
            output.hard_codes = x

        elif self.training:
            # Copy the gradients from the soft codes to the hard codes
            output.hard_codes = output.soft_codes + (output.hard_codes - output.soft_codes).detach()

        return output

    def lookup_codes(self, x: LongTensor, level: int = 0) -> Tensor:
        return F.embedding(x, self.codebook[level], padding_idx=0)

    # Prepare the codes for the decoder
    def upsample_codes(self, x: FloatTensor, level: int = 0) -> FloatTensor:
        return self.upsamplers[level](x)

    # Used for logging to TensorBoard, mainly
    def used_code_info_dict(self):
        if self.num_levels == 1:
            return {'num_used_codes': self.num_used_codes()}

        info_dict = {}
        for level in range(self.num_levels):
            info_dict[f'code_entropy_{level}'] = self.entropies[level]
            info_dict[f'num_used_codes_{level}'] = self.num_used_codes(level)

        return info_dict

    def num_used_codes(self, level: int = 0):
        return self.used_code_mask[level].sum()

    def reset_code_usage_info(self):
        self.used_code_mask.zero_()

    @torch.no_grad()
    def perform_kmeans_update(self, soft_codes: Union[Tensor, Sequence[Tensor]], num_restarts: int = 3):
        best_codebooks = self.codebook.data

        # We do the whole K means algorithm in half precision because we don't need to be super accurate and this
        # can be a very memory-intensive operation
        cur_codebooks = torch.empty_like(best_codebooks, dtype=torch.float16)
        num_levels, num_codes, code_depth = best_codebooks.shape

        # Make sure this is always a list so that the code below can be generic
        if not isinstance(soft_codes, Sequence):
            soft_codes = [soft_codes]

        # soft_codes = [x.flatten(end_dim=-2) for x in soft_codes]  # noqa
        codes_per_cluster = [x.shape[0] / num_codes for x in soft_codes]
        min_codes_per_cluster = min(codes_per_cluster)
        max_codes_per_cluster = max(codes_per_cluster)

        assert min_codes_per_cluster > 1.0, "Not enough soft codes to perform K means codebook update"
        assert len(soft_codes) == num_levels, "Number of soft codes doesn't match number of levels in the codebook"

        max_iter = min(100, int(ceil(max_codes_per_cluster)))
        pbar = tqdm(total=max_iter * num_restarts * self.num_levels, postfix=dict(best_loss=float('inf')))

        for level, level_soft_codes in enumerate(soft_codes):
            best_loss = float('inf')
            cur_loss = float('inf')

            for i in range(num_restarts):
                pbar.desc = f'Fitting codebook {level} (restart {i + 1} of {num_restarts})'
                prev_loss = float('inf')

                # Initialize centroids with actually observed values
                cur_codebook = cur_codebooks[level]
                uniform = cur_codebook.new_ones([1]).expand(level_soft_codes.shape[0])
                rand_indices = torch.multinomial(uniform, num_samples=num_codes)
                cur_codebook.copy_(level_soft_codes[rand_indices])

                for j in range(max_iter):
                    distances, centroid_indices = torch.cdist(level_soft_codes, cur_codebook).min(dim=-1)
                    if j > 0:
                        prev_loss = cur_loss

                    cur_loss = distances.pow(2.0).mean().div(code_depth).item()
                    if cur_loss < best_loss:
                        best_loss = cur_loss
                        pbar.set_postfix(best_loss=best_loss)

                    # Stop early if the loss has effectively stopped going down
                    if prev_loss - cur_loss < 1e-4:
                        pbar.update(max_iter - j)
                        break

                    cur_codebook.zero_()

                    # Sum together the soft codes assigned to a cluster, then divide by the number in that cluster
                    counts = centroid_indices.bincount(minlength=num_codes)

                    # Check for centroids to which zero points were assigned
                    lonely_centroid_mask = counts.eq(0)
                    counts[lonely_centroid_mask] = 1  # Avoid NaNs

                    cur_codebook.scatter_add_(
                        dim=0,
                        index=centroid_indices[:, None].repeat(1, code_depth),
                        src=level_soft_codes
                    )
                    cur_codebook /= counts.unsqueeze(-1)

                    # Replace the lonely centroids with the top K points most distant from their assigned centroids
                    num_lonely_centroids = lonely_centroid_mask.sum()
                    if num_lonely_centroids:
                        furthest_indices = distances.topk(num_lonely_centroids).indices
                        cur_codebook.masked_scatter_(
                            mask=lonely_centroid_mask[:, None],
                            source=level_soft_codes[furthest_indices]
                        )

                    pbar.update()

                if cur_loss < best_loss:
                    best_codebooks[level] = cur_codebook.type_as(best_codebooks)

        # Reset exponential moving average statistics
        if self.ema_decay > 0.0:
            self.cluster_sizes.zero_()
            self.cluster_sums.zero_()

        pbar.close()
