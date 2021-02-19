from dataclasses import dataclass, field
from math import ceil
from torch import nn, FloatTensor, LongTensor
from tqdm import tqdm
from typing import *
import torch
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    soft_codes: FloatTensor

    code_indices: Optional[LongTensor] = None
    hard_codes: Optional[FloatTensor] = None

    # Both these are zero when not training or when the Quantizer is called with quantize=False
    commitment_loss: FloatTensor = field(default_factory=lambda: torch.tensor(0.0))
    embedding_loss: FloatTensor = field(default_factory=lambda: torch.tensor(0.0))


# Essentially a fancier version of nn.Embedding which can combine multiple embedding matrices into one object. This is
# useful for hierarchical VQ-VAEs which have multiple latent scales. This class also provides a nice method for updating
# the codebook with k-means.
class Quantizer(nn.Module):
    def __init__(self, num_codes: int, code_depth: int, d_model: int = None, num_levels: int = 1):
        super(Quantizer, self).__init__()
        self.codebook = nn.Parameter(torch.randn(num_levels, num_codes, code_depth))

        if d_model is not None:
            self.downsamplers = nn.ModuleList([
                nn.Linear(d_model, code_depth)
                for _ in range(num_levels)
            ])
            self.upsamplers = nn.ModuleList([
                nn.Linear(code_depth, d_model)
                for _ in range(num_levels)
            ])
        else:
            self.downsamplers = None
            self.upsamplers = None

    def forward(self, x: FloatTensor, level: int = 0, quantize: bool = True) -> QuantizerOutput:
        if self.downsamplers:
            x = self.downsamplers[level](x)

        # We set quantize = False during the first epoch of training- we still want to use the downsampling and
        # upsampling linear layers but we don't want to actually quantize just yet
        output = QuantizerOutput(soft_codes=x)
        if quantize:
            output.code_indices = torch.cdist(x, self.codebook[level]).argmin(dim=-1)
            output.hard_codes = F.embedding(output.code_indices, self.codebook[level])

            if self.training:
                output.commitment_loss = F.mse_loss(output.hard_codes.detach(), x)
                output.embedding_loss = F.mse_loss(output.hard_codes, x.detach())

                # Copy the gradients from the soft codes to the hard codes
                output.hard_codes = output.soft_codes + (output.hard_codes - output.soft_codes).detach()

        return output

    def lookup_codes(self, x: LongTensor, level: int = 0) -> FloatTensor:
        book = self.codebook[level]
        return F.embedding(x, book)

    # Prepare the codes for the decoder
    def upsample_codes(self, x: FloatTensor, level: int = 0) -> FloatTensor:
        return self.upsamplers[level](x)

    @torch.no_grad()
    def perform_kmeans_update(self, soft_codes: Union[FloatTensor, List[FloatTensor]], num_restarts: int = 3):
        best_codebooks = self.codebook.data
        cur_codebooks = torch.empty_like(best_codebooks)
        num_levels, num_codes, code_depth = best_codebooks.shape

        # Make sure this is always a list so that the code below can be generic
        if not isinstance(soft_codes, list):
            soft_codes = [soft_codes]

        soft_codes = [x.flatten(end_dim=-2) for x in soft_codes]
        codes_per_cluster = [x.shape[0] / num_codes for x in soft_codes]
        min_codes_per_cluster = min(codes_per_cluster)
        max_codes_per_cluster = max(codes_per_cluster)

        assert min_codes_per_cluster > 1.0, "Not enough soft codes to perform K means codebook update"
        assert len(soft_codes) == num_levels, "Number of soft codes doesn't match number of levels in the codebook"

        best_losses = [float('inf')] * self.codebook.size(0)
        max_iter = min(100, int(ceil(max_codes_per_cluster)))
        pbar = tqdm(total=max_iter * num_restarts, postfix=dict(best_loss=float('inf')))

        for i in range(num_restarts):
            pbar.desc = f'Computing centroids (restart {i + 1} of {num_restarts})'

            # Initialize centroids with actually observed values
            for level, x in enumerate(soft_codes):
                cur_codebooks[level] = x[num_codes * i:num_codes * (i + 1)]

            for _ in range(max_iter):
                for level, level_soft_codes in enumerate(soft_codes):
                    cur_codebook = cur_codebooks[level]
                    hard_code_indices = torch.cdist(level_soft_codes, cur_codebook).argmin(dim=-1)

                    # Sum together all the soft codes assigned to a cluster, then divide by the number in that cluster
                    counts = hard_code_indices.bincount(minlength=num_codes)
                    cur_codebook.zero_()
                    cur_codebook.scatter_add_(
                        dim=0,
                        index=hard_code_indices[:, None].repeat(1, code_depth),
                        src=level_soft_codes
                    )
                    cur_codebook /= counts.unsqueeze(-1)

                pbar.update()

            for level, x in enumerate(soft_codes):
                cur_codebook = cur_codebooks[level]
                cur_loss = torch.cdist(x, cur_codebook).min(dim=-1).values.pow(2.0).mean() / code_depth  # MSE

                if cur_loss < best_losses[level]:
                    best_codebooks[level] = cur_codebook
                    best_losses[level] = cur_loss

            pbar.set_postfix(best_loss=sum(best_losses) / len(best_losses))

        pbar.close()
