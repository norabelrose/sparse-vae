from torch import nn, Tensor
from torch.distributions import Normal
from typing import *
from text_vae.core.LanguageModel import LanguageModel
import faiss.contrib.torch_utils  # noqa; this allows us to use PyTorch tensors seemlessly with FAISS
import math
import torch


# Wrapper around faiss library functions for efficiently looking up the K nearest neighbors to a given vector in an
# embedding table. Used by HierarchicalAutoencoder when using the Gaussian output type
class KNNLookupTable:
    def __init__(self, embedding: nn.Embedding, exact: bool = True):
        self.embedding = embedding
        self.faiss_index = None

        self.exact = exact  # If False, we're allowed to use approximate algorithms to speed up search
        self.on_gpu = (embedding.weight.data.device.type == 'cuda')

    def build_index(self):
        raw_embedding = self.embedding.weight.data
        embedding_dim = raw_embedding.shape[-1]

        if self.on_gpu:
            gpu_resource = faiss.StandardGpuResources()
            cur_device = raw_embedding.device.index

            index_config = faiss.GpuIndexFlatConfig()
            index_config.device = cur_device

            if self.exact:
                index = faiss.GpuIndexFlatL2(gpu_resource, embedding_dim, index_config)
            else:
                num_clusters = 16 * round(raw_embedding.shape[0] ** 0.5)
                index = faiss.GpuIndexIVFFlat(gpu_resource, embedding_dim, num_clusters, faiss.METRIC_L2, index_config)

        elif self.exact:
            index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Hierarchical Navigable Small World is recommended for small (i.e. N < 1e6) datasets on the CPU:
            # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
            index = faiss.IndexHNSWFlat(raw_embedding.shape[-1], 64, faiss.METRIC_L2)

        if index.is_trained:
            index.add(raw_embedding)  # noqa
        self.faiss_index = index

    # Trains the underlying index using samples from a given language model
    def fit_to_model(self, model: LanguageModel, sample_seq_len: int, vectors_per_cluster: int = 256, **kwargs):
        assert self.faiss_index and not self.faiss_index.is_trained, "There's no index that needs training"
        assert isinstance(self.faiss_index, faiss.GpuIndexIVFFlat)

        num_clusters = self.faiss_index.getNumLists()
        samples_needed = vectors_per_cluster * num_clusters // sample_seq_len

        output = model.sample(max_length=sample_seq_len, count=samples_needed, **kwargs)
        self.faiss_index.train(output)  # noqa
        self.faiss_index.add(self.embedding.weight.data)  # noqa

    # Given a diagonal Gaussian distribution over the embedding space, find the K embedding vectors with the highest
    # probability density
    def k_most_likely(self, gaussian: Normal, k: int, return_log_probs: bool = False):
        raw_embedding = self.embedding.weight.data

        # Given a diagonal Gaussian where sigma_i is the ith element along the diagonal of the covariance matrix, the
        # top K most likely elements will be those with the smallest *weighted* L2 distance:
        #   sum from i=0 to k of {(x_i - mu_i) ** 2 / sigma_i}; or equivalently
        #   sum from i=0 to k of {(x_i ** 2 - 2 * x_i * mu_i + mu_i ** 2) / sigma_i}; or
        #   sum of {x_i ** 2 / sigma_i} - 2 * sum of {x_i * mu_i / sigma_i} + sum of {mu_i ** 2 / sigma_i}
        # We use the final formulation here in order to save memory.
        sq_embedding = raw_embedding ** 2
        sq_centroids = gaussian.mean ** 2

        var = gaussian.variance
        inv_var = var.reciprocal()

        weighted_embed_norms = (sq_embedding @ inv_var.unsqueeze(-1)).squeeze(-1)  # [..., vocab_size]
        weighted_centroid_norms = sq_centroids.mul(inv_var).sum(dim=-1, keepdim=True)  # [..., 1]

        weighted_centroids = gaussian.mean * inv_var
        weighted_inner_products = (raw_embedding @ weighted_centroids.unsqueeze(-1)).squeeze(-1)  # [..., vocab_size]

        # [batch, seq_len, vocab_size]
        weighted_distances = weighted_embed_norms - 2 * weighted_inner_products + weighted_centroid_norms
        smallest_dists, most_likely_ids = weighted_distances.topk(k, largest=False)

        if return_log_probs:
            num_dims = raw_embedding.shape[-1]
            log_probs = (-num_dims * math.log(2 * math.pi) - var.log().sum() - smallest_dists) / 2
            return most_likely_ids, log_probs

        return most_likely_ids

    # Returns a tuple of Tensors of the form (distances, indices) where distances are the *squared* Euclidean distance.
    def k_nearest(self, x: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        if self.faiss_index:
            assert self.faiss_index.is_trained, "Index has been built, but needs to be trained"

            leading_dims = x.shape[:-1]
            x = x.flatten(end_dim=-2)  # Flatten all dimensions except the embedding dim since faiss expects a matrix

            dists, indices = self.faiss_index.search(x, k)
            return dists.view(*leading_dims, k), indices.view(*leading_dims, k)

        # We haven't built an index yet (probably because we're training a model and we just want to periodically
        # log a sample), so just use PyTorch functions to do this
        raw_embedding = self.embedding.weight.data
        return torch.cdist(x, raw_embedding).topk(k, largest=False)

    def most_likely_ids(self, distribution: Normal) -> Tensor:
        return self.k_most_likely(distribution, 1).squeeze(-1)

    # Convenience methods for when you want just the very nearest vector
    def nearest(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        distances, indices = self.k_nearest(x, 1)
        return distances.squeeze(-1), indices.squeeze(-1)

    def nearest_ids(self,  x: Tensor) -> Tensor:
        _, indices = self.k_nearest(x, 1)
        return indices.squeeze(-1)
