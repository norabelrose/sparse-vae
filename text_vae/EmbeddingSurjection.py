from math import ceil, log2, log
from survae.distributions import ConditionalMeanStdNormal
from survae.transforms import Surjection, UniformDequantization
from .KNNLookupTable import *
import torch


class EmbeddingSurjection(Surjection):
    def __init__(self, vocab_size: int, out_features: int, embedding_scale_factor: int = 2,
                 use_knn_inverse: bool = False):
        super(EmbeddingSurjection, self).__init__()

        # Using a full d_model * 2 embedding matrix can easily blow up our parameter count, so we
        # use a smaller d_model-deep embedding and them upsample it
        embedding_dim = (out_features - 1) // embedding_scale_factor
        self.embedding_scale_factor = embedding_scale_factor

        self.dequantizer = UniformDequantization(num_bits=int(ceil(log2(vocab_size))))
        embedding = nn.Embedding(vocab_size, embedding_dim)

        embedding_dist = ConditionalMeanStdNormal(net=embedding, scale_shape=[embedding_dim])
        embedding_dist.log_scale.data.fill_(-log(vocab_size))
        self.noised_embedding = embedding_dist
        self.knn_lookup_table = KNNLookupTable(embedding) if use_knn_inverse else None

    # Returns the tensor along with the log Jacobian determinant
    def forward(self, x: Tensor) -> tuple:
        components = []
        ldj = 0.0

        for _ in range(self.embedding_scale_factor):
            dequantized, ldj_dq = self.dequantizer(x)
            embedded, log_prob_emb = self.noised_embedding.sample_with_log_prob(context=x)

            components += [dequantized.unsqueeze(-1), embedded]
            ldj += ldj_dq - log_prob_emb

        return torch.cat(components, dim=-1), 0.0

    def inverse(self, z) -> Tensor:
        if self.knn_lookup_table:
            chunk_size = z.shape[-1] // self.embedding_scale_factor
            dist = Normal(loc=z[..., 1:chunk_size], scale=self.noised_embedding.log_scale.exp())
            return self.knn_lookup_table.most_likely_ids(dist)

        return self.dequantizer.inverse(z[..., 0])

    @property
    def stochastic_forward(self):
        return True

    @property
    def stochastic_inverse(self):
        return False
