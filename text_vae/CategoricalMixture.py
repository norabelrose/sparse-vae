from torch import nn, Tensor
from torch.distributions import Categorical, MixtureSameFamily
import torch
import torch.nn.functional as F


# Input should be a tensor of shape [...length, in_features]. The first vector along the length dimension
# (assumed to be a [CLS] token or something similar) is used to determine the weight assigned to each
# mixture component and is thus not included within each component distribution.
class ConditionalCategoricalMixture(nn.Module):
    def __init__(self, num_mixtures: int, num_features: int, num_classes: int):
        super(ConditionalCategoricalMixture, self).__init__()

        self.num_mixtures = num_mixtures
        self.in_features = num_features
        self.num_classes = num_classes

        # Outputs one logit for each mixture component, per group
        self.selector = nn.Conv1d(num_features, num_mixtures, kernel_size=num_mixtures, stride=num_mixtures)

        self.component_weights = nn.Parameter(torch.randn(num_mixtures, num_features, num_features) * 0.02)  # noqa
        self.component_biases = nn.Parameter(torch.zeros(num_mixtures, 1, num_features))

        # Normalize the outputs of all mixture components to have the same variance and mean
        self.layer_norm = nn.LayerNorm(num_features)

        # The shared logit layer
        self.embedding = nn.Parameter(torch.randn(num_classes, num_features) * 0.02)  # noqa

        # Each mixture component gets to add its own bias term to the logit for each class- this way
        # components can "specialize" in different types of classes
        self.output_biases = nn.Parameter(torch.zeros(num_mixtures, 1, num_classes))

    def forward(self, x: Tensor) -> 'CategoricalMixture':
        mixture_logits = self.selector(x.movedim(-1, -2)).movedim(-2, -1)

        # Group the sequence into segments num_mixtures long
        x = x.unflatten(-2, (self.num_mixtures, x.shape[-2] // self.num_mixtures))  # noqa
        x = x.unsqueeze(-3)  # Add num mixtures dimension right after the group dim
        x = x @ self.component_weights + self.component_biases
        x = F.gelu(x)

        x = self.layer_norm(x)
        x = x @ self.embedding.t() + self.output_biases  # Logits of shape [...group, mixtures, group len, classes]

        # Mixture distribution shape: [batch, group, group length (expanded), num_mixtures]
        # Component distribution shape: [batch, group, group length, num_mixtures, num_classes]
        component_dist = Categorical(logits=x.movedim(-3, -2))

        mixture_logits = mixture_logits.unsqueeze(-3)
        mixture_shape = list(mixture_logits.shape)
        mixture_shape[-3] = x.shape[-4]
        mixture_logits = mixture_logits.expand(*mixture_shape)

        mixture_dist = Categorical(logits=mixture_logits)
        return CategoricalMixture(mixture_dist, component_dist)


# noinspection PyAbstractClass
class CategoricalMixture(MixtureSameFamily):
    # Automatically reshape the tensor for convenience
    def log_prob(self, x):
        categorical = self.component_distribution
        assert isinstance(categorical, Categorical)

        group_size = categorical.logits.shape[-4]
        x = x.unflatten(-1, (group_size, x.shape[-1] // group_size))  # noqa

        return super().log_prob(x).flatten(-2, -1)

    # Gather the logits for the highest probability mixture components
    @property
    def logits(self):
        logits = self.component_distribution.logits
        modes = self.mixture_distribution.logits.argmax(dim=-1, keepdim=True).unsqueeze(-1)
        mode_shape = list(modes.shape)
        mode_shape[-1] = logits.shape[-1]
        modes = modes.expand(*mode_shape)

        return logits.gather(dim=-2, index=modes).squeeze(-2).flatten(1, 2)
