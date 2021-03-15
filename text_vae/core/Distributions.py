from torch import nn, Tensor
from torch.distributions import Normal
import torch


class ConditionalGaussian(nn.Module):
    def __init__(self, in_features: int, out_features: int, reduce_dim: int = None, zero_initialized: bool = False):
        super(ConditionalGaussian, self).__init__()

        linear = nn.Linear(in_features, out_features * 2)
        if zero_initialized:
            linear.bias.data.zero_()
            linear.weight.data.zero_()

        self.linear = nn.Sequential(nn.GELU(), linear)
        self.reduce_dim = reduce_dim

    def forward(self, x: Tensor, temperature: float = 1.0) -> Normal:
        mu, logsigma = self.linear(x).chunk(2, dim=-1)
        if self.reduce_dim is not None:
            mu = mu.mean(dim=self.reduce_dim)
            logsigma = logsigma.mean(dim=self.reduce_dim)

        return Normal(loc=mu, scale=logsigma.exp() * temperature)


class AutoregressiveGaussian(nn.Module):
    def __init__(self, num_features: int, stepwise_noise: bool = True):
        super(AutoregressiveGaussian, self).__init__()

        inner_depth = num_features * 2
        self.num_features = num_features
        self.noise_mlp = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features)
        )
        self.hidden_linear = nn.Linear(num_features, inner_depth)
        self.output_linear = nn.Linear(inner_depth, num_features)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=inner_depth, batch_first=True)

        if stepwise_noise:
            self.next_z_dist = ConditionalGaussian(num_features, num_features, zero_initialized=False)
        else:
            self.next_z_dist = None

    def forward(self, batch: int, seq_len: int):
        results = []

        # Sample the initial noise vector
        noise = torch.randn(batch, 1, self.num_features, device=self.output_linear.weight.device)
        inputs = self.noise_mlp(noise)

        hidden = self.hidden_linear(inputs).movedim(1, 0)  # [num_layers, batch, embedding]
        cell = hidden.tanh()

        for i in range(seq_len):
            outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
            outputs = self.output_linear(outputs)

            if self.next_z_dist:
                # For each z_t, we sample from a Gaussian distribution conditioned on all previouz z's
                outputs = self.next_z_dist(outputs).rsample()

            results.append(outputs.squeeze(-2))
            inputs = outputs

        return torch.stack(results, dim=1)
