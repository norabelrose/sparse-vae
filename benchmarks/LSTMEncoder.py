import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Gaussian LSTM Encoder with constant-length batching"""
    def __init__(self, hparams):
        super(LSTMEncoder, self).__init__()
        self.ni = hparams.ni
        self.nh = hparams.enc_nh
        self.nz = hparams.latent_depth

        self.embed = nn.Embedding(hparams.vocab_size, hparams.ni)

        self.lstm = nn.LSTM(input_size=hparams.ni,
                            hidden_size=hparams.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)

        # dimension transformation to z (mean and logvar)
        self.linear = nn.Linear(hparams.enc_nh, 2 * hparams.latent_depth, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.01, 0.01)

        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        distribution = self.forward(input)

        # (batch, nsamples, nz)
        z = distribution.rsample([nsamples])

        return z, distribution

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(x)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)
        return Normal(mean.squeeze(0), logvar.squeeze(0).exp())
