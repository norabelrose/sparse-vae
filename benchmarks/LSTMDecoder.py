import torch
import torch.nn as nn
from text_vae import autoregressive_decode, GenerationStrategy


class LSTMDecoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, hparams):
        super(LSTMDecoder, self).__init__()

        self.ni = hparams.ni
        self.nh = hparams.dec_nh
        self.nz = hparams.latent_depth
        self.vocab_size = hparams.vocab_size
        self.cls_id = hparams.cls_id
        self.sep_id = hparams.sep_id

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(self.vocab_size, hparams.ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(hparams.dec_dropout_in)
        self.dropout_out = nn.Dropout(hparams.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(hparams.latent_depth, hparams.dec_nh, bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=hparams.ni + hparams.latent_depth,
                            hidden_size=hparams.dec_nh,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(hparams.dec_nh, self.vocab_size, bias=False)

        vocab_mask = torch.ones(self.vocab_size)
        # vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduction='none')
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.01, 0.01)

        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)


    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)

    def autoregressive_decode(self, z, strategy: GenerationStrategy, k: int = 5):
        return autoregressive_decode(
            rnn=self.lstm,
            z=z,
            embedding=self.embed,
            initial_hidden_state=self.trans_linear(z),
            logit_callable=self.pred_linear,
            start_symbol=self.cls_id,
            end_symbol=self.sep_id,
            strategy=strategy,
            k=k
        )
