import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


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
                            num_layers=1,
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

    def beam_search_decode(self, z, k=5):
        """beam search decoding, code is based on
        https://github.com/pcyin/pytorch_basic_nmt/blob/master/nmt.py

        the current implementation decodes sentence one by one, further batching would improve the speed

        Args:
            z: (batch_size, nz)
            k: the beam width

        Returns: List1
            List1: the decoded word sentence list
        """

        decoded_batch = []
        batch_size, nz = z.size()
        device = z.device

        # (1, batch_size, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            # Start with the start of the sentence token
            decoder_input = torch.tensor([[self.cls_id]], dtype=torch.long, device=device)
            decoder_hidden = (h_init[:,idx,:].unsqueeze(1), c_init[:,idx,:].unsqueeze(1))

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0., 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < k and t < 100:
                t += 1

                # (len(live), 1)
                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=0)

                # (1, len(live), nh)
                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                # (len(live), 1, ni) --> (len(live), 1, ni+nz)
                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    len(live_hypotheses), 1, nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                # (len(live), 1, vocab_size)
                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses], dtype=torch.float, device=device)
                decoder_output = decoder_output + prev_logp.view(len(live_hypotheses), 1, 1)

                # (len(live) * vocab_size)
                decoder_output = decoder_output.view(-1)

                # (K)
                log_prob, indexes = torch.topk(decoder_output, k - len(completed_hypotheses))

                live_ids = indexes // self.vocab_size
                word_ids = indexes % self.vocab_size

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.sep_id:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == k:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            n = max(completed_hypotheses, key=lambda node: node.logp)
            utterance = [n.wordid.item()]

            # back trace
            while n := n.prevNode:
                utterance.append(n.wordid.item())

            utterance.reverse()
            decoded_batch.append(utterance)

        return decoded_batch

    def greedy_decode(self, z):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        device = z.device
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.cls_id] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        end_symbol = torch.tensor([self.sep_id] * batch_size, dtype=torch.long, device=device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            max_index = torch.argmax(output_logits, dim=1)
            # max_index = torch.multinomial(probs, num_samples=1)

            decoder_input = max_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(max_index[i].item())

            mask = torch.mul((max_index != end_symbol), mask)

        return decoded_batch

    def sample_decode(self, z):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        device = z.device
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.cls_id] * batch_size, dtype=torch.long, device=device).unsqueeze(1)
        end_symbol = torch.tensor([self.sep_id] * batch_size, dtype=torch.long, device=device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            sample_prob = F.softmax(output_logits, dim=1)
            sample_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = sample_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(sample_index[i].item())

            mask = torch.mul((sample_index != end_symbol), mask)

        return decoded_batch


class VarLSTMDecoder(LSTMDecoder):
    """LSTM decoder with variable-length batching"""
    def __init__(self, hparams):
        super(VarLSTMDecoder, self).__init__(hparams)

        self.embed = nn.Embedding(hparams.vocab_size, hparams.ni, padding_idx=0)
        self.loss = nn.CrossEntropyLoss(reduce=False, ignore_index=0)

    def decode(self, input, z):
        """
        Args:
            input: tuple which contains x and sents_len
                    x: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        """

        input, sents_len = input

        # not predicting start symbol
        sents_len = sents_len - 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni).contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        sents_len = sents_len.unsqueeze(1).expand(batch_size, n_sample).contiguous().view(-1)
        packed_embed = pack_padded_sequence(word_embed, sents_len.tolist(), batch_first=True)

        z = z.view(batch_size * n_sample, self.nz)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(packed_embed, (h_init, c_init))
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: tuple which contains x_ and sents_len
                    x_: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        x, sents_len = x

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode((src, sents_len), z)

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
