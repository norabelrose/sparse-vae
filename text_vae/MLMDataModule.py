from .TextDataModule import *


@dataclass
class MLMDataModuleHparams(TextDataModuleHparams):
    max_mlm_span_length: int = 1
    whole_word_masking: bool = True

    include_unmasked_tokens_in_labels: bool = True
    mask_token_prob: float = 0.3
    random_token_prob: float = 0.015
    use_smart_random_tokens: bool = False  # Sample random tokens proportional to their frequency in the dataset
    yield_segment_pairs: bool = False


class MLMDataModule(TextDataModule):
    def prepare_data(self, *args, **kwargs):
        super().prepare_data()

        if self.hparams.random_token_prob > 0.0 and self.hparams.use_smart_random_tokens:
            print("Computing token frequencies...")

            token_freqs = self.token_freqs
            vocab_size = token_freqs.numel()

            def get_word_frequencies(batch: Dict[str, list]):
                nonlocal token_freqs
                for sample in batch['text']:
                    token_freqs += torch.tensor(sample).bincount(minlength=vocab_size)

            self.dataset.map(get_word_frequencies, batched=True)
            self.token_freqs = self.token_freqs.float()  # So we can sample from it as a distribution

    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if self.hparams.yield_segment_pairs:
            tokens = [x['text'] for x in inputs]

            # I don't actually understand why we would ever get an odd number of samples here, but around 23,000
            # batches this starts happening and we crash if we don't include this check
            if len(tokens) % 2 != 0:
                tokens.pop()

            pairs = []
            token_types = []
            for i in range(0, len(tokens), 2):
                seqA, seqB = tokens[i], tokens[i + 1][1:]  # Omit the [CLS] token at the beginning of the second sample
                pairs += [torch.cat([seqA, seqB])]
                seg_ids = torch.cat([
                    torch.zeros_like(seqA, dtype=torch.float16),
                    torch.ones_like(seqB, dtype=torch.float16)
                ])
                seg_ids[0] = 2.0  # Special ID for the [CLS] token

                token_types += [seg_ids]

            seg_ids = torch.nn.utils.rnn.pad_sequence(token_types, batch_first=True, padding_value=1.0)
            tokens = torch.nn.utils.rnn.pad_sequence(pairs, batch_first=True)
            padding_mask = tokens.eq(0).float()

            batch = {'token_ids': tokens, 'padding_mask': padding_mask, 'segment_ids': seg_ids}
        else:
            batch = super().collate(inputs)
            tokens, padding_mask = batch['token_ids'], batch['padding_mask']

        mask_prob = self.hparams.mask_token_prob
        random_prob = self.hparams.random_token_prob
        noise_prob = mask_prob + random_prob
        assert noise_prob > 0.0, "Why are you using MLMDataModule if you don't want to corrupt the input?"

        labels = tokens.clone()
        vocab = self.tokenizer.get_vocab()

        if self.hparams.whole_word_masking:
            word_counts = torch.tensor([x['num_words'] for x in inputs])
            word_ids = torch.nn.utils.rnn.pad_sequence([x['word_ids'] for x in inputs], batch_first=True,
                                                       padding_value=-1.0)

            noise_mask = whole_word_mask_(tokens, word_counts, word_ids, mask_prob, vocab['[MASK]'])
        else:
            # We sample a few tokens in each sequence for MLM training (with 15% probability)
            probability_matrix = torch.full(labels.shape, noise_prob)
            special_tokens_mask = tokens.lt(self.special_token_threshold)  # Token IDs under 5 are special tokens

            probability_matrix.masked_fill_(special_tokens_mask | padding_mask.bool(), value=0.0)
            noise_mask = torch.bernoulli(probability_matrix).bool()

            # 80% of the time, we replace masked input tokens with [MASK]
            mask_ratio = mask_prob / noise_prob
            if mask_ratio > 0.0:
                replaced_mask = torch.bernoulli(torch.full(labels.shape, mask_ratio)).bool() & noise_mask
                tokens[replaced_mask] = vocab['[MASK]']
                noise_mask &= ~replaced_mask

            # 20% of the time, we replace masked input tokens with random word
            if random_prob > 0.0:
                if self.hparams.use_smart_random_tokens:
                    random_words = self.token_freqs.multinomial(num_samples=noise_mask.sum())
                else:
                    random_words = torch.randint(low=self.special_token_threshold, high=self.hparams.vocab_size,
                                                 size=[noise_mask.sum()], device=noise_mask.device)
                tokens.masked_scatter_(noise_mask, random_words)

        if not self.hparams.include_unmasked_tokens_in_labels:
            labels[~noise_mask] = -100  # We only compute loss on noised tokens

        batch['labels'] = labels
        batch['token_ids'] = tokens
        return batch

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if not self.hparams.yield_segment_pairs:
            return super().train_dataloader(*args, **kwargs)

        # If we're yielding pairs of segments then we need to get twice as many samples from the dataset as our
        # batch size. Then we dynamically create the pairs in collate()
        return DataLoader(
            self.dataset['train'], batch_size=self.batch_size * 2, shuffle=True,
            collate_fn=self.collate, num_workers=min(20, cpu_count()), pin_memory=True
        )
