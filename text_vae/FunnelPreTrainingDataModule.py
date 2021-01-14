from .Datasets import *


# noinspection PyAbstractClass
class FunnelPreTrainingDataModule(TextVaeDataModule):
    dataset_name: ClassVar[str] = 'funnel_pretraining'

    def create_dataset(self):
        wikipedia = datasets.load_dataset('wikipedia', '20200501.en', split='train[:5%]')
        bookcorpus = datasets.load_dataset('bookcorpusopen', split='train[:5%]')
        openwebtext = datasets.load_dataset('openwebtext', split='train[:5%]')
        wikipedia.remove_columns_('title')
        bookcorpus.remove_columns_('title')

        combined = datasets.concatenate_datasets([wikipedia, bookcorpus, openwebtext])
        self.dataset = combined.shuffle()   # This is just to make the tqdm ETA not be totally off when tokenizing

    # Create MLM batches for the generator. Adapted from DataCollatorForLanguageModeling from huggingface/transformers
    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Combine into a single batched tensor
        inputs = torch.nn.utils.rnn.pad_sequence([x['token_ids'] for x in inputs], batch_first=True)
        labels = inputs.clone()
        vocab = self.tokenizer.get_vocab()

        # We sample a few tokens in each sequence for MLM training (with 15% probability)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = inputs.lt(1000)    # Token IDs under 1000 are special tokens or unused
        padding_mask = inputs.eq(0)              # Padding token id is 0

        probability_matrix.masked_fill_(special_tokens_mask | padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = vocab['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=1000, high=len(vocab), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return {'token_ids': inputs, 'labels': labels, 'padding_mask': padding_mask}