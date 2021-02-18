from dataclasses import dataclass
from datasets import load_dataset
from itertools import chain, islice
from math import ceil
from multiprocessing import cpu_count
from omegaconf import OmegaConf
from pathlib import Path
from tokenizers import BertWordPieceTokenizer, Tokenizer  # noqa
from torch import Tensor
from torch.utils.data import DataLoader  # noqa
from text_vae.core.Utilities import *
import os
import numpy as np
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
import warnings


@dataclass
class AutoencoderDataModuleHparams:
    batch_size: int = 32
    chunking_strategy: str = 'token'  # 'sentence', 'token', or 'none'
    dataset_name: str = 'pg19'
    masked_lm: bool = False
    max_sentences_per_sample: Optional[int] = None
    min_tokens_per_sample: int = 10
    max_tokens_per_sample: int = 512
    pad_to_max_length: bool = False
    split: str = 'train'    # Any string of the format supported by the HuggingFace datasets library
    uniform_length_batching: bool = True
    use_finetuned_tokenizer: bool = True
    vocab_size: int = 30522
    dataset_save_dir: str = os.path.join(os.getcwd(), 'text-vae-datasets')


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self, hparams: OmegaConf):
        super(AutoencoderDataModule, self).__init__()

        # These warnings are spurious and seem to pop up due to a bug in PyTorch which was fixed in PR #47160
        warnings.filterwarnings('ignore', message='The given NumPy array is not writeable')
        warnings.filterwarnings('ignore', message='Loading cached dataset')

        # This is just for compatibility with pl.Trainer's auto_scale_batch_size feature
        self.batch_size = hparams.batch_size
        self.data_indices = []  # Used when uniform_length_batching is true
        self.hparams = hparams
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits
        self.tokenizer = None

        # Make sure dataset save dir exists
        os.makedirs(self.hparams.dataset_save_dir, exist_ok=True)

    # Finds a reasonable (per thread) batch size given the average length of the samples
    def get_reasonable_preprocessing_batch_size(self, num_samples: int = 10):
        prng = np.random.default_rng(seed=7295)  # Make this deterministic

        # Get Monte Carlo estimate of the average sample length without actually iterating through the whole dataset
        indices = prng.choice(len(self.dataset), num_samples, replace=False)
        elements = [self.dataset[int(i)] for i in indices]
        avg_len_estimate = sum(len(x['text']) for x in elements) / len(elements)

        # 1,000 for 1,000 character samples
        return round(1e6 / avg_len_estimate)

    # Subclass hook
    def create_dataset(self):
        self.dataset = load_dataset(self.hparams.dataset_name, split=self.hparams.split,
                                    cache_dir=self.hparams.dataset_save_dir)

    def prepare_data(self, *args, **kwargs):
        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        b_sz = self.get_reasonable_preprocessing_batch_size()
        if self.hparams.use_finetuned_tokenizer:
            tokenizers_dir = Path.cwd() / 'text-vae-pretrained' / 'tokenizers'
            tokenizers_dir.mkdir(parents=True, exist_ok=True)
            vocab_path = tokenizers_dir / (self.hparams.dataset_name + '.json')

            if vocab_path.exists():
                print(f'Loading pretrained tokenizer from {vocab_path}')
                self.tokenizer = Tokenizer.from_file(str(vocab_path))
                assert self.tokenizer.get_vocab_size() == self.hparams.vocab_size
            else:
                print(f'Training a WordPiece tokenizer for the dataset {self.hparams.dataset_name}')

                self.tokenizer = BertWordPieceTokenizer()
                def text_iterator():
                    data_len = len(self.dataset)
                    for i in range(0, data_len, b_sz):
                        end = i + b_sz
                        if end > data_len:
                            end = data_len

                        yield self.dataset[i:end]['text']

                self.tokenizer.train_from_iterator(text_iterator(), vocab_size=self.hparams.vocab_size)
                self.tokenizer.save(str(vocab_path))
        else:
            vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
            print('Using default pretrained tokenizer')
            self.tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)

        min_tokens = self.hparams.min_tokens_per_sample
        max_tokens = self.hparams.max_tokens_per_sample
        sep_token: int = self.tokenizer.get_vocab()['[SEP]']

        nontext_cols = self.dataset.column_names
        nontext_cols.remove('text')

        # Convert text into WordPiece tokens, while also saving some important stats
        tokenizer = self.tokenizer  # Seems to help the datasets library hash this function and cache the results
        def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
            encodings = [x for x in tokenizer.encode_batch(batch['text'])
                         if min_tokens <= len(x.ids) <= max_tokens]

            # HuggingFace's tokenizers library currently doesn't allow you to directly get the whole word count from
            # an Encoding, but they do expose a word_ids attribute which gives you a list of Optional[int] which is
            # None for special tokens, and which indicates the index of the word to which a token belongs. Since tokens
            # that belong to the same word are always continguous, we just iterate over this list and increment the
            # word count whenever we see a word index that isn't the same as the previous one we saw.
            def word_count(word_ids: List[int]):
                count = 1
                for i in range(1, len(word_ids)):
                    word, prev_word = word_ids[i], word_ids[i - 1]
                    if word is not None and word != prev_word:
                        count += 1

                return count

            return {
                'text': [x.ids for x in encodings],
                'num_tokens': [len(x) for x in encodings],
                'num_words': [word_count(x.word_ids) for x in encodings]
            }

        chunk_strategy = self.hparams.chunking_strategy
        # We're either generating single-sentence samples, or multi-sentence samples that respect sentence boundaries
        if chunk_strategy == 'sentence':
            max_sents = self.hparams.max_sentences_per_sample or int(1e9)

            # Flattens large lists faster than list(itertools.chain.from_iterable(x))
            def fast_flatten(original):
                # Pre-allocate memory
                total_size = sum(len(x) for x in original)
                output = [None] * total_size

                cur_idx = 0
                for x in original:
                    next_idx = cur_idx + len(x)
                    output[cur_idx:next_idx] = x
                    cur_idx = next_idx

                return output

            # The NLTK Punkt tokenizer is actually a fancy learned model that needs to be downloaded
            import nltk.data
            nltk_dir = os.path.join(os.getcwd(), 'text-vae-pretrained/nltk/')
            punkt_dir = os.path.join(nltk_dir, 'tokenizers/punkt/english.pickle')
            os.makedirs(nltk_dir, exist_ok=True)

            if not os.path.exists(os.path.join(nltk_dir, 'tokenizers/punkt/')):
                nltk.download('punkt', download_dir=nltk_dir)

            def sentence_split(batch: Dict[str, list]) -> Dict[str, list]:
                sent_tokenizer = nltk.data.load(punkt_dir, cache=True)
                sentences = sent_tokenizer.tokenize_sents(batch['text'])
                return {'text': fast_flatten(sentences)}  # Chain lists of sentences together from different samples

            print(f"Finding sentence boundaries for '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(sentence_split, batched=True, batch_size=b_sz, remove_columns=nontext_cols)

            print(f"Tokenizing '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(tokenize, batched=True, batch_size=b_sz * max(10, cpu_count()))

            # This is for when we're generating batches of multiple full sentences
            if max_sents > 1:
                def chunk_getter(batch: Dict[str, list]):
                    current_batch = []
                    current_length = 0

                    for sentence in batch['text']:
                        next_length = len(sentence)

                        # This sentence is so long it exceeds the max length by itself, so skip it
                        if next_length > max_tokens:
                            continue

                        # Adding this sentence would make the current batch too long, so yield the current batch,
                        # start a new batch and add this sentence too it
                        elif next_length + current_length > max_tokens or len(current_batch) >= max_sents:
                            chunk_to_yield = fast_flatten(current_batch)
                            current_batch.clear()
                            current_batch.append(sentence)

                            yield chunk_to_yield
                        else:
                            current_batch.append(sentence)

                print(f"Grouping '{self.hparams.dataset_name}' into batches...")
                self.dataset = self.dataset.map(lambda batch: {'text': list(chunk_getter(batch))}, batched=True)

        # We're ignoring sentence boundaries
        else:
            print(f"Tokenizing '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(tokenize, batched=True, remove_columns=nontext_cols)

            # Split large samples into smaller chunks, and group smaller samples together
            if chunk_strategy == 'token':
                def chunk(batch: Dict[str, list]) -> Dict[str, list]:
                    chained_iter = chain.from_iterable(batch['text'])

                    def chunk_getter():
                        while chunk := list(islice(chained_iter, max_tokens - 1)):
                            yield chunk + [sep_token]

                    return {'text': list(chunk_getter())}

                print(f"Grouping '{self.hparams.dataset_name}' into batches...")
                self.dataset = self.dataset.map(chunk, batched=True)
            elif chunk_strategy != 'none':
                raise ValueError(f"Invalid chunk_strategy '{chunk_strategy}'")

        # Silence annoying warnings from the tokenizers package after we spawn DataLoader processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.dataset = self.dataset.train_test_split(test_size=0.025, shuffle=True)
        if self.hparams.uniform_length_batching:
            print("Sorting samples by length...")
            self.dataset = self.dataset.sort('num_tokens')

        self.dataset.rename_column_('text', 'token_ids')

    def setup(self, stage: Optional[str] = None):
        # Keep track of the indices of samples we haven't yielded yet
        if self.hparams.uniform_length_batching:
            self.data_indices = list(range(len(self.dataset['train'])))

        self.dataset.set_format('torch')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.hparams.uniform_length_batching:
            sampler = SequentialRandomSampler(dataset_length=len(self.dataset['train']), batch_size=self.batch_size)
            shuffle = False
            batch_size = 1
        else:
            sampler = None
            shuffle = True
            batch_size = self.batch_size

        return DataLoader(
            self.dataset['train'], batch_sampler=sampler, batch_size=batch_size, shuffle=shuffle,
            collate_fn=self.collate, num_workers=min(20, cpu_count()), pin_memory=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate,
            num_workers=min(20, cpu_count()), pin_memory=True
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Combine into a single batched and padded tensor
        tokens = torch.nn.utils.rnn.pad_sequence([x['token_ids'] for x in inputs], batch_first=True)
        if self.hparams.pad_to_max_length:
            padding_needed = self.hparams.max_tokens_per_sample - tokens.shape[1]
            if padding_needed > 0:
                inputs = F.pad(tokens, (0, padding_needed))

        padding_mask = tokens.eq(0).float()
        word_counts = torch.stack([x['num_words'] for x in inputs])
        if not self.hparams.masked_lm:
            return {'token_ids': tokens, 'padding_mask': padding_mask, 'word_count': word_counts}

        # Mask tokens for MLM. Adapted from DataCollatorForLanguageModeling from huggingface/transformers
        labels = inputs.clone()
        vocab = self.tokenizer.get_vocab()

        # We sample a few tokens in each sequence for MLM training (with 15% probability)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = inputs.lt(1000)  # Token IDs under 1000 are special tokens or unused

        probability_matrix.masked_fill_(special_tokens_mask | padding_mask.bool(), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = vocab['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=1000, high=len(vocab), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return {'token_ids': inputs, 'labels': labels, 'padding_mask': padding_mask, 'word_count': word_counts}

# Used to return random batches that are of roughly uniform length
@dataclass
class SequentialRandomSampler:
    dataset_length: int
    batch_size: int

    def __post_init__(self):
        self.batch_starts = list(range(0, self.dataset_length, self.batch_size))

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.dataset_length / self.batch_size))

    def __next__(self):
        if not self.batch_starts:
            self.batch_starts = list(range(0, self.dataset_length, self.batch_size))  # Prepare for next epoch
            raise StopIteration

        index_idx = random.randrange(0, len(self.batch_starts))
        start_idx = self.batch_starts[index_idx]
        end = start_idx + self.batch_size
        if end > self.dataset_length:
            end = self.dataset_length

        del self.batch_starts[index_idx]  # Don't yield the same indices twice
        return list(range(start_idx, end))
