from dataclasses import dataclass
from datasets import load_dataset
from itertools import chain, islice
from multiprocessing import cpu_count
from omegaconf import OmegaConf
from pathlib import Path
from tokenizers import BertWordPieceTokenizer  # noqa
from torch import Tensor
from torch.utils.data import DataLoader  # noqa
from .Utilities import *
import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import warnings


@dataclass
class AutoencoderDataModuleHparams:
    batch_size: int = 32
    chunking_strategy: str = 'token'  # 'sentence', 'token', or 'none'
    dataset_name: str = 'pg19'
    max_sentences_per_sample: Optional[int] = None
    min_tokens_per_sample: int = 10
    max_tokens_per_sample: int = 512
    split: str = 'train'    # Any string of the format supported by the HuggingFace datasets library
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
        self.hparams = hparams
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits

        vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
        self.tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)

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

        min_tokens = self.hparams.min_tokens_per_sample
        max_tokens = self.hparams.max_tokens_per_sample
        sep_token: int = self.tokenizer.get_vocab()['[SEP]']

        nontext_cols = self.dataset.column_names
        nontext_cols.remove('text')

        # Convert text into WordPiece tokens
        tokenizer = self.tokenizer  # Seems to help the datasets library hash this function and cache the results
        def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
            encodings = [x.ids for x in tokenizer.encode_batch(batch['text'])
                         if min_tokens <= len(x.ids) <= max_tokens]
            return {'text': encodings}

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

            b_sz = self.get_reasonable_preprocessing_batch_size()

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

        self.dataset.rename_column_('text', 'token_ids')

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.dataset.train_test_split(test_size=0.05, shuffle=True)
        self.dataset.set_format('torch')

    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Combine into a single batched and padded tensor
        inputs = torch.nn.utils.rnn.pad_sequence([x['token_ids'] for x in inputs], batch_first=True)
        padding_needed = self.hparams.max_tokens_per_sample - inputs.shape[1]
        if padding_needed > 0:
            inputs = F.pad(inputs, (0, padding_needed))

        return {'token_ids': inputs, 'padding_mask': inputs.eq(0).float()}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate, num_workers=min(20, cpu_count()), pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate,
                          num_workers=min(20, cpu_count()), pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()
