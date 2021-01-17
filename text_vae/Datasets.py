from dataclasses import dataclass, field
from itertools import chain, islice
from omegaconf import OmegaConf
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from typing import *
from .Utilities import *
import datasets
import math
import multiprocessing
import os
import pytorch_lightning as pl
import torch
import warnings


@dataclass
class TextVaeDataModuleHparams:
    batch_size: int = 10
    chunking_strategy: str = 'token'  # 'sentence' or 'token'
    dataset_name: str = 'pg19'
    max_sentences_per_sample: Optional[int] = None
    max_tokens_per_sample: int = 512
    split: str = 'train'    # Any string of the format supported by the HuggingFace datasets library
    dataset_save_dir: str = os.path.join(os.getcwd(), 'text-vae-datasets')


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class TextVaeDataModule(pl.LightningDataModule):
    def __init__(self, hparams: OmegaConf):
        super(TextVaeDataModule, self).__init__()

        # These warnings are spurious and seem to pop up due to a bug in PyTorch which was fixed in PR #47160
        warnings.filterwarnings('ignore', message='The given NumPy array is not writeable')

        self.hparams = hparams
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits

        vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
        self.tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)

        # Make sure dataset save dir exists
        os.makedirs(self.hparams.dataset_save_dir, exist_ok=True)

    # Subclass hook
    def create_dataset(self):
        self.dataset = datasets.load_dataset(self.hparams.dataset_name, split='train')

    def prepare_data(self, *args, **kwargs):
        # Check if we already have the dataset
        processed_path = os.path.join(self.hparams.dataset_save_dir, self.hparams.dataset_name)
        if os.path.exists(processed_path):
            self.dataset = datasets.load_from_disk(str(processed_path))
            return

        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        # Flattens large lists faster than list(itertools.chain.from_iterable(x))
        def fast_flatten(original, delimiter=None):
            # Pre-allocate memory
            total_size = sum(len(x) for x in original)
            if delimiter:
                total_size += len(original) - 1
            output = [0] * total_size

            cur_idx = 0
            for x in original:
                next_idx = cur_idx + len(original)
                output[cur_idx:next_idx] = x
                if delimiter:
                    output[next_idx] = delimiter
                    cur_idx = next_idx + 1
                else:
                    cur_idx = next_idx

            return output

        # Order of magnitude faster than list(iter(list(islice(x, chunk_size)), []))
        def fast_flatten_and_chunk(original, chunk_size, delimiter):
            total_size = sum(len(x) for x in original) + len(original) - 1
            num_chunks = int(math.ceil(total_size / chunk_size))

            # Use our SizedIterator wrapper which implements __length_hint__ to avoid a bunch of copies and reallocs
            chained_iter = chain.from_iterable(original)
            def chunk_getter():
                chunk = list(SizedIterator(islice(chained_iter, chunk_size - 1), chunk_size))
                if chunk:
                    chunk += [delimiter]
                return chunk

            return list(SizedIterator(iter(chunk_getter, []), num_chunks))

        # Accumulates elements from an iterable into a list until a predicate, called on the list, is True
        def accumulate_until(iterable, predicate):
            accumulated_elems = []

            for elem in iterable:
                old_elems = accumulated_elems
                accumulated_elems = old_elems + [elem]

                if predicate(accumulated_elems):
                    accumulated_elems = []
                    yield old_elems

        if self.hparams.chunking_strategy == 'sentence':
            # The NLTK Punkt tokenizer is actually a fancy learned model that needs to be downloaded
            import nltk.data

            punkt_dir = os.getcwd()
            nltk.download('punkt', download_dir=punkt_dir)

            sent_tokenizer = nltk.data.load(os.path.join(punkt_dir, 'tokenizers/punkt/english.pickle'))
            max_sents = self.hparams.max_sentences_per_sample or int(1e9)

            def sentence_tokenize(batch: Dict[str, list]) -> Dict[str, list]:
                sentences = sent_tokenizer.tokenize_sents(batch['text'])
                sentences = fast_flatten(sentences)  # Chain lists of sentences together from different samples
                return {'text', sentences}

            self.dataset = self.dataset.map(sentence_tokenize, batched=True)

        max_tokens = self.hparams.max_tokens_per_sample
        sep_token: int = self.tokenizer.get_vocab()['[SEP]']

        #def chunk_sentences():
# This code is currently causing weird crashes from inside dill, a HuggingFace datasets dependency, ostensibly due to the fact that
# we are using closures inside the function. Needs to be rewritten to get around this dill bug.
#                sent_list_iter = accumulate_until(
#                    # Take off the [SEP] at the end of each sentence- we only want one [SEP] at the end of each batch
#                    map(lambda sentence: sentence[:-1], iter(encoded_sentences)),
#                    # Take the last list of full sentences that isn't too long
#                    lambda sents: len(sents) > max_sents or sum(len(x) for x in sents) > max_tokens
#                )
#                # Flatten the lists of sentences into lists of tokens
#                batch_iter = map(lambda sents: fast_flatten(sents, delimiter=sep_token), sent_list_iter)
#                return {'text': list(batch_iter)}
        
        def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
            encodings = [x.ids for x in self.tokenizer.encode_batch(batch['text'])]
            return {'text': encodings}

        def chunk(batch: Dict[str, list]) -> Dict[str, list]:
            return {'text': fast_flatten_and_chunk(batch['text'], max_tokens, delimiter=sep_token)}

        nontext_cols = self.dataset.column_names
        nontext_cols.remove('text')

        # Tokenize, chunk, and save
        print(f"Tokenizing '{self.hparams.dataset_name}'...")
        self.dataset = self.dataset.map(tokenize, batched=True, remove_columns=nontext_cols)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Silence annoying warnings after we spawn DataLoader processes
        
        print(f"Chunking '{self.hparams.dataset_name}'...")
        self.dataset = self.dataset.map(chunk, batched=True)
        self.dataset.rename_column_('text', 'token_ids')

        print(f"Saving '{self.hparams.dataset_name}'...")
        self.dataset.save_to_disk(os.path.join(self.hparams.dataset_save_dir, self.hparams.dataset_name))

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.dataset.train_test_split(test_size=0.05, shuffle=True)
        self.dataset.set_format('torch')

    @staticmethod
    def collate(inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Combine into a single batched and padded tensor
        inputs = torch.nn.utils.rnn.pad_sequence([x['token_ids'] for x in inputs], batch_first=True)
        return {'token_ids': inputs, 'padding_mask': inputs.eq(0)}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.hparams.batch_size, shuffle=True,
                          collate_fn=self.collate, num_workers=multiprocessing.cpu_count(), pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset['test'], batch_size=self.hparams.batch_size, collate_fn=self.collate,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()
