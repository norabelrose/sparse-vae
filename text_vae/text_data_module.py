from datasets import Features, Value, Sequence, load_dataset
from multiprocessing import cpu_count
from omegaconf import DictConfig
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from torch.utils.data import DataLoader  # noqa
from typing import Optional
from .core import PaddedTensor
from .data_utils import *
import os
import numpy as np
import pytorch_lightning as pl
import torch
import warnings


@dataclass
class TextDataModuleHparams:
    # These two options are mutually exclusive
    batch_size: Optional[int] = None
    tokens_per_batch: Optional[int] = 12_500
    uniform_length_batches: bool = True

    chunking_strategy: str = 'none'             # 'sentence', 'token', or 'none'
    dataset_name: str = 'yelp_review_full'
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None
    min_tokens_per_sample: int = 16
    max_tokens_per_sample: int = 512
    split: Optional[str] = None                 # Any string of the format supported by the HuggingFace datasets library
    vocab_size: int = 2 ** 15
    dataset_save_dir: str = os.path.join(os.getcwd(), 'sparse-vae-datasets')


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class TextDataModule(pl.LightningDataModule):
    def __init__(self, hparams: DictConfig):
        super(TextDataModule, self).__init__()

        # These warnings are spurious and seem to pop up due to a bug in PyTorch which was fixed in PR #47160
        warnings.filterwarnings('ignore', message='The given NumPy array is not writeable')
        warnings.filterwarnings('ignore', message='Loading cached dataset')

        # We should make sure that we filter out any samples that are larger than the max number of tokens
        # we can fit in an entire batch- this could happen with some Wikipedia articles or books
        max_per_batch, max_per_sample = hparams.tokens_per_batch, hparams.max_tokens_per_sample
        if max_per_batch and (not max_per_sample or max_per_sample > max_per_batch):
            hparams.max_tokens_per_sample = max_per_batch

        self.batches = {}
        self.batch_size = hparams.batch_size
        self.dataset = None  # HuggingFace Dataset object, possibly with both train and test splits
        self.hparams = hparams
        self.pad_to_multiple_of = 1
        self.special_token_threshold = 5

        self._preproc_batch_size = None
        self._tokenizer = None

        # Make sure dataset save dir exists
        os.makedirs(self.hparams.dataset_save_dir, exist_ok=True)

    # Finds a reasonable (per thread) batch size given the average length of the samples
    @property
    def preproc_batch_size(self):
        if not self._preproc_batch_size:
            prng = np.random.default_rng(seed=7295)  # Make this deterministic

            # If we have a dataset with multiple splits, use the largest split
            if not isinstance(self.dataset, DatasetDict):
                dataset = self.dataset
            else:
                dataset = max(self.dataset.values(), key=lambda x: x.num_rows)

            # Get Monte Carlo estimate of the average sample length without actually iterating through the whole dataset
            indices = prng.choice(len(dataset), 10, replace=False)
            elements = [dataset[int(i)] for i in indices]
            avg_len_estimate = sum(len(x['text']) for x in elements) / len(elements)

            # 1,000 for 1,000 character samples
            self._preproc_batch_size = round(1e6 / avg_len_estimate)

        return self._preproc_batch_size

    @property
    def tokenizer(self) -> ByteLevelBPETokenizer:
        if not self._tokenizer:
            self.setup_tokenizer()

        return self._tokenizer

    # Subclass hook
    def create_dataset(self):
        if path := self.hparams.dataset_path:
            self.dataset = DatasetDict.load_from_disk(path)
            self.hparams.dataset_name = self.dataset['train'].config_name
        else:
            self.dataset = load_dataset(
                self.hparams.dataset_name, name=self.hparams.dataset_config,
                split=self.hparams.split, cache_dir=self.hparams.dataset_save_dir
            )

    def prepare_data(self, *args, **kwargs):
        self.create_dataset()   # Download or load a big file from disk

        features = get_features_all_equal(self.dataset)
        assert (text_column := features.get('text')), "Can't find text colum in dataset"

        # Dataset is already pre-tokenized
        if hasattr(text_column, 'feature') and text_column.feature.dtype.startswith('int'):
            print(f"Computing sample lengths...")
            self.dataset = self.dataset.map(
                lambda batch: {'num_tokens': [len(row) for row in batch['text']]}, batched=True, batch_size=1000
            )

        # We need to tokenize the raw text
        else:
            cols_to_remove = list(features.keys())
            cols_to_remove.remove('text')

            # We're generating single-sentence samples
            chunk_strategy = self.hparams.chunking_strategy
            if chunk_strategy == 'sentence':
                # The NLTK Punkt tokenizer is actually a fancy learned model that needs to be downloaded
                import nltk.data
                nltk_dir = os.path.join(os.getcwd(), 'sparse-vae-pretrained/nltk/')
                punkt_dir = os.path.join(nltk_dir, 'tokenizers/punkt/english.pickle')
                os.makedirs(nltk_dir, exist_ok=True)

                if not os.path.exists(os.path.join(nltk_dir, 'tokenizers/punkt/')):
                    nltk.download('punkt', download_dir=nltk_dir)

                def sentence_split(batch: Dict[str, list]) -> Dict[str, list]:
                    sent_tokenizer = nltk.data.load(punkt_dir, cache=True)
                    sentences = sent_tokenizer.tokenize_sents(batch['text'])
                    return {'text': list(chain.from_iterable(sentences))}

                print(f"Finding sentence boundaries for '{self.hparams.dataset_name}'...")
                self.dataset = self.dataset.map(
                    sentence_split, batched=True, batch_size=self.preproc_batch_size, remove_columns=cols_to_remove
                )
                cols_to_remove.clear()

            print(f"Tokenizing '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(
                tokenize,
                batched=True, batch_size=1000,
                remove_columns=cols_to_remove,
                features=Features({
                    'num_char': Value('int32'),
                    'num_tokens': Value('int32'),
                    'num_words': Value('int32'),
                    'text': Sequence(Value('uint16'))
                }),
                fn_kwargs=dict(
                    chunk=self.hparams.chunking_strategy == 'token',
                    min_tokens=self.hparams.min_tokens_per_sample,
                    max_tokens=self.hparams.max_tokens_per_sample,
                    tokenizer=self.tokenizer
                )
            )

        # Silence annoying warnings from the tokenizers package after we spawn DataLoader processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Ensure that there actually is a train-test split
        if not isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset.train_test_split(test_size=50_000, shuffle=True)  # noqa

        # Annoyingly the wikipedia dataset *is* a DatasetDict but it only has a 'train' split
        elif 'test' not in self.dataset:
            self.dataset = self.dataset['train'].train_test_split(test_size=50_000, shuffle=True)  # noqa

        if self.hparams.uniform_length_batches:
            self.dataset = self.dataset.sort('num_tokens')

        if tokens_per_batch := self.hparams.tokens_per_batch:
            batch_dataset = self.dataset.map(
                compute_uniform_sized_batches,
                batched=True, batch_size=None, input_columns=['num_tokens'],
                remove_columns=get_columns_all_equal(self.dataset),
                with_indices=True,
                fn_kwargs=dict(max_size=tokens_per_batch)
            )
            self.batches = {
                name: list(zip(split['start'], split['length']))
                for name, split in batch_dataset.items()
            }

    def setup(self, stage: Optional[str] = None):
        self.dataset.set_format('numpy')    # We manually call torch.from_numpy to handle uint16 conversion

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        split = kwargs.get('split') or 'train'
        sampler = None
        shuffle = False
        batch_size = 1

        if self.batches:
            sampler = PrebatchedRandomSampler(batches=self.batches[split])
        elif self.hparams.uniform_length_batches:
            sampler = ContiguousRandomSampler(dataset_length=len(self.dataset[split]), batch_size=self.batch_size)
        else:
            shuffle = True
            batch_size = self.batch_size

        return DataLoader(
            self.dataset[split], batch_sampler=sampler, batch_size=batch_size,
            shuffle=shuffle, collate_fn=self.collate, pin_memory=True, num_workers=max(20, cpu_count())
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.train_dataloader(split='test')

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def collate(self, inputs: List[dict]) -> Dict[str, Tensor]:
        # Annoyingly PyTorch doesn't have a uint16 type- check if the vocab size would overflow an
        # int16; if so we copy into an int32 tensor, otherwise we do a no-copy reinterpret cast to int16
        upcast = self.tokenizer.get_vocab_size() > 2 ** 15
        return {
            'num_char': torch.tensor([x['num_char'] for x in inputs]),
            'token_ids': PaddedTensor.from_raw(self.pad_pack([
                torch.from_numpy(x['text'].view(np.int16) if not upcast else x['text'].astype(np.int32))
                for x in inputs
            ])),
            'token_count': torch.tensor([x['num_tokens'] for x in inputs])
        }

    def pad_pack(self, batch: List[Tensor], pad_value: int = 0) -> Tensor:
        buffer_len = max(len(x) for x in batch)

        factor = self.pad_to_multiple_of
        if factor > 1:
            remainder = buffer_len % factor
            buffer_len += factor - remainder if remainder else 0

        buffer = torch.full([len(batch), buffer_len], pad_value, dtype=batch[0].dtype)
        for i, sequence in enumerate(batch):
            buffer[i, :len(sequence)] = sequence

        return buffer

    # Called from prepare_data, as well as from setup when we load from a checkpoint
    def setup_tokenizer(self):
        tokenizers_dir = Path.cwd() / 'sparse-vae-pretrained' / 'tokenizers'
        tokenizers_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = tokenizers_dir / (self.hparams.dataset_name + '.json')

        if vocab_path.exists():
            print(f'Loading pretrained tokenizer from {vocab_path}')
            self._tokenizer = Tokenizer.from_file(str(vocab_path))
            assert self.tokenizer.get_vocab_size() == self.hparams.vocab_size
        else:
            print(f'Training a BPE tokenizer for the dataset {self.hparams.dataset_name}')
            self._tokenizer = ByteLevelBPETokenizer()
            self._tokenizer.post_processor = RobertaProcessing(sep=("[SEP]", 2), cls=("[CLS]", 1))
            batch_size = self.preproc_batch_size

            def text_iterator(dataset):
                data_len = len(dataset)
                for i in range(0, data_len, batch_size):
                    end = i + batch_size
                    if end > data_len:
                        end = data_len

                    yield dataset[i:end]['text']

            # Don't train the vocabulary on the validation dataset just to be safe
            data = self.dataset['train'] if isinstance(self.dataset, DatasetDict) else self.dataset
            self.tokenizer.train_from_iterator(
                text_iterator(data),
                vocab_size=self.hparams.vocab_size,
                special_tokens=["[PAD]", "[CLS]", "[SEP]"]
            )
            self.tokenizer.save(str(vocab_path))

        # This hyperparameter could change every run of the application so we don't want to save
        # the truncation policy with the tokenizer
        if self.hparams.chunking_strategy == 'token':
            self._tokenizer.enable_truncation(self.hparams.max_tokens_per_sample)
