from datasets import load_dataset, DatasetDict
from multiprocessing import cpu_count
from omegaconf import DictConfig
from pathlib import Path
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import DataLoader  # noqa
from .core import PaddedTensor
from .data_utils import *
import os
import numpy as np
import pytorch_lightning as pl
import torch
import warnings


@dataclass
class TextDataModuleHparams:
    batch_size: int = 16

    batching_strategy: str = 'uniform_size'     # 'uniform_size', 'uniform_size_prebatched', 'uniform_length', 'random'
    chunking_strategy: str = 'none'             # 'sentence', 'token', or 'none'
    dataset_name: str = 'yelp_polarity'
    dataset_config: Optional[str] = None
    max_sentences_per_sample: Optional[int] = None
    min_tokens_per_sample: int = 16
    max_tokens_per_sample: int = 512
    pad_to_multiple_of: int = 1
    split: Optional[str] = None                 # Any string of the format supported by the HuggingFace datasets library
    use_finetuned_tokenizer: bool = True
    vocab_size: int = 30522
    dataset_save_dir: str = os.path.join(os.getcwd(), 'text-vae-datasets')


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class TextDataModule(pl.LightningDataModule):
    def __init__(self, hparams: DictConfig):
        super(TextDataModule, self).__init__()

        # These warnings are spurious and seem to pop up due to a bug in PyTorch which was fixed in PR #47160
        warnings.filterwarnings('ignore', message='The given NumPy array is not writeable')
        warnings.filterwarnings('ignore', message='Loading cached dataset')

        # This is just for compatibility with pl.Trainer's auto_scale_batch_size feature
        self.batch_size = hparams.batch_size
        self.hparams = hparams
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits
        self.special_token_threshold = 5 if hparams.use_finetuned_tokenizer else 1000

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
    def tokenizer(self) -> Tokenizer:
        if not self._tokenizer:
            self.setup_tokenizer()

        return self._tokenizer

    # Subclass hook
    def create_dataset(self):
        self.dataset = load_dataset(self.hparams.dataset_name, name=self.hparams.dataset_config,
                                    split=self.hparams.split, cache_dir=self.hparams.dataset_save_dir)

    def prepare_data(self, *args, **kwargs):
        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        min_tokens = self.hparams.min_tokens_per_sample
        max_tokens = self.hparams.max_tokens_per_sample

        if isinstance(self.dataset, DatasetDict):
            cols_by_split = self.dataset.column_names.values()
            columns = next(iter(cols_by_split))

            assert all(cols == columns for cols in cols_by_split), "All splits must have the same columns"
            nontext_cols = columns
        else:
            nontext_cols = self.dataset.column_names
        nontext_cols.remove('text')

        # Weirdly the yelp_polarity dataset uses the sequence '\ n' instead of actual newlines, and
        # '\ " "' instead of simple double quotes, so we fix that here.
        if self.hparams.dataset_name == 'yelp_polarity':
            def remove_backslashes(batch: Dict[str, list]) -> Dict[str, list]:
                return {'text': [text.replace(r'\ n', '\n').replace(r'\ " "', '"') for text in batch['text']]}

            self.dataset = self.dataset.map(remove_backslashes, batched=True, batch_size=self.preproc_batch_size)

        chunk_strategy = self.hparams.chunking_strategy
        # We're either generating single-sentence samples, or multi-sentence samples that respect sentence boundaries
        if chunk_strategy == 'sentence':
            max_sents = self.hparams.max_sentences_per_sample or int(1e9)

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
            self.dataset = self.dataset.map(sentence_split, batched=True, batch_size=self.preproc_batch_size,
                                            remove_columns=nontext_cols)

            print(f"Tokenizing '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(
                tokenize, batched=True, batch_size=self.preproc_batch_size * max(10, cpu_count()),
                fn_kwargs=dict(
                    should_chunk=False,
                    tokenizer=self.tokenizer,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens
                )
            )

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

            self.dataset = self.dataset.map(
                tokenize, batched=True, batch_size=self.preproc_batch_size, remove_columns=nontext_cols,
                fn_kwargs=dict(
                    should_chunk=self.hparams.chunking_strategy == 'token',
                    tokenizer=self.tokenizer,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens
                )
            )

        # Silence annoying warnings from the tokenizers package after we spawn DataLoader processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        if not isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset.train_test_split(test_size=0.025, shuffle=True)  # noqa

        batch_strategy = self.hparams.batching_strategy
        if batch_strategy != 'random':
            print("Sorting samples by length...")
            self.dataset = self.dataset.sort('num_tokens')

            if batch_strategy == 'uniform_size_prebatched':
                print("Grouping into roughly equal sized batches...")
                self.dataset = self.dataset.map(
                    perform_uniform_size_batching, batched=True, batch_size=self.preproc_batch_size,
                    fn_kwargs=dict(max_size=self.batch_size * max_tokens, length_key='num_tokens')
                )

    def setup(self, stage: Optional[str] = None):
        self.dataset.set_format('torch')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        split = kwargs.get('split') or 'train'
        strategy = self.hparams.batching_strategy
        sampler = None
        shuffle = False
        batch_size = 1

        if strategy == 'uniform_length':
            sampler = ContiguousRandomSampler(dataset_length=len(self.dataset[split]), batch_size=self.batch_size)
        elif strategy == 'uniform_size':
            sampler = UniformSizeRandomSampler(
                sample_lengths=self.dataset[split]['num_tokens'].tolist(),
                max_tokens_per_batch=self.batch_size * self.hparams.max_tokens_per_sample
            )
        else:
            shuffle = True
            if strategy != 'uniform_size_prebatched':
                batch_size = self.batch_size

        return DataLoader(
            self.dataset[split], batch_sampler=sampler, batch_size=batch_size,
            shuffle=shuffle, collate_fn=self.collate, pin_memory=True, num_workers=min(20, cpu_count())
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.train_dataloader(split='test')

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [self.train_dataloader(), self.val_dataloader()]

    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if self.hparams.batching_strategy == 'uniform_size_prebatched':
            text = inputs[0]['text']
            word_counts = inputs[0]['num_words']
        else:
            text = [x['text'] for x in inputs]
            word_counts = [x['num_words'] for x in inputs]

        return {'token_ids': PaddedTensor.from_raw(self.pad_pack(text)), 'word_count': torch.stack(word_counts)}

    def pad_pack(self, batch: List[Tensor], pad_value: int = 0) -> Tensor:
        buffer_len = max(len(x) for x in batch)

        factor = self.hparams.pad_to_multiple_of
        if factor > 1:
            remainder = buffer_len % factor
            buffer_len += factor - remainder if remainder else 0

        buffer = torch.full([len(batch), buffer_len], pad_value, dtype=batch[0].dtype)
        for i, sequence in enumerate(batch):
            buffer[i, :len(sequence)] = sequence

        return buffer

    # Called from prepare_data, as well as from setup when we load from a checkpoint
    def setup_tokenizer(self):
        if self.hparams.use_finetuned_tokenizer:
            tokenizers_dir = Path.cwd() / 'text-vae-pretrained' / 'tokenizers'
            tokenizers_dir.mkdir(parents=True, exist_ok=True)
            vocab_path = tokenizers_dir / (self.hparams.dataset_name + '.json')

            if vocab_path.exists():
                print(f'Loading pretrained tokenizer from {vocab_path}')
                self._tokenizer = Tokenizer.from_file(str(vocab_path))
                assert self.tokenizer.get_vocab_size() == self.hparams.vocab_size
            else:
                print(f'Training a WordPiece tokenizer for the dataset {self.hparams.dataset_name}')

                self._tokenizer = BertWordPieceTokenizer()
                batch_size = self.preproc_batch_size

                def text_iterator():
                    data_len = len(self.dataset)
                    for i in range(0, data_len, batch_size):
                        end = i + batch_size
                        if end > data_len:
                            end = data_len

                        yield self.dataset[i:end]['text']

                self.tokenizer.train_from_iterator(text_iterator(), vocab_size=self.hparams.vocab_size)  # noqa
                self.tokenizer.post_processor = BertProcessing(sep=("[SEP]", 3), cls=("[CLS]", 2))  # noqa
                self.tokenizer.save(str(vocab_path))
        else:
            vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
            print('Using default pretrained tokenizer')
            self._tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)

        # This hyperparameter could change every run of the application so we don't want to save
        # the truncation policy with the tokenizer
        if self.hparams.chunking_strategy == 'token':
            self._tokenizer.enable_truncation(self.hparams.max_tokens_per_sample)
