from datasets import concatenate_datasets
from .MLMDataModule import *


# noinspection PyAbstractClass
class ElectraDataModule(MLMDataModule):
    def __init__(self, hparams: DictConfig):
        hparams.dataset_name = 'electra'
        hparams.max_tokens_per_sample = 256
        hparams.use_finetuned_tokenizer = False
        hparams.uniform_length_batching = False

        hparams.include_unmasked_tokens_in_labels = False
        hparams.pad_to_max_length = True
        hparams.use_smart_random_tokens = False
        hparams.yield_segment_pairs = True

        super(ElectraDataModule, self).__init__(hparams)

    def create_dataset(self):
        cache_dir = self.hparams.dataset_save_dir

        wikipedia = load_dataset('wikipedia', '20200501.en', split='train[:5%]', cache_dir=cache_dir)
        bookcorpus = load_dataset('bookcorpusopen', split='train[:5%]', cache_dir=cache_dir)
        openwebtext = load_dataset('openwebtext', split='train[:5%]', cache_dir=cache_dir)
        wikipedia.remove_columns_('title')
        bookcorpus.remove_columns_('title')

        self.dataset = concatenate_datasets([wikipedia, bookcorpus, openwebtext])
