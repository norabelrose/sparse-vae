from datasets import concatenate_datasets
from text_vae.AutoencoderDataModule import *


# noinspection PyAbstractClass
class FunnelPreTrainingDataModule(AutoencoderDataModule):
    dataset_name: ClassVar[str] = 'funnel_pretraining'

    def create_dataset(self):
        self.hparams.masked_lm = True
        cache_dir = self.hparams.dataset_save_dir

        wikipedia = load_dataset('wikipedia', '20200501.en', split='train[:5%]', cache_dir=cache_dir)
        bookcorpus = load_dataset('bookcorpusopen', split='train[:5%]', cache_dir=cache_dir)
        openwebtext = load_dataset('openwebtext', split='train[:5%]', cache_dir=cache_dir)
        wikipedia.remove_columns_('title')
        bookcorpus.remove_columns_('title')

        combined = concatenate_datasets([wikipedia, bookcorpus, openwebtext])
        self.dataset = combined.shuffle()   # This is just to make the tqdm ETA not be totally off when tokenizing
