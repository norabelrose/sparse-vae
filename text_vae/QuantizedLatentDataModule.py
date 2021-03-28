from .TextDataModule import *
from datasets import load_from_disk
import torch.nn.functional as F


@dataclass
class QuantizedLatentDataModuleHparams(TextDataModuleHparams):
    codebook_size: int = 4096
    gpu: Optional[int] = None  # GPU to use if we need to gather latent codes
    latent_key: str = 'top'  # The key to use to look up the latents in the dataset
    context_key: Optional[str] = None  # Key to look up the next level up latents, used by the prior as context
    vae_version_name: Optional[str] = None  # If None, the most recent version is used


class QuantizedLatentDataModule(TextDataModule):
    def __init__(self, hparams: DictConfig):
        super(QuantizedLatentDataModule, self).__init__(hparams)

        num_codes = self.hparams.codebook_size
        self.start_code = num_codes
        self.end_code = num_codes + 1

    # We don't need to do any of the tokenizing, chunking, etc. that is done in super().prepare_data
    def prepare_data(self, *args, **kwargs):
        dataset_name = self.hparams.dataset_name
        dir_path = os.path.join(self.hparams.dataset_save_dir, 'latents', dataset_name)

        latent_dir = os.path.join(dir_path, self.hparams.vae_version_name)
        assert os.path.exists(latent_dir)

        self.dataset = load_from_disk(latent_dir)
        os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

    def collate(self, inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        key = self.hparams.latent_key
        codes = self.pad_pack([x[key] for x in inputs])
        batch = {'token_ids': codes, 'padding_mask': codes.eq(0)}

        ctx_key = self.hparams.context_key
        if ctx_key:
            ctx = self.pad_pack([x[ctx_key] for x in inputs])
            batch['context'] = ctx
            batch['padding_mask_ctx'] = ctx.eq(0)

        return batch

    # Add special start and end codes to each latent sequence- then pad pack the sequences together.
    def pad_pack(self, tokens: List[Tensor], pad_value: float = 0) -> Tensor:
        for i in range(len(tokens)):
            x = F.pad(tokens[i], (1, 1), value=self.start_code)
            x[..., -1] = self.end_code
            tokens[i] = x

        return super().pad_pack(tokens)
