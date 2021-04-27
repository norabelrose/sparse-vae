from .quantized_vae import *
from .quantized_latent_data_module import *
from .batch_generation import batch_generate_samples
from .core import LanguageModel, Transformer, get_checkpoint_path_for_name, select_best_gpu
from collections import defaultdict
from contextlib import nullcontext
from datasets import Dataset, DatasetDict, load_from_disk
from functools import partial
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import re


@dataclass
class QuantizedVAESamplingOptions:
    batch_size: int = 1
    benchmark: bool = False
    decode_tokens: bool = True
    profile: bool = False
    max_length: int = 512


@dataclass
class QuantizedVAESampler:
    vae_name: str
    vae: QuantizedVAE
    priors: Dict[int, LanguageModel] = field(default_factory=dict)

    @staticmethod
    def for_vae(name: str):
        sampler_path = Path.cwd() / 'sparse-vae-samplers' / name
        ckpt_path, hparams_path = sampler_path / 'vae.ckpt', sampler_path / 'hparams.yaml'

        # We haven't trained prior models for this VAE yet- prepare to train some
        ckpt_exists, hparams_exist = ckpt_path.exists(), hparams_path.exists()
        if not ckpt_exists or not hparams_exist:
            old_ckpt_path = get_checkpoint_path_for_name('vq-vae', name)
            old_hparams_path = old_ckpt_path.parent.parent / 'hparams.yaml'

            # Create hard link to the checkpoint and hparams in the sparse-vae-samplers folder
            os.makedirs(sampler_path, exist_ok=True)
            if not ckpt_exists:
                old_ckpt_path.link_to(ckpt_path)
            if not hparams_exist:
                old_hparams_path.link_to(hparams_path)

            vae = QuantizedVAE.load_from_checkpoint(ckpt_path, hparams_file=str(hparams_path))
            return QuantizedVAESampler(vae_name=name, vae=vae)  # noqa

        # We've already trained the priors, so just load them
        vae = QuantizedVAE.load_from_checkpoint(ckpt_path, hparams_file=str(hparams_path))

        num_priors = len(vae.quantizers)
        priors = {}

        for x in os.scandir(sampler_path):
            if not x.is_file():
                continue

            match = re.fullmatch(r'prior_([0-9]+)\.ckpt', x.name)
            if not match:
                continue

            prior_id = int(match[1])
            assert 0 <= prior_id < num_priors, \
                f"Expected exactly {num_priors} prior models, but found checkpoint named {str(match)}"

            priors[prior_id] = Transformer.load_from_checkpoint(x.path)

        return QuantizedVAESampler(vae_name=name, vae=vae, priors=priors)  # noqa

    def gather_latents_if_needed(self, datamodule: QuantizedLatentDataModule, dataset_name: str, gpu: Optional[int]):
        if datamodule.dataset:
            return

        latent_dir = os.path.join(os.getcwd(), 'sparse-vae-datasets', 'latents', dataset_name, self.vae_name)
        try:
            datamodule.dataset = load_from_disk(latent_dir)
        except:
            pass
        else:
            return

        os.makedirs(latent_dir, exist_ok=True)

        print(f"No pre-computed latents found for version '{self.vae_name}' and dataset '{dataset_name}'. "
              f"Preparing to gather latent codes from the VQ-VAE for each sample in the dataset.")

        vae_dm_hparams = OmegaConf.structured(TextDataModuleHparams)
        vae_dm_hparams.dataset_name = dataset_name
        vae_dm_hparams.batch_size = 64
        vae_datamodule = TextDataModule(vae_dm_hparams)

        self.vae.gathering_latents = True

        gpu = gpu or select_best_gpu() if torch.cuda.is_available() else None
        trainer = Trainer(precision=16, gpus=[gpu])
        outputs = trainer.predict(model=self.vae, datamodule=vae_datamodule)
        outputs = sum(outputs, [])  # Concatenate the results from the dataloaders

        z_dataset = defaultdict(list)
        for features in outputs:
            for name, result in features.items():
                lengths = (~result['padding']).sum(-1)
                trimmed_z = [z[:length] for z, length in zip(result['data'], lengths)]
                z_dataset[name].extend(trimmed_z)

        z_dataset = Dataset.from_dict(z_dataset)
        z_dataset = z_dataset.train_test_split(test_size=0.05, shuffle=False)
        z_dataset.save_to_disk(latent_dir)
        datamodule.dataset = z_dataset

        self.vae = self.vae.cpu()  # Save GPU memory for the prior models
        self.vae.datamodule = None

    def train_priors(self, hparams: DictConfig, force_retrain: bool = False):
        for level in range(len(self.vae.quantizers)):
            if not force_retrain and level in self.priors:
                continue

            self.train_prior(level, hparams)

    def train_prior(self, level: int, hparams: DictConfig):
        num_levels = len(self.vae.quantizers)
        prior_name = QuantizedVAE.name_for_latent_level(level, num_levels)

        num_codes = self.vae.quantizers[level].num_codes
        start_code = num_codes
        end_code = num_codes + 1

        gpus = hparams.trainer.gpus
        dm_hparams = deepcopy(hparams.data)

        # scales = self.vae.decoder.hparams.scaling_factors
        dm_hparams.codebook_size = num_codes
        dm_hparams.latent_key = prior_name
        dm_hparams.vae_version_name = self.vae_name
        if level > 0:
            dm_hparams.context_key = QuantizedVAE.name_for_latent_level(level - 1, num_levels)

        datamodule = QuantizedLatentDataModule(dm_hparams)
        self.gather_latents_if_needed(datamodule, hparams.data.dataset_name, gpus[0] if gpus else None)

        prior_hparams = deepcopy(hparams.prior)
        prior_hparams.log_samples = False
        prior_hparams.num_layers = 6
        prior_hparams.start_token = start_code
        prior_hparams.end_token = end_code
        prior_hparams.cross_attention = level > 0
        prior_hparams.vocab_size = num_codes + 2  # Include the start and end codes
        prior = Transformer(prior_hparams)

        if torch.cuda.is_available() and 'gpus' not in hparams.trainer:
            hparams.trainer.gpus = [select_best_gpu()]

        trainer = Trainer(**hparams.trainer, logger=TensorBoardLogger(
            save_dir='sparse-vae-logs',
            name='vq-vae-prior'
        ))

        print(f"Fitting {prior_name} prior...")
        trainer.fit(prior, datamodule=datamodule)

        ckpt_monitor = trainer.checkpoint_callback
        ckpt_path = ckpt_monitor.best_model_path
        if not ckpt_path:
            print("Training seems to have failed.")
            return

        dest_path = os.path.join(os.getcwd(), 'sparse-vae-samplers', self.vae_name, f'prior_{level}.ckpt')
        print(f"Fitted {prior_name} prior with val loss {ckpt_monitor.best_model_score}. "
              f"Creating hard link to checkpoint at {dest_path}.")

        if os.path.exists(dest_path):
            print(f"Deleting old checkpoint at {dest_path} in order to make room for the new one.")
            os.remove(dest_path)

        os.link(ckpt_path, dest_path)

        del prior  # Save GPU memory- also, this may not be the best checkpoint
        self.priors[level] = Transformer.load_from_checkpoint(dest_path)  # This is on the CPU

    @torch.no_grad()
    def sample(self, options: QuantizedVAESamplingOptions):
        vae, priors = self.vae, self.priors
        num_levels = len(vae.quantizers)
        assert len(priors) == num_levels,\
            "You need to train a prior for each latent level before you can sample"

        if options.benchmark:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = None; end = None

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True, record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(os.getcwd(), 'sparse-vae-profiling'))
        ) if options.profile else nullcontext()

        with profiler:
            logits = self._raw_sample(vae, priors, options)

        if end:
            end.record()
            torch.cuda.synchronize()

            num_tokens = options.batch_size * options.max_length
            print(f"Generated roughly {num_tokens} tokens in {start.elapsed_time(end)} ms.")

        return logits

    def gather_samples_if_needed(self, datamodule: TextDataModule, dataset_name: str):
        if datamodule.dataset:
            return

        sample_dir = os.path.join(os.getcwd(), 'sparse-vae-datasets', 'samples', dataset_name, self.vae_name)
        try:
            datamodule.dataset = load_from_disk(sample_dir)
        except:
            pass
        else:
            return

        os.makedirs(sample_dir, exist_ok=True)

        original_dataset = load_dataset(dataset_name, cache_dir='sparse-vae-datasets')
        if not isinstance(original_dataset, DatasetDict):
            num_samples = len(original_dataset)
        else:
            num_samples = sum(len(x) for x in original_dataset.values())

        del original_dataset
        print(f"No pre-computed samples found for version '{self.vae_name}' and dataset '{dataset_name}'. "
              f"Preparing to gather {num_samples} samples from the VQ-VAE.")

        self.write_samples_to_disk(Path(sample_dir), num_samples // 500, QuantizedVAESamplingOptions(batch_size=500))  # noqa

    @torch.no_grad()
    def write_samples_to_disk(self, path: Path, num_batches: int, options: QuantizedVAESamplingOptions):
        sample_func = partial(self._raw_sample, self.vae, self.priors, options)
        outputs = batch_generate_samples(sample_func, num_batches, options.batch_size, end_token=self.vae.end_token)

        print("Writing to disk...")
        dataset = Dataset.from_dict({'text': outputs})
        dataset.save_to_disk(str(path))
        print("Done.")

    @staticmethod
    @torch.cuda.amp.autocast()
    def _raw_sample(vae: QuantizedVAE, priors: List[Transformer], options: QuantizedVAESamplingOptions):
        # Find the maximum sequence length that the top latent sequence can be allowed to have in order to ensure
        # that the final overt sequence length does not exceed max_length
        num_codes = vae.quantizers[0].num_codes
        strides = vae.decoder.strides()
        if not vae.hparams.include_full_res_latents:
            del strides[-1]

        latent_seqs = []
        for level, stride in enumerate(strides):
            prior = priors[level]
            context = latent_seqs[-1].data if latent_seqs else None

            max_z_length = options.max_length // stride
            z = prior.sample(max_length=max_z_length + 2, batch_size=options.batch_size, context=context)[..., 1:].squeeze(1)
            z[z > num_codes] = 0
            latent_seqs.append(PaddedTensor.from_raw(z))

        return vae.decode_prior_samples(latent_seqs).argmax(dim=-1)

    def to(self, device: Union[torch.device, str]):
        self.vae = self.vae.to(device)  # noqa
        self.priors = {k: v.to(device) for k, v in self.priors}
        return self


# Implemented as a freestanding function in order to avoid annoying dill errors when using dataset.map()
def _gather_latents(batch: Dict[str, list], vae: QuantizedVAE, datamodule: TextDataModule) -> Dict[str, list]:
    batch = [dict(zip(batch, x)) for x in zip(*batch.values())]  # dict of lists -> list of dicts
    batch = datamodule.collate(batch)

    results = vae.predict(batch, batch_idx=0)
    output = {}

    # Trim the padding off the latents
    for name, result in results.items():
        latents = result['data'].tolist()
        lengths = result['padding'].logical_not().sum(dim=-1).tolist()

        output[name] = [z[:length] for z, length in zip(latents, lengths)]

    return output
