from .QuantizedVAE import *
from .QuantizedLatentDataModule import *
from .core import GenerationStrategy, Transformer, select_best_gpu
from contextlib import nullcontext
from datasets import Dataset, load_from_disk
from numpy import cumprod
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from tqdm.auto import tqdm
import gc
import re


@dataclass
class QuantizedVAESamplingOptions:
    batch_size: int = 1
    benchmark: bool = False
    decode_tokens: bool = True
    profile: bool = False
    max_length: int = 250
    strategy: GenerationStrategy = GenerationStrategy.SamplingTopK


@dataclass
class QuantizedVAESampler:
    vae_name: str
    vae: QuantizedVAE
    priors: Dict[int, Transformer] = field(default_factory=dict)

    @staticmethod
    def for_vae(name: str):
        sampler_path = Path.cwd() / 'text-vae-samplers' / name
        ckpt_path, hparams_path = sampler_path / 'vae.ckpt', sampler_path / 'hparams.yaml'

        # We haven't trained prior models for this VAE yet- prepare to train some
        ckpt_exists, hparams_exist = ckpt_path.exists(), hparams_path.exists()
        if not ckpt_exists or not hparams_exist:
            old_ckpt_path = get_checkpoint_path_for_name('vq-vae', name)
            old_hparams_path = old_ckpt_path.parent.parent / 'hparams.yaml'

            # Create hard link to the checkpoint and hparams in the text-vae-samplers folder
            os.makedirs(sampler_path, exist_ok=True)
            if not ckpt_exists:
                old_ckpt_path.link_to(ckpt_path)
            if not hparams_exist:
                old_hparams_path.link_to(hparams_path)

            vae = QuantizedVAE.load_from_checkpoint(ckpt_path, hparams_file=str(hparams_path), strict=False)
            return QuantizedVAESampler(vae_name=name, vae=vae)  # noqa

        # We've already trained the priors, so just load them
        vae = QuantizedVAE.load_from_checkpoint(ckpt_path, hparams_file=str(hparams_path), strict=False)

        num_priors = vae.quantizer.num_levels
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

        latent_dir = os.path.join(os.getcwd(), 'text-vae-datasets', 'latents', dataset_name, self.vae_name)
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
        vae_datamodule = TextDataModule(vae_dm_hparams)

        self.vae.datamodule = vae_datamodule
        self.vae.gathering_latents = True
        self.vae.freeze()
        self.vae.setup('test')

        if torch.cuda.is_available():
            gpu = select_best_gpu() if gpu is None else gpu
            self.vae = self.vae.to('cuda:' + str(gpu))

        vae_datamodule.prepare_data()
        vae_datamodule.setup(stage='predict')
        text_dataset = vae_datamodule.dataset
        text_cols = text_dataset.column_names['train']

        datamodule.dataset = text_dataset.map(
            _gather_latents, batched=True, batch_size=500, fn_kwargs=dict(vae=self.vae, datamodule=vae_datamodule),
            load_from_cache_file=False, remove_columns=text_cols
        )
        datamodule.dataset.save_to_disk(latent_dir)

        self.vae = self.vae.cpu()  # Save GPU memory for the prior models
        self.vae.datamodule = None

    def train_priors(self, hparams: DictConfig, force_retrain: bool = False):
        for level in range(self.vae.quantizer.num_levels):
            if not force_retrain and level in self.priors:
                continue

            self.train_prior(level, hparams)

    def train_prior(self, level: int, hparams: DictConfig):
        num_levels = self.vae.quantizer.num_levels
        prior_name = QuantizedVAE.name_for_latent_level(level, num_levels)

        num_codes = self.vae.quantizer.num_codes
        start_code = num_codes
        end_code = num_codes + 1

        gpus = hparams.trainer.gpus
        dm_hparams = deepcopy(hparams.data)

        scales = self.vae.decoder.hparams.scaling_factors
        dm_hparams.batch_size *= int(cumprod(scales)[::-1][level])
        dm_hparams.codebook_size = num_codes
        dm_hparams.latent_key = prior_name
        dm_hparams.vae_version_name = self.vae_name
        if level > 0:
            dm_hparams.context_key = QuantizedVAE.name_for_latent_level(level - 1, num_levels)

        datamodule = QuantizedLatentDataModule(dm_hparams)
        self.gather_latents_if_needed(datamodule, hparams.data.dataset_name, gpus[0] if gpus else None)

        prior_hparams = deepcopy(hparams.prior)
        prior_hparams.log_samples = False
        prior_hparams.lr = 1e-4
        prior_hparams.num_layers = scales[::-1][level]
        prior_hparams.start_token = start_code
        prior_hparams.end_token = end_code
        prior_hparams.cross_attention = level > 0
        prior_hparams.vocab_size = num_codes + 2  # Include the start and end codes
        prior_hparams.warmup_steps = 250
        prior = Transformer(prior_hparams)

        if torch.cuda.is_available() and 'gpus' not in hparams.trainer:
            hparams.trainer.gpus = [select_best_gpu()]

        ckpt_monitor = ModelCheckpoint(monitor='val_nll', mode='min')
        trainer = Trainer(**hparams.trainer, logger=TensorBoardLogger(
            save_dir='text-vae-logs',
            name='vq-vae-prior'
        ))
        trainer.callbacks = [
            EarlyStopping(monitor='val_nll', mode='min'),
            ckpt_monitor
        ]

        print(f"Fitting {prior_name} prior...")
        trainer.fit(prior, datamodule=datamodule)

        ckpt_path = ckpt_monitor.best_model_path
        if not ckpt_path:
            print("Training seems to have failed.")
            return

        dest_path = os.path.join(os.getcwd(), 'text-vae-samplers', self.vae_name, f'prior_{level}.ckpt')
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
        num_levels = vae.quantizer.num_levels
        assert len(priors) == num_levels,\
            "You need to train a prior for each latent level before you can sample"

        if options.benchmark:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = None; end = None

        with torch.autograd.profiler.profile(use_cuda=vae.on_gpu) if options.profile else nullcontext():
            logits = self._raw_sample(vae, priors, options)

        if end:
            end.record()
            torch.cuda.synchronize()

            num_tokens = options.batch_size * options.max_length
            print(f"Generated roughly {num_tokens} tokens in {start.elapsed_time(end)} ms.")

        return logits

    @torch.no_grad()
    def write_samples_to_disk(self, path: Path, num_batches: int, options: QuantizedVAESamplingOptions):
        vae, priors = self.vae, self.priors

        batches_left = num_batches
        batch_size = options.batch_size
        total_samples = num_batches * batch_size
        pbar = tqdm(desc='Generating samples', total=total_samples, unit='samples')

        assert options.decode_tokens
        outputs = []

        # CPU sampling
        if not vae.on_gpu:
            while batches_left > 0:
                batch = self._raw_sample(vae, priors, options)
                outputs.extend(batch.tolist())

                # Let user know we've finished a batch right now
                pbar.update(n=batch_size)
                batches_left -= 1

        # GPU sampling
        else:
            gpu_outputs = []
            oom_count = 0

            def flush_to_cpu():
                nonlocal gpu_outputs, outputs

                # Sequentially move batches from the GPU to the CPU, blocking the current thread for each one.
                gpu_outputs.reverse()

                while gpu_outputs:
                    gpu_batch = gpu_outputs.pop()
                    cpu_batch = gpu_batch.tolist()

                    del gpu_batch  # Get it out of GPU memory ASAP
                    outputs.extend(cpu_batch)

                    # Let user know we've finished a batch right now
                    pbar.update(n=batch_size)

            # Dispatch all the instructions to the GPU all at once, without forcing a blocking
            # device-host synchronization between each batch. If we get an OOM error, move everything
            # to the CPU and flush to disk, then continue dispatching to the GPU.
            while batches_left > 0:
                try:
                    gpu_batch = self._raw_sample(vae, priors, options)

                # Catch OOM errors
                except RuntimeError:
                    # Normal case, we have outstanding GPU batches we need to flush
                    if gpu_outputs:
                        flush_to_cpu()
                        gc.collect()

                    # This shouldn't really happen, but we'll clear the memory cache as a last resort
                    elif oom_count == 0:
                        torch.cuda.empty_cache()
                        oom_count += 1

                    # Something is seriously wrong, we're getting OOMs even after we clear the CUDA cache
                    else:
                        raise
                else:
                    gpu_outputs.append(gpu_batch)
                    batches_left -= 1

            if gpu_outputs:
                flush_to_cpu()

        pbar.close()

        # Remove padding/garbage tokens from the generated samples after [SEP]
        sep = vae.end_token
        last_index = options.max_length - 1
        for sample in outputs:
            try:
                sep_index = sample.index(sep, 0, last_index)
            except ValueError:
                pass
            else:
                del sample[sep_index + 1:]

        dataset = Dataset.from_dict({'text': outputs})
        dataset.save_to_disk(str(path))

    @staticmethod
    def _raw_sample(vae: QuantizedVAE, priors: List[Transformer], options: QuantizedVAESamplingOptions):
        # Find the maximum sequence length that the top latent sequence can be allowed to have in order to ensure
        # that the final overt sequence length does not exceed max_length
        funnel_hparams = vae.hparams.funnel
        strides = cumprod(funnel_hparams.scaling_factors)[::-1]

        latent_seqs = []
        for level, stride in enumerate(strides):
            prior = priors[level]
            context = latent_seqs[-1] if latent_seqs else None

            max_z_length = options.max_length // stride
            z = prior.sample(max_length=max_z_length + 2, count=options.batch_size, context=context,
                             include_padding=True)
            latent_seqs.append(z)

        vae.decoder.attention_state.configure_for_input(
            seq_len=options.max_length
        )
        latent_seqs = [vae.quantizer.lookup_codes(codes[..., 1:-1], level) for level, codes in enumerate(latent_seqs)]

        vae_input = QuantizedVAEState()
        vae_input.decoder_input = vae.quantizer.upsample_codes(latent_seqs[0], level=0)
        vae_input.encoder_states = latent_seqs[1:]
        return vae.decoder_forward(vae_input).logits.argmax(dim=-1)

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
