from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from benchmarks import *
from text_vae import *
from tokenizers import BertWordPieceTokenizer  # noqa
import os
import sys
import torch
from hparam_search import run_hparam_search


def get_checkpoint_path_for_name(experiment: str, ckpt_name: str) -> str:
    ckpt_path = Path.cwd() / 'text-vae-logs' / experiment / ckpt_name / "checkpoints"
    try:
        # Open the most recent checkpoint
        ckpt = max(ckpt_path.glob('*.ckpt'), key=lambda file: file.lstat().st_mtime)
        return str(ckpt)
    except ValueError:
        print(f"Couldn't find checkpoint at path {ckpt_path}")
        exit(1)


def main(args):
    command = args[1]
    model_str = args[2]

    seed_everything(7295)  # Reproducibility

    gpu_available = torch.cuda.is_available()
    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            # 'auto_select_gpus': gpu_available,
            'gpus': int(gpu_available),
            'num_sanity_val_steps': 0
        }
    })
    data_class = AutoencoderDataModule
    hparam_class = None
    model_class = None
    experiment = None

    if model_str == 'funnel':
        data_class = FunnelPreTrainingDataModule
        hparam_class = FunnelTransformerHparams
        model_class = FunnelForPreTraining
        experiment = 'funnel-transformer'

    elif model_str == 'hvae':
        hparam_class = ContinuousHierarchicalVAEHparams
        model_class = ContinuousHierarchicalVAE
        experiment = 'hierarchical-vae'

    elif model_str == 'ar-vae':
        hparam_class = AutoregressiveAutoencoderHparams
        model_class = AutoregressiveAutoencoder
        experiment = 'ar-vae'

    elif model_str == 'flow':
        hparam_class = TextFlowHparams
        model_class = TextFlow
        config.unconditional_sampler = True
        experiment = 'flow'

    elif model_str == 'lstm':
        hparam_class = LSTMAutoencoderHparams
        model_class = LSTMAutoencoder
        experiment = 'lstm-vae'

    elif model_str == 'lstm-lm':
        hparam_class = LSTMLanguageModelHparams
        model_class = LSTMLanguageModel
        experiment = 'lstm-lm'

    elif model_str == 'transformer-lm':
        hparam_class = TransformerLanguageModelHparams
        model_class = TransformerLanguageModel
        experiment = 'transformer-lm'

    elif model_str == 'vq-vae':
        hparam_class = QuantizedVAEHparams
        model_class = QuantizedVAE
        experiment = 'vq-vae'
    else:
        exit(1)

    # Turn these callbacks on by default for VAEs
    if issubclass(model_class, ContinuousVAE):
        config.update(kl_annealing=True)
    else:
        config.trainer.update(precision=16)  # Any model other than a continuous VAE should be able to handle AMP

    if issubclass(model_class, VAE):
        config.update(reconstruction_sampler=True)
    if issubclass(model_class, LanguageModel):
        config.update(unconditional_sampler=True)

    config.data = OmegaConf.structured(AutoencoderDataModuleHparams)
    config.model = OmegaConf.structured(hparam_class)
    config.merge_with_dotlist(args[2:])

    if config.get('anomaly_detection'):
        torch.autograd.set_detect_anomaly(True)

    if command == 'train':
        if config.get('retrain_tokenizer'):
            tokenizer = BertWordPieceTokenizer()
            tokenizer.train_from_iterator()

        print(f"Training {experiment}...")
        if ckpt_name := config.get('from_checkpoint'):
            config.trainer.resume_from_checkpoint = get_checkpoint_path_for_name(experiment, ckpt_name)

    if command == 'extract-posteriors':
        assert issubclass(model_class, VAE)
        ckpt_name = config.get('from_checkpoint')
        assert ckpt_name, "We need a checkpoint to load a model from"

        # Tell AttentionState not to use functools.lru_cache to get around annoying limitations in dill,
        # a HuggingFace datasets dependency
        os.environ['DILL_COMPATIBILITY'] = 'TRUE'
        model = model_class.load_from_checkpoint(get_checkpoint_path_for_name(experiment, ckpt_name))
        if gpu_idx := config.get('gpu'):
            model = model.to('cuda:' + str(gpu_idx))

        data = data_class(hparams=config.data)
        data.prepare_data()
        model.extract_posteriors_for_dataset(data)
        return

    elif command == 'test':
        print(f"Testing a {experiment}...")

    elif command == 'tune':
        run_hparam_search(OmegaConf.from_dotlist(args[2:]))
        return

    model = model_class(config.model)
    data = data_class(hparams=config.data)

    # Automatically add all the callbacks included in the config file by their snake_case names
    callbacks = []
    if config.get('early_stopping'):
        callbacks.append(EarlyStopping(monitor='val_loss'))

    for name, callback_class in AutoencoderCallbackRegistry.items():
        if config.get(name):
            callbacks.append(callback_class())

    # Find the appropriate batch size for this machine and task
    if config.trainer.auto_scale_batch_size or config.trainer.auto_lr_find:
        trainer = Trainer(**config.trainer)
        trainer.tune(model, datamodule=data)

    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir='text-vae-logs',
            name=experiment,
            version=config.get('name')
        )
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
