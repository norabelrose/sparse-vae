from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from sparse_vae import *
from hparam_presets import hparam_presets
from omegaconf import OmegaConf
import sys
import torch


def main(args):
    model_str = args[1]

    seed_everything(7295)  # Reproducibility
    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'accumulate_grad_batches': 2,
            'precision': 16
        }
    })

    hparam_class = None
    model_class = None
    experiment = None

    if model_str == 'lstm-vae':
        hparam_class = LSTMVAEHparams
        model_class = LSTMVAE
        experiment = 'lstm-vae'

    elif model_str == 'lstm-lm':
        hparam_class = LSTMLanguageModelHparams
        model_class = LSTMLanguageModel
        experiment = 'lstm-lm'

    elif model_str == 'transformer-lm':
        hparam_class = TransformerHparams
        model_class = Transformer
        experiment = 'transformer-lm'

    elif model_str == 'transformer-vae':
        hparam_class = TransformerVAEHparams
        model_class = TransformerVAE
        experiment = 'transformer-vae'

    else:
        print(f"Unrecognized model type '{model_str}'.")
        exit(1)

    config.data = OmegaConf.structured(TextDataModuleHparams)
    config.model = OmegaConf.structured(hparam_class)

    config.merge_with_dotlist(args[2:])
    if preset := config.get('preset'):
        preset_config = hparam_presets.get(preset)
        assert preset_config, f"Preset name '{preset}' not recognized."
        config.merge_with(preset_config)

    if torch.cuda.is_available() and 'gpus' not in config.trainer:
        config.trainer.gpus = [select_best_gpu()]

    if config.get('anomaly_detection'):
        torch.autograd.set_detect_anomaly(True)

    print(f"Training {experiment}...")
    if ckpt_name := config.get('from_checkpoint'):
        config.trainer.resume_from_checkpoint = str(get_checkpoint_path_for_name(experiment, ckpt_name))

    model = model_class(config.model)
    data = TextDataModule(hparams=config.data)

    if config.get('fp16_weights'):
        torch.set_default_dtype(torch.float16)

    if config.get('no_log'):
        logger = False
    else:
        logger = TensorBoardLogger(
            save_dir='sparse-vae-logs',
            name=experiment,
            version=config.get('name')
        )

    profiler = PyTorchProfiler(
        profile_memory=True,
        sort_by_key='cuda_memory_usage',
        use_cuda=True
    ) if config.get('profile') else None

    trainer = Trainer(**config.trainer, logger=logger, profiler=profiler)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
