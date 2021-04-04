from pytorch_lightning import seed_everything
from text_vae import *
from hparam_presets import hparam_presets
import sys
import torch


def main(args):
    model_str = args[1]

    seed_everything(7295)  # Reproducibility
    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'accumulate_grad_batches': 4,
            'precision': 16,
            'num_sanity_val_steps': 2
        }
    })

    data_class = TextDataModule
    data_hparam_class = TextDataModuleHparams
    hparam_class = None
    model_class = None
    experiment = None

    if model_str == 'lstm':
        hparam_class = LSTMAutoencoderHparams
        model_class = LSTMAutoencoder
        experiment = 'lstm-vae'

    elif model_str == 'lstm-lm':
        hparam_class = LSTMLanguageModelHparams
        model_class = LSTMLanguageModel
        experiment = 'lstm-lm'

    elif model_str == 'transformer-lm':
        hparam_class = TransformerHparams
        model_class = Transformer
        experiment = 'transformer-lm'

    elif model_str == 'vq-vae':
        hparam_class = QuantizedVAEHparams
        model_class = QuantizedVAE
        experiment = 'vq-vae'
    else:
        print(f"Unrecognized model type '{model_str}'.")
        exit(1)

    config.data = OmegaConf.structured(data_hparam_class)
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
    data = data_class(hparams=config.data)

    # Find the appropriate batch size for this machine and task
    if config.trainer.auto_scale_batch_size or config.trainer.auto_lr_find:
        trainer = Trainer(**config.trainer)
        trainer.tune(model, datamodule=data)

    warnings.filterwarnings('ignore', module='pytorch_lightning')
    warnings.filterwarnings('ignore', module='torch')  # Bug in PL

    if config.get('no_log'):
        logger = False
    else:
        logger = TensorBoardLogger(
            save_dir='text-vae-logs',
            name=experiment,
            version=config.get('name')
        )

    trainer = Trainer(**config.trainer, logger=logger)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
