from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from benchmarks.LSTMAutoencoder import LSTMAutoencoder, LSTMAutoencoderHparams
from text_vae import *
import sys
import torch


def main(args):
    command = args[1]

    seed_everything(7295)  # Reproducibility

    # torch.autograd.set_detect_anomaly(True)
    gpu_available = torch.cuda.is_available()
    config = OmegaConf.create({
        'aggressive_encoder_training': False,
        'kl_annealing': False,
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'auto_select_gpus': gpu_available,
            'gpus': int(gpu_available)
        }
    })
    data_class = AutoencoderDataModule

    if command == 'finetune-funnel':
        print("Finetuning a pretrained Funnel Transformer for Performer attention...")

        data_class = FunnelPreTrainingDataModule
        hparam_class = FunnelTransformerHparams
        model_class = FunnelForPreTraining

    elif command == 'train':
        print("Training a Text VAE...")

        hparam_class = HierarchicalAutoencoderHparams
        model_class = HierarchicalAutoencoder

    elif command == 'train-lstm':
        print("Training an LSTM VAE...")

        hparam_class = LSTMAutoencoderHparams
        model_class = LSTMAutoencoder
    else:
        raise NotImplementedError

    config.data = OmegaConf.structured(AutoencoderDataModuleHparams)
    config.model = OmegaConf.structured(hparam_class)
    config.merge_with_dotlist(args[2:])

    if ckpt_name := config.get('from_checkpoint'):
        ckpt_path = Path.cwd() / "lightning_logs" / ckpt_name / "checkpoints"
        try:
            # Open the most recent checkpoint
            ckpt = max(ckpt_path.glob('*.ckpt'), key=lambda file: file.lstat().st_mtime)
        except ValueError:
            print(f"Couldn't find checkpoint at path {ckpt_path}")
            exit(1)
        else:
            config.trainer.resume_from_checkpoint = str(ckpt)

    data = data_class(hparams=config.data)
    model = model_class(config.model)

    callbacks = [EarlyStopping(monitor='val_loss'), UnconditionalSampler()]
    if config.aggressive_encoder_training:
        callbacks.append(AggressiveEncoderTraining())
    if config.kl_annealing:
        callbacks.append(KLAnnealing())

    trainer = Trainer(**config.trainer, callbacks=callbacks)

    # Find the appropriate batch size for this machine and task
    if config.trainer.auto_scale_batch_size:
        trainer.tune(model, datamodule=data)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
