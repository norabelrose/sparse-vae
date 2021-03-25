from pytorch_lightning import seed_everything
from benchmarks import *
from text_vae import *
import sys
import torch
from hparam_search import run_hparam_search


def main(args):
    command = args[1]
    model_str = args[2]

    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'precision': 16,
            'num_sanity_val_steps': 2,
            'terminate_on_nan': True
        }
    })

    if command == 'train-prior':
        sampler = QuantizedVAESampler.for_vae(model_str)

        config.data = OmegaConf.structured(QuantizedLatentDataModuleHparams)
        config.prior = OmegaConf.structured(TransformerHparams)

        config.merge_with_dotlist(args[2:])
        sampler.train_priors(config)
        return

    data_class = TextDataModule
    data_hparam_class = TextDataModuleHparams
    hparam_class = None
    model_class = None
    experiment = None

    if model_str == 'electra':
        data_class = ElectraDataModule
        data_hparam_class = MLMDataModuleHparams
        hparam_class = ElectraModelHparams
        model_class = ElectraModel
        experiment = 'electra'

    elif model_str == 'hvae':
        hparam_class = ContinuousHierarchicalVAEHparams
        model_class = ContinuousHierarchicalVAE
        experiment = 'hierarchical-vae'

    elif model_str in ('adv-ae', 'daae'):
        hparam_class = AdversarialAutoencoderHparams
        model_class = AdversarialAutoencoder
        experiment = model_str

        if model_str == 'daae':
            data_class = MLMDataModule
            data_hparam_class = MLMDataModuleHparams

    elif model_str == 'lstm':
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
        exit(1)

    config.data = OmegaConf.structured(data_hparam_class)
    config.model = OmegaConf.structured(hparam_class)
    config.merge_with_dotlist(args[2:])

    if torch.cuda.is_available() and 'gpus' not in config.trainer:
        config.trainer.gpus = [select_best_gpu()]

    if config.get('anomaly_detection'):
        torch.autograd.set_detect_anomaly(True)

    if command == 'train':
        seed_everything(7295)  # Reproducibility

        print(f"Training {experiment}...")
        if ckpt_name := config.get('from_checkpoint'):
            config.trainer.resume_from_checkpoint = str(get_checkpoint_path_for_name(experiment, ckpt_name))

    elif command == 'sample':
        ckpt_name = config.get('from_checkpoint')
        assert ckpt_name, "We need a checkpoint to load a model from"

        model = model_class.load_from_checkpoint(get_checkpoint_path_for_name(experiment, ckpt_name))
        if gpu_idx := config.get('gpu'):
            model = model.to('cuda:' + str(gpu_idx))

        num_samples = config.get('samples', 1)
        max_length = config.get('max_length', 40)
        dataset_name = config.get('dataset_name', 'yelp_polarity')

        vocab_path = Path.cwd() / 'text-vae-pretrained' / 'tokenizers' / (dataset_name + '.json')
        assert vocab_path.exists(), f"Couldn't find pretrained tokenizer for {dataset_name}"

        tokenizer = Tokenizer.from_file(str(vocab_path))
        model.tokenizer = tokenizer

        while True:
            # Running the Markov chain takes a while so we want to print each iteration
            # to the screen as it runs
            if model_class == AdversarialAutoencoder:
                num_iter = config.get('num_iter', 200)
                masked_tokens_per_iter = config.get('masked_tokens_per_iter', 1)

                for i, iteration in enumerate(model.markov_chain_sample(max_length, num_samples, num_iter=num_iter,
                                                                        masked_tokens_per_iter=masked_tokens_per_iter)):
                    if i == 0:
                        print("Initial sample:")
                    else:
                        print(f"Iteration {i}:")

                    text = tokenizer.decode_batch(iteration.tolist())
                    if len(text) == 1:
                        text = text[0]

                    print(text)
            else:
                samples = model.sample(max_length, num_samples)
                print(samples)

            if input("Would you like to sample again? (y/n): ")[0].lower() != "y":
                return

    elif command == 'test':
        print(f"Testing a {experiment}...")

    elif command == 'tune':
        run_hparam_search(OmegaConf.from_dotlist(args[2:]))
        return

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
