from text_vae import HierarchicalAutoencoder
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


# Uses combination of Bayesian optimization and HyperBand early stopping
def run_hparam_search(tune_config: DictConfig, num_trials: int = 100, max_epochs_per_trial: int = 10):
    # Avoid importing RayTune unless we have to
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.integration.pytorch_lightning import TuneReportCallback
    from ray.tune.schedulers import HyperBandForBOHB
    from ray.tune.suggest.bohb import TuneBOHB

    def run_trial(config: dict):
        autoencoder = HierarchicalAutoencoder(OmegaConf.create(config))
        trainer = Trainer(
            max_epochs=max_epochs_per_trial,
            logger=TensorBoardLogger(
                save_dir=tune.get_trial_dir(),
                name="",
                version="."
            ),
            progress_bar_refresh_rate=0,
            callbacks=[
                TuneReportCallback("val_loss")  # Report the validation loss back to Ray Tune at every epoch
            ]
        )
        trainer.fit(autoencoder)

    analysis = tune.run(
        run_or_experiment=run_trial,
        name="tune_text_vae",
        metric="val_loss",
        mode="min",
        config={
            'lr': tune.loguniform(5e-5, 1e-3),
            'latent_depth': tune.randint(1, 16),
            'num_latent_groups': tune.randint(1, 50)
        },
        progress_reporter=CLIReporter(),
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        scheduler=HyperBandForBOHB(
            max_t=max_epochs_per_trial,
            metric="val_loss",
            mode="min"
        ),
        search_alg=TuneBOHB(
            max_concurrent=2,
            metric="val_loss",
            mode="min"
        ),
        # Pass in command line arguments
        **tune_config
    )
    print(f"Best configuration was trial {analysis.best_trial} with loss {analysis.best_result} and "
          f"hyperparameters {analysis.best_config}.")
