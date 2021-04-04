# Groups of hyperparameters for reproducing papers
hparam_presets = {
    'he2019': {
        'data': dict(
            batch_size=32,
            batching_strategy='uniform_length',
            chunking_strategy='sentence',
        ),
        'model': dict(
            adam_beta1=0.0,  # 0.0
            adam_beta2=0.0,
            init_scale=0.01,
            lr=1.0,
            lr_plateau_patience=2,
            warmup_steps=0
        ),
        'trainer': dict(
            accumulate_grad_batches=1
        )
    }
}
