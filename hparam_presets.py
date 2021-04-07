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
            decoder_input_dropout=0.5,
            decoder_output_dropout=0.5,
            divide_loss_by_length=False,
            grad_clip_threshold=5.0,
            init_scale=0.01,
            lr=1.0,
            lr_plateau_patience=2,
            tie_embedding_weights=False,
            warmup_steps=0
        ),
        'trainer': dict(
            accumulate_grad_batches=1,
            precision=32    # Diverges without this
        )
    },
    # From https://github.com/timbmg/Sentence-VAE/
    'sentence-vae': {
        'data': dict(
            batch_size=32,
            batching_strategy='uniform_length',
            chunking_strategy='sentence',
        ),
        'model': dict(
            d_model=256,
            d_embedding=300,
            divide_loss_by_length=False,
            decoder_input_dropout=0.5,
            init_scale=None,    # Default PyTorch initialization
            latent_depth=16,
            lr=1e-3,
            tie_embedding_weights=True,
            warmup_steps=0
        ),
        'trainer': dict(
            accumulate_grad_batches=1,
            precision=32
        )
    },
    'belrose-lstm': {
        'data': dict(
            batch_size=64,
            batching_strategy='uniform_length',
            chunking_strategy='none',
        ),
        'model': dict(
            bidirectional_encoder=True,
            divide_loss_by_length=True,
            d_model=1024,
            d_embedding=512,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.3,
            kl_annealing_steps=8_000,
            latent_depth=512,
            lr=5e-4,
            num_latent_vectors=1,
            tie_embedding_weights=True,
            tie_logit_weights=True,
            transformer_encoder=True,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=1
        )
    },
    'belrose-transformer': {
        'data': dict(
            batch_size=64,
            batching_strategy='uniform_length',
            chunking_strategy='none',
        ),
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.0,
            kl_annealing_steps=50_000,
            latent_depth=64,
            lr=1e-4,
            num_latent_vectors=1,
            num_layers=3,
            tie_embedding_weights=True,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=1
        )
    },
}
