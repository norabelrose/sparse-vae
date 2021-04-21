# Groups of hyperparameters for reproducing papers
hparam_presets = {
    'he2019': {
        'data': dict(
            batch_size=32,
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
    'lstm-benchmark': {
        'model': dict(
            bidirectional_encoder=True,
            divide_loss_by_length=True,
            d_model=1024,
            d_embedding=512,
            grad_clip_threshold=5.0,
            init_scale=None,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_latent_vectors=1,
            tie_embedding_weights=True,
            tie_logit_weights=True,
            transformer_encoder=False,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'lstm-wikipedia': {
        'data': dict(
            chunking_strategy='none',
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            max_tokens_per_sample=12_500
        ),
        'model': dict(
            bidirectional_encoder=True,
            divide_loss_by_length=True,
            d_model=1024,
            d_embedding=512,
            grad_clip_threshold=5.0,
            init_scale=None,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_latent_vectors=1,
            tie_embedding_weights=True,
            tie_logit_weights=True,
            transformer_encoder=False,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=2,
            val_check_interval=0.25
        )
    },
    'dense-benchmark': {
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_latent_vectors=1,
            num_layers=3,
            sparse_self_attention=False,
            tie_embedding_weights=True,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'sparse-benchmark': {
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_latent_vectors=1,
            num_layers=3,
            sparse_self_attention=True,
            tie_embedding_weights=True,
            warmup_steps=500
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'wikipedia': {
        'data': dict(
            chunking_strategy='none',
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=62_500,
            # Stub articles (< 160 tokens) make up nearly 1/4 of the dataset and don't help
            # the model learn long range dependencies. This way we force the model to get used
            # to not having the whole document in its sliding window attention window
            min_tokens_per_sample=160,
            max_tokens_per_sample=12_500
        ),
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_checkpointing=False,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=128,
            lr=3e-4,
            # lr_decay_steps=1_000_000,
            num_latent_vectors=1,
            num_layers=6,
            sparse_self_attention=True,
            tie_embedding_weights=True,
            warmup_steps=1000
        ),
        'trainer': dict(
            accumulate_grad_batches=2,
            val_check_interval=0.1
        )
    },
}
