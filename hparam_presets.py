hparam_presets = {
    'lstm-benchmark': {
        'model': dict(
            bidirectional_encoder=True,
            d_model=1024,
            d_embedding=512,
            grad_clip_threshold=150.0,
            init_scale=None,
            kl_weight_start=0.2,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            tie_embedding_weights=True,
            tie_logit_weights=True,
            transformer_encoder=False
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'lstm-wikipedia': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=25_000
        ),
        'model': dict(
            bidirectional_encoder=True,
            d_model=2048,
            d_embedding=512,
            grad_clip_threshold=150.0,
            init_scale=None,
            kl_weight_start=1.0,
            kl_annealing_steps=0,
            latent_depth=64,
            lr=3e-4,
            tie_embedding_weights=True,
            tie_logit_weights=True,
            transformer_encoder=False
        ),
        'trainer': dict(
            accumulate_grad_batches=2,
            val_check_interval=0.25
        )
    },
    'dense-benchmark': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=3_125
        ),
        'model': dict(
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=150.0,
            init_scale=0.02,
            kl_weight_start=0.3,
            kl_weight_end=1.0,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_layers=6,
            sparse_self_attention=False,
            tie_embedding_weights=True
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'sparse-benchmark': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=3_125
        ),
        'model': dict(
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=150.0,
            init_scale=0.02,
            kl_weight_start=1.0,
            kl_annealing_steps=0,
            latent_depth=64,
            lr=3e-4,
            num_layers=6,
            sparse_self_attention=True,
            tie_embedding_weights=True
        ),
        'trainer': dict(
            accumulate_grad_batches=2
        )
    },
    'nonvae-wikipedia': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=3_125
        ),
        'model': dict(
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=150.0,
            init_scale=0.02,
            lr=3e-4,
            num_layers=6,
            sparse_self_attention=False,
            tie_embedding_weights=True
        ),
        'trainer': dict(
            accumulate_grad_batches=2,
            val_check_interval=0.1
        )
    },
    'wikipedia': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=25_000
        ),
        'model': dict(
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=150.0,
            init_scale=0.02,
            attn_window_size=12,
            kl_weight_start=0.2,
            kl_weight_end=1.0,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=3e-4,
            num_layers=6,
            sparse_self_attention=True,
            tie_embedding_weights=True
        ),
        'trainer': dict(
            accumulate_grad_batches=2,
            val_check_interval=0.1
        )
    },
    'pg19': {
        'data': dict(
            dataset_name='pg19',
            dataset_config=None,
            tokens_per_batch=55_296,
            min_tokens_per_sample=512,
            max_tokens_per_sample=55_296
        ),
        'model': dict(
            # adam_beta1=0.95,
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=150.0,
            init_scale=0.02,
            attn_window_size=16,
            kl_weight_start=0.3,
            kl_weight_end=1.0,
            kl_annealing_steps=8000,
            latent_depth=64,
            lr=1e-3,
            num_layers=6,
            sparse_self_attention=True,
            tie_embedding_weights=True
        ),
        'trainer': dict(
            accumulate_grad_batches=4,
            val_check_interval=0.5
        )
    },
}
