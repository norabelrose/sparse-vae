hparam_presets = {
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
    'nonvae-wikipedia': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=160,
            max_tokens_per_sample=25_000
        ),
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            lr=3e-4,
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
    'wikipedia': {
        'data': dict(
            dataset_name='wikipedia',
            dataset_config='20200501.en',
            tokens_per_batch=50_000,
            min_tokens_per_sample=512,
            max_tokens_per_sample=25_000
        ),
        'model': dict(
            divide_loss_by_length=True,
            d_model=512,
            grad_checkpointing=True,
            grad_clip_threshold=5.0,
            init_scale=0.02,
            kl_weight_start=0.3,
            kl_annealing_steps=4000,
            latent_depth=64,
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
