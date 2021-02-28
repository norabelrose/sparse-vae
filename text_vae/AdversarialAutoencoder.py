from .HierarchicalAutoencoder import *
from .core.Transformers import Transformer
from dataclasses import dataclass
from math import log
from numpy import cumsum
import torch


@dataclass
class AdversarialAutoencoderHparams(FunnelAutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(3, 3),
        scaling_factors=(4,)
    )
    latent_depth: int = 32

    # Hyperparameter that balances the adversarial loss against the reconstruction loss
    adv_weight: float = 1.0

class AdversarialAutoencoder(FunnelAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(AdversarialAutoencoder, self).__init__(hparams)

        # We need to be able to use two optimizers within the same training_step() invocation
        self.automatic_optimization = False

        lr_lambda = get_cosine_decay_with_warmup_schedule(hparams.lr_decay_steps, hparams.warmup_steps)
        self.discr_lr_adjuster = AdversarialLRAdjuster(lr_lambda)

        encoder_hparams = hparams.encoder
        block_ends = cumsum(encoder_hparams.block_sizes)
        num_scales = len(encoder_hparams.block_sizes)
        self.discriminator = Discriminator(
            num_layers=block_ends[-1],
            num_scales=num_scales,
            d_model=256,
            latent_depth=hparams.latent_depth
        )

        # for i in block_ends[:-1]:
        #     self.decoder.layers[i].use_cross_attention = True
        #     self.discriminator.transformer.layers[i].use_cross_attention = True

        overt_depth = encoder_hparams.d_model
        self.latent_downsamplers = nn.ModuleList([
            nn.Linear(overt_depth, hparams.latent_depth)
            for _ in range(num_scales)
        ])
        self.latent_upsamplers = nn.ModuleList([
            nn.Linear(hparams.latent_depth, overt_depth)
            for _ in range(num_scales)
        ])

    def configure_optimizers(self):
        discr_params = set(self.discriminator.parameters())
        nondiscr_params = [param for param in self.parameters() if param not in discr_params]

        base_lr = self.hparams.lr
        adam_discr = AdamW(list(discr_params), lr=base_lr * 4, weight_decay=self.hparams.weight_decay)
        adam_nondiscr = AdamW(nondiscr_params, lr=base_lr, weight_decay=self.hparams.weight_decay)

        scheduler_discr = LambdaLR(adam_discr, lr_lambda=self.discr_lr_adjuster)
        scheduler_nondiscr = LambdaLR(adam_nondiscr, lr_lambda=self.discr_lr_adjuster.raw_lambda)

        return [adam_nondiscr, adam_discr], [{'scheduler': scheduler_nondiscr, 'interval': 'step'},
                                             {'scheduler': scheduler_discr, 'interval': 'step'}]

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.reconstruct(batch, compute_loss=False)['logits']

    # This method is reused for both training and inference. The compute_loss parameter is only used when
    # self.training == False; during training the loss is always computed.
    def reconstruct(self, batch: Dict[str, Tensor], compute_loss: bool = None) -> Dict[str, Any]:
        nondiscr_opt, discr_opt = self.optimizers()
        if self.training:
            compute_loss = True

        encoder_output = self.encoder_forward(batch)
        latents = [downsampler(z) for downsampler, z in zip(
            self.latent_downsamplers,
            reversed(encoder_output.hidden_states)
        )]

        discr_loss, enc_loss, nll = 0.0, 0.0, 0.0
        log_dict = {}
        if compute_loss:
            fake_latents = [torch.randn_like(z) for z in latents]  # Standard Gaussian prior over the latents

            discr_input1 = torch.cat([latents[0], fake_latents[0]], dim=0)
            discr_input2 = [torch.cat([z, fake]) for z, fake in zip(latents[1:], fake_latents[1:])]

            # We just batch the fake and real latents together
            batch_size = discr_input1.shape[0] // 2
            discr_labels = torch.cat([
                torch.ones(batch_size, 1, device=self.device),
                torch.zeros(batch_size, 1, device=self.device)
            ], dim=0)

            # Compute discriminator loss w/ encoder.requires_grad == False
            with discr_opt.toggle_model():
                discr_pred = self.discriminator(discr_input1, cross_attn_target=discr_input2)
                discr_loss = F.binary_cross_entropy_with_logits(discr_pred, discr_labels)
                if self.training:
                    # RETAIN THE GRAPH to reuse it for generator
                    self.manual_backward(discr_loss, retain_graph=True)
                    log_dict['discr_grad_norm'] = nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                                                           max_norm=150.0)

        # Now run the forward pass through the decoder
        self.decoder.attention_state.upsampling = True
        big_latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]
        logits = self.output_layer(self.decoder(big_latents[0], padding_mask=batch['padding_mask'],
                                                cross_attn_target=big_latents[1:]).final_state)

        if (target := batch.get('labels')) is None:
            target = batch['token_ids']

        if compute_loss:
            nll = F.cross_entropy(logits.flatten(end_dim=-2), target.flatten(), ignore_index=0)

            # When the discriminator is utterly confused, it will output roughly 0.5 for all inputs. This will
            # result in a BCE loss of ln(0.5) = -0.6931471805599. We normalize the adversarial loss term so that
            # it is 0 when the discriminator's BCE loss is equal to this value.
            enc_loss = -(log(0.5) + discr_loss)

        pbar_dict = {'nll': nll, 'adv_loss': enc_loss}
        log_dict.update(pbar_dict)
        if self.training:
            # Check the actual moments of the latent distribution to see how well the discriminator is doing.
            # Empirically it seems that the mean, skew, and kurtosis of the latent distribution will stay very close to
            # their standard Gaussian values even without an adversarial loss term, but the variance tends to stick
            # around the 3-6 range, and the discriminator has a very hard time noticing this. It turns out that simply
            # tacking on the absolute value of the log variance of the latents onto the loss fixes this problem nicely.
            all_latents = torch.cat([z.flatten() for z in latents])
            z_var, z_mean = torch.var_mean(all_latents)
            normalized = (all_latents - z_mean) / z_var.sqrt()
            z_skew = (normalized ** 3.0).mean()
            z_kurtosis = (normalized ** 4.0).mean()

            # Compute generator/encoder loss w/ discriminator.requires_grad == False
            with nondiscr_opt.toggle_model():
                self.discr_lr_adjuster.adv_loss = enc_loss
                self.manual_backward(nll + enc_loss * self.hparams.adv_weight + z_var.log().abs())

                params = nondiscr_opt.param_groups[0]['params']
                log_dict['ae_grad_norm'] = nn.utils.clip_grad_norm_(params, max_norm=150.0)

                nondiscr_opt.step(); discr_opt.step()  # Update all weights
                self.zero_grad()

            log_dict.update(z_var=z_var, z_mean=z_mean, z_skew=z_skew, z_kurtosis=z_kurtosis)

        return {'logits': logits, 'log': log_dict, 'loss': nll, 'progress_bar': pbar_dict}

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, optimizer_idx: int) -> Dict[str, Any]:  # noqa
        return self.reconstruct(batch, compute_loss=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        self.reconstruct(batch, compute_loss=True)

    def sample(self, max_length: int, count: int = 1, **kwargs):
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder
        strides = cumprod(funnel_hparams.scaling_factors)
        lengths = max_length // strides[::-1]

        self.decoder.attention_state.configure_for_input(
            seq_len=lengths[0] * strides[0],  # May not be the same as max_length if it doesn't divide evenly
            dtype=torch.long,
            device=self.device,
            padding_mask=None
        )

        depth = self.hparams.latent_depth
        latents = [torch.randn([count, length, depth], device=self.device) for length in lengths]
        latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]

        logits = self.output_layer(self.decoder(latents[0], cross_attn_target=latents[1:]).final_state)
        tokens = self.decode_logits(logits)

        if num_iter := kwargs.get('num_iter'):
            for _ in range(num_iter - 1):
                batch = {'token_ids': tokens, 'padding_mask': None}
                tokens = self.decode_logits(self.forward(batch))

        return tokens

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

class Discriminator(nn.Module):
    def __init__(self, num_layers: int, num_scales: int, d_model: int, latent_depth: int):
        super(Discriminator, self).__init__()

        self.transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=d_model // 64)
        self.upsamplers = nn.ModuleList([
            nn.Linear(latent_depth, d_model)
            for _ in range(num_scales)
        ])
        self.prediction_head = nn.Sequential(
            LambdaLayer(lambda output: output[..., 0, :]),  # Get [CLS] token
            nn.Linear(d_model, 1)  # Get binary logits
        )

    def forward(self, x: Tensor, cross_attn_target: List[Tensor]) -> Tensor:
        x = self.upsamplers[0](x)

        if cross_attn_target:
            cross_attn_target = [upsampler(x) for upsampler, x in zip(self.upsamplers[1:], cross_attn_target)]  # noqa
        else:
            cross_attn_target = None

        return self.prediction_head(self.transformer(x, cross_attn_target=cross_attn_target))

@dataclass
class AdversarialLRAdjuster:
    raw_lambda: Callable
    adv_loss: Union[float, Tensor] = 0.0
    for_discriminator: bool = True

    def __call__(self, *args, **kwargs):
        raw_multiplier = self.raw_lambda(*args, **kwargs)
        adv_multiplier = log(2.0) - self.adv_loss if self.for_discriminator else self.adv_loss  # [0, 0.693]
        adv_multiplier /= log(2.0)  # [0, 1.0]

        if isinstance(adv_multiplier, Tensor):
            adv_multiplier = adv_multiplier.clamp(0.0, 1.0)
        else:
            adv_multiplier = max(0.0, min(adv_multiplier, 1.0))

        return raw_multiplier * adv_multiplier
