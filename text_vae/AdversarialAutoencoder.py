from .HierarchicalAutoencoder import *
from dataclasses import dataclass
from math import log
from numpy import prod
import torch


@dataclass
class AdversarialAutoencoderHparams(FunnelAutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(3, 3),
        scaling_factors=(2,)
    )
    discriminator: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=256,
        num_heads=4,
        block_sizes=(3, 3),
        scaling_factors=(1,),
        use_embedding=False
    )
    latent_depth: int = 16

    # Hyperparameter that balances the adversarial loss against the reconstruction loss
    adv_weight: float = 0.5

class AdversarialAutoencoder(FunnelAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(AdversarialAutoencoder, self).__init__(hparams)

        # We need to be able to use two optimizers within the same training_step() invocation
        self.automatic_optimization = False

        discr_hparams = hparams.discriminator
        discr_funnel = FunnelTransformer(discr_hparams)

        self.discriminator = nn.Sequential(
            nn.Linear(hparams.latent_depth, discr_hparams.d_model),
            discr_funnel,
            LambdaLayer(lambda output: output.final_state[..., 0, :]),  # Get [CLS] token
            nn.Linear(discr_hparams.d_model, 1)  # Get binary logits
        )

        # We need a reference to the LR schedulers so we can update them with the most recent adversarial loss
        # self.discr_lr_adjuster = AdversarialLRAdjuster(lr_lambda)
        # self.nondiscr_lr_adjuster = AdversarialLRAdjuster(lr_lambda, for_discriminator=False)

        overt_depth = hparams.encoder.d_model
        self.latent_downsampler = nn.Linear(overt_depth, hparams.latent_depth)
        self.latent_upsampler = nn.Linear(hparams.latent_depth, overt_depth)

    def configure_optimizers(self):
        discr_params = set(self.discriminator.parameters())
        nondiscr_params = [param for param in self.parameters() if param not in discr_params]

        base_lr = self.hparams.lr
        adam_discr = AdamW(list(discr_params), lr=base_lr * 8.0, weight_decay=self.hparams.weight_decay)
        adam_nondiscr = AdamW(nondiscr_params, lr=base_lr, weight_decay=self.hparams.weight_decay)

        lr_lambda = get_cosine_decay_with_warmup_schedule(self.hparams.lr_decay_steps, self.hparams.warmup_steps)
        scheduler_discr = LambdaLR(adam_discr, lr_lambda=lr_lambda)
        scheduler_nondiscr = LambdaLR(adam_nondiscr, lr_lambda=lr_lambda)

        return [adam_nondiscr, adam_discr], [{'scheduler': scheduler_nondiscr, 'interval': 'step'},
                                             {'scheduler': scheduler_discr, 'interval': 'step'}]

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, optimizer_idx: int) -> Dict[str, Any]:  # noqa
        nondiscr_opt, discr_opt = self.optimizers()
        nondiscr_opt.zero_grad(); discr_opt.zero_grad()

        encoder_output = self.encoder_forward(batch).final_state

        latent = self.latent_downsampler(encoder_output)
        fake_latent = torch.randn_like(latent)  # Standard Gaussian prior over the latents

        # We just batch the fake and real latents together
        batch_size = latent.shape[0]
        discr_input = torch.cat([latent.detach(), fake_latent], dim=0)
        discr_labels = torch.cat([
            torch.ones(batch_size, 1, device=latent.device),
            torch.zeros(batch_size, 1, device=latent.device)
        ], dim=0)

        # Update discriminator
        discr_pred = self.discriminator(discr_input)
        discr_loss = F.binary_cross_entropy_with_logits(discr_pred, discr_labels)
        self.manual_backward(discr_loss, discr_opt)
        discr_opt.step()

        # Compute loss for "generator"/encoder
        discr_input = torch.cat([latent, fake_latent], dim=0)
        discr_pred = self.discriminator(discr_input)

        # When the discriminator is utterly confused, it will output roughly 0.5 for all inputs. This will result in a
        # BCE loss of ln(0.5) = -0.6931471805599453. We normalize the adversarial loss term so that it is 0 when the
        # discriminator's BCE loss is equal to this value.
        enc_loss = -(log(0.5) + F.binary_cross_entropy_with_logits(discr_pred, discr_labels))

        # adjuster_loss = enc_loss.detach() if self.trainer.current_epoch < 2 else 0.5
        # self.discr_lr_adjuster.adv_loss = adjuster_loss
        # self.nondiscr_lr_adjuster.adv_loss = adjuster_loss

        # Now run the forward pass through the decoder
        self.decoder.attention_state.upsampling = True
        big_latents = self.latent_upsampler(latent)
        logits = self.output_layer(self.decoder(big_latents).final_state)

        if (target := batch.get('labels')) is None:
            target = batch['token_ids']

        # Check the actual moments of the latent distribution to see how well the discriminator is doing. Empirically
        # it seems that the mean, skew, and kurtosis of the latent distribution will stay very close to their standard
        # Gaussian values even without an adversarial loss term, but the variance tends to stick around the 3-6 range,
        # and the discriminator has a very hard time noticing this. It turns out that simply tacking on the absolute
        # value of the log variance of the latents onto the loss fixes this problem quite nicely.
        z_var, z_mean = torch.var_mean(latent)
        normalized = (latent - z_mean) / z_var.sqrt()
        z_skew = (normalized ** 3.0).mean()
        z_kurtosis = (normalized ** 4.0).mean()

        nll = F.cross_entropy(logits.flatten(end_dim=-2), target.flatten(), ignore_index=0)
        self.manual_backward(nll + enc_loss * self.hparams.adv_weight + z_var.log().abs(), nondiscr_opt)
        nondiscr_opt.step()  # Update the encoder & decoder weights

        self.log_dict({'z_var': z_var, 'z_mean': z_mean, 'z_skew': z_skew, 'z_kurtosis': z_kurtosis},
                      on_step=True, on_epoch=False)
        self.log_dict({'nll': nll, 'adv_loss': enc_loss}, on_step=True, on_epoch=False, prog_bar=True)
        return {'logits': logits}

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        encoder_output = self.encoder_forward(batch).final_state

        latent = self.latent_downsampler(encoder_output)
        fake_latent = torch.randn_like(latent)  # Standard Gaussian prior over the latents

        batch_size = latent.shape[0]
        discr_input = torch.cat([latent, fake_latent], dim=0)
        discr_labels = torch.cat([
            torch.ones(batch_size, 1, device=latent.device),
            torch.zeros(batch_size, 1, device=latent.device)
        ], dim=0)

        discr_pred = self.discriminator(discr_input)
        enc_loss = -(log(0.5) + F.binary_cross_entropy_with_logits(discr_pred, discr_labels))

        self.decoder.attention_state.upsampling = True
        big_latents = self.latent_upsampler(latent)
        logits = self.output_layer(self.decoder(big_latents).final_state)

        if (target := batch.get('labels')) is None:
            target = batch['token_ids']

        nll = F.cross_entropy(logits.flatten(end_dim=-2), target.flatten(), ignore_index=0)
        self.log_dict({
            'val_nll': nll,
            'val_adv_loss': enc_loss
        })

    def sample(self, max_length: int, count: int = 1, **kwargs):
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder
        total_scaling = prod(funnel_hparams.scaling_factors)
        seed_length = max_length // total_scaling

        # (batch, seq_len, vocab_size)
        self.decoder.attention_state.configure_for_input(
            seq_len=total_scaling * seed_length,  # May not be the same as max_length if it doesn't divide evenly
            dtype=torch.long,
            device=self.device,
            padding_mask=None
        )

        latent = torch.randn([count, seed_length, self.hparams.latent_depth], device=self.device)
        big_latent = self.latent_upsampler(latent)
        logits = self.output_layer(self.decoder(big_latent).final_state)

        return self.decode_logits(logits)

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

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
            adv_multiplier = adv_multiplier.clamp(0.25, 1.0)
        else:
            adv_multiplier = max(0.25, min(adv_multiplier, 1.0))

        return raw_multiplier * adv_multiplier
