from .core import Transformer
from .HierarchicalAutoencoder import *
from contextlib import contextmanager
from dataclasses import dataclass
from math import log
import torch


@dataclass
class AdversarialAutoencoderHparams(FunnelAutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(3, 3),
        scaling_factors=(2,),
        inject_absolute_pos_encodings=True
    )
    latent_depth: int = 128
    adam_beta1: float = 0.5
    warmup_steps: int = 200

    # Hyperparameter that balances the adversarial loss against the reconstruction loss
    adv_weight: float = 10.0

class AdversarialAutoencoder(HierarchicalAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(AdversarialAutoencoder, self).__init__(hparams)

        # We need to be able to use two optimizers within the same training_step() invocation
        self.automatic_optimization = False

        encoder_hparams = hparams.encoder
        num_scales = 1  # len(encoder_hparams.block_sizes)
        self.discriminator = Discriminator(
            encoder_hparams=encoder_hparams,
            num_scales=num_scales,
            latent_depth=hparams.latent_depth
        )

        overt_depth = encoder_hparams.d_model
        self.latent_downsamplers = nn.ModuleList([
            nn.Linear(overt_depth, hparams.latent_depth)
            for _ in range(num_scales)
        ])
        self.latent_upsamplers = nn.ModuleList([
            nn.Linear(hparams.latent_depth, overt_depth)
            for _ in range(num_scales)
        ])

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.reconstruct(batch, compute_loss=False)['logits']

    @contextmanager
    def toggle_discriminator(self):
        self.discriminator.requires_grad_(True)
        self.generator_requires_grad_(False)
        yield
        self.requires_grad_(True)

    @contextmanager
    def toggle_generator(self):
        self.generator_requires_grad_(True)
        self.discriminator.requires_grad_(False)
        yield
        self.requires_grad_(True)

    def generator_requires_grad_(self, value: bool):
        discriminator_params = set(self.discriminator.parameters())
        generator_params = filter(lambda x: x not in discriminator_params, self.parameters())

        for param in generator_params:
            param.requires_grad = value

    # This method is reused for both training and inference. The compute_loss parameter is only used when
    # self.training == False; during training the loss is always computed.
    def reconstruct(self, batch: Dict[str, Tensor], compute_loss: bool = True, val: bool = False) -> Dict[str, Any]:
        encoder_output = self.encoder_forward(batch)
        latents = [downsampler(z) for downsampler, z in zip(
            self.latent_downsamplers,
            reversed(encoder_output.hidden_states)
        )]

        discr_loss, enc_loss, nll = 0.0, 0.0, 0.0
        log_dict = {}
        if compute_loss:
            fake_latents = [torch.randn_like(z) for z in latents]
            discr_input = torch.cat([latents[0], fake_latents[0]], dim=0)
            # discr_input2 = [torch.cat([z, fake]) for z, fake in zip(latents[1:], fake_latents[1:])]

            # We just batch the fake and real latents together
            batch_size = discr_input.shape[0] // 2
            discr_labels = torch.cat([
                torch.ones(batch_size, 1, device=self.device),
                torch.zeros(batch_size, 1, device=self.device)
            ], dim=0)

            # Compute discriminator loss w/ encoder.requires_grad == False
            with self.toggle_discriminator():
                discr_pred = self.discriminator(discr_input)
                discr_loss = F.binary_cross_entropy_with_logits(discr_pred, discr_labels)
                if not val:
                    # RETAIN THE GRAPH to reuse it for generator
                    self.manual_backward(discr_loss, retain_graph=True)  # noqa

        # Now run the forward pass through the decoder
        self.decoder.attention_state.upsampling = True
        big_latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]

        ae_state = HierarchicalAutoencoderState()
        ae_state.decoder_input = big_latents[0]
        # ae_state.encoder_states = big_latents[1:]
        ae_state = self.decoder_forward(ae_state, padding_mask=batch['padding_mask'])

        if (target := batch.get('labels')) is None:
            target = batch['token_ids']

        if compute_loss:
            nll = -ae_state.p_of_x_given_z.log_prob(target)  # Ignore padding; P(padding) = 1.0
            nll = nll.where(target != 0, nll.new_zeros([])).mean()
            # nll = F.cross_entropy(logits.flatten(end_dim=-2), target.flatten(), ignore_index=0)

            # When the discriminator is utterly confused, it will output roughly 0.5 for all inputs. This will
            # result in a BCE loss of ln(0.5) = -0.6931471805599. We normalize the adversarial loss term so that
            # it is 0 when the discriminator's BCE loss is equal to this value.
            enc_loss = -(log(0.5) + discr_loss)

        pbar_dict = {'nll': nll, 'adv_loss': enc_loss}
        if compute_loss and not val:
            # Check the actual moments of the latent distribution to see how well the discriminator is doing.
            all_latents = torch.cat([z.flatten() for z in latents])
            z_var, z_mean = torch.var_mean(all_latents)
            normalized = (all_latents - z_mean) / z_var.sqrt()
            z_skew = (normalized ** 3.0).mean()
            z_kurtosis = (normalized ** 4.0).mean()

            with self.toggle_generator():
                self.manual_backward(nll + enc_loss.abs() * self.hparams.adv_weight)  # noqa

            grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e3)
            log_dict['grad_norm'] = grad_norm

            self.optimizers().step()
            self.zero_grad()

            log_dict.update(z_var=z_var, z_mean=z_mean, z_skew=z_skew, z_kurtosis=z_kurtosis)
            log_dict.update(pbar_dict)
        else:
            log_dict.update(val_nll=nll, val_adv_loss=enc_loss)

        return {'logits': ae_state.p_of_x_given_z.logits, 'log': log_dict, 'loss': nll, 'progress_bar': pbar_dict}

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        return enc_state + dec_state

    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Any]:  # noqa
        return self.reconstruct(batch)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        self.reconstruct(batch, val=True)

    def markov_chain_sample(self, max_length: int, num_samples: int, num_iter: int, mask_prob: float, k: int,
                            anneal_temperature: bool = False):
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
        latents = [torch.randn(num_samples, length, depth, device=self.device) for length in lengths]
        latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]

        logits = self.output_layer(self.decoder(latents[0], cross_attn_target=latents[1:]).final_state)
        tokens = self.decode_logits(logits)

        mask_probs = torch.tensor([[mask_prob]], device=self.device)
        mask_probs = mask_probs.expand_as(tokens)
        mask_token = self.tokenizer.get_vocab()['[MASK]']

        for _ in range(num_iter - 1):
            mlm_mask = mask_probs.bernoulli().bool()
            tokens[mlm_mask] = mask_token

            batch = {'token_ids': tokens, 'padding_mask': None}
            tokens = self.decode_logits(self.reconstruct(batch, compute_loss=False)['logits'])

        return tokens

    def sample(self, max_length: int, count: int = 1, **kwargs):
        return self.markov_chain_sample(max_length, num_samples=count, num_iter=2, mask_prob=0.15, k=10)

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass

class Discriminator(nn.Module):
    def __init__(self, encoder_hparams: DictConfig, num_scales: int, latent_depth: int):
        super(Discriminator, self).__init__()

        d_model = encoder_hparams.d_model
        # self.dropout = nn.Dropout(p=0.5)
        self.transformer = Transformer(num_layers=1, d_model=d_model, num_heads=encoder_hparams.num_heads)
        self.upsamplers = nn.ModuleList([
            nn.Linear(latent_depth, d_model)
            for _ in range(num_scales)
        ])
        self.prediction_head = nn.Sequential(
            LambdaLayer(lambda output: output[..., 0, :]),  # Get [CLS] token
            nn.Linear(d_model, 1)  # Get binary logits
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.dropout(x)
        x = self.upsamplers[0](x)
        x = self.transformer(x)
        return self.prediction_head(x)
