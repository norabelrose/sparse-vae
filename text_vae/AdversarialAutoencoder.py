from .core import AutoregressiveGaussian, Transformer
from .FunnelAutoencoder import *
from dataclasses import dataclass
from math import log
from torch.distributions import Categorical
import gc
import torch


@dataclass
class AdversarialAutoencoderHparams(FunnelAutoencoderHparams):
    funnel: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(4, 4),
        scaling_factors=(2,)
    )
    latent_depth: int = 128
    max_seq_length: int = 512

    adam_beta1: float = 0.5
    grad_acc_steps: int = 1
    warmup_steps: int = 0

    # Have the generator try to minimize the absolute value of the (normalized) negative critic loss, instead of the
    # negative critic loss itself. This way the generator isn't rewarded (and in fact is punished) for making the critic
    # perform *worse* than chance. This seems to greatly stabilize training and prevent the critic from collapsing.
    abs_trick: bool = False

    # Hyperparameter that balances the adversarial loss against the reconstruction loss
    adv_weight: float = 10.0

    # Whether to constrain the generator output and the samples from the prior to be within the unit sphere using a
    # tanh activation function
    constrain_to_unit_sphere: bool = False
    critic_layers: int = 2
    critic_lr_multiplier: float = 1.0  # Critic LR = Generator LR * multiplier
    critic_steps_per_gen_step: int = 2
    critic_weight_clip_threshold: Optional[float] = None
    wasserstein: bool = False

@dataclass
class AdversarialAutoencoderState:
    ground_truth: Optional[Tensor] = None
    decoder_input: Optional[Tensor] = None
    encoder_states: List[Tensor] = field(default_factory=list)
    p_of_x_given_z: Optional[Categorical] = None

class AdversarialAutoencoder(FunnelAutoencoder):
    def __init__(self, hparams: DictConfig):
        super(AdversarialAutoencoder, self).__init__(hparams)

        # We need to be able to do multiple backward passes within the same training_step() invocation
        self.automatic_optimization = False
        self.skip_validation = False
        self.training_status = 'critic'

        encoder_hparams = hparams.funnel
        self.num_scales = len(encoder_hparams.block_sizes)
        self.critic = Critic(
            encoder_hparams=encoder_hparams,
            layers_per_scale=hparams.critic_layers,
            num_scales=self.num_scales,
            latent_depth=hparams.latent_depth,
            wasserstein=hparams.wasserstein
        )

        overt_depth = encoder_hparams.d_model
        latent_depth = hparams.latent_depth

        self.latent_downsamplers = nn.ModuleList([
            nn.Linear(overt_depth, latent_depth)
            for _ in range(self.num_scales)
        ])
        self.latent_upsamplers = nn.ModuleList([
            nn.Linear(latent_depth, overt_depth)
            for _ in range(self.num_scales)
        ])
        self.generator = Generator(latent_depth=latent_depth, encoder=self.encoder)

    def configure_optimizers(self, **kwargs):
        discriminator_params = set(self.critic.parameters())
        generator_params = filter(lambda x: x not in discriminator_params, self.parameters())

        critic_opts, critic_schedules = super().configure_optimizers(
            lr=self.hparams.lr * self.hparams.critic_lr_multiplier,
            params=self.critic.parameters()
        )
        gen_opts, gen_schedules = super().configure_optimizers(params=generator_params)

        return critic_opts + gen_opts, critic_schedules + gen_schedules

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.reconstruct(batch, compute_loss=False)['logits']

    def fake_latents_like(self, latents: List[Tensor]) -> List[Tensor]:
        return self.fake_latent_sample(list(latents[0].shape))

    def fake_latent_sample(self, shape: List[int]) -> List[Tensor]:
        fake_latents = self.generator(shape[0], shape[1])
        return fake_latents if not self.hparams.constrain_to_unit_sphere else [z.tanh() for z in fake_latents]

    def get_latents(self, x: Tensor) -> List[Tensor]:
        self.encoder.attention_state.configure_for_input(x.shape[-1], padding_mask=None)
        encoder_output = self.encoder(x)

        latents = [downsampler(z) for downsampler, z in zip(
            self.latent_downsamplers,
            reversed(encoder_output.hidden_states)
        )]
        return latents if not self.hparams.constrain_to_unit_sphere else [z.tanh() for z in latents]

    def get_reconstruction(self, latents: List[Tensor]) -> Categorical:
        # Now run the forward pass through the decoder
        big_latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]

        ae_state = AdversarialAutoencoderState()
        ae_state.decoder_input = big_latents[0]
        ae_state.encoder_states = big_latents[1:]
        ae_state = self.decoder_forward(ae_state)

        return ae_state.p_of_x_given_z

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        pbar_dict = super(AdversarialAutoencoder, self).get_progress_bar_dict()

        if self.hparams.critic_steps_per_gen_step:
            pbar_dict.pop('v_num')
            pbar_dict['training'] = self.training_status

        return pbar_dict

    def decoder_forward(self, vae_state: AdversarialAutoencoderState, padding_mask: Tensor = None, **kwargs):
        attn_state = self.decoder.attention_state
        attn_state.upsampling = True

        coroutine = self.decoder.forward_coroutine(
            vae_state.decoder_input,
            padding_mask=padding_mask,
            cross_attn_iter=kwargs.get('cross_attn_target')
        )
        encoder_state_iter = iter(vae_state.encoder_states)

        for block_idx, decoder_state in coroutine:
            # Final output
            if isinstance(decoder_state, FunnelTransformerOutput):
                vae_state.decoder_output = decoder_state
                raw_output = decoder_state.final_state
                vae_state.p_of_x_given_z = self.output_layer(raw_output)

                return vae_state

            if encoder_state_iter:
                encoder_state = next(encoder_state_iter, None)
                if encoder_state is None:  # We ran out of encoder states to use
                    continue
            else:
                encoder_state = None

            dec_state = self.decoder_block_end(vae_state, decoder_state, encoder_state, block_idx, **kwargs)
            if dec_state is not None:
                coroutine.send(dec_state)

            # Abort the forward pass. Used when gathering latent codes for training the prior model. After we get
            # the bottom latent feature map we don't actually need to run the last few layers.
            else:
                return vae_state

    # This method is reused for both training and inference. The compute_loss parameter is only used when
    # self.training == False; during training the loss is always computed.
    def reconstruct(self, batch: Dict[str, Tensor], compute_loss: bool = True) -> Dict[str, Any]:
        x = batch['token_ids']
        latents = self.get_latents(x)

        return_dict = {}
        if compute_loss:
            prior_z = self.fake_latents_like(latents[0])
            return_dict['adv_loss'] = -self.critic.loss(latents[0], prior_z)

            if not self.critic.wasserstein:
                return_dict['adv_loss'] += log(2.0)

        reconstruction = self.get_reconstruction(latents)
        return_dict['logits'] = reconstruction.logits

        if compute_loss:
            if (target := batch.get('labels')) is None:
                target = batch['token_ids']

            # We DON'T ignore padding tokens, so that the decoder can *generate* padding tokens when sampling
            return_dict['nll'] = -reconstruction.log_prob(target).mean()

        return return_dict

    def decoder_block_end(self, vae_state: Any, dec_state: Tensor, enc_state: Tensor, block_idx: int, **kwargs):
        return enc_state + dec_state

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int, optimizer_idx: int) -> Dict[str, Any]:  # noqa
        inner_loop_len = self.hparams.grad_acc_steps
        critic_step_ratio = self.hparams.critic_steps_per_gen_step
        outer_loop_len = (critic_step_ratio or 0) + 1

        # The outer loop is the alternation between training the critic and training the generator, and the inner
        # loop is the gradient accumulation loop.
        critic_opt, gen_opt = self.optimizers()
        batch_idx %= outer_loop_len * inner_loop_len
        outer_idx, inner_idx = divmod(batch_idx, inner_loop_len)

        # When critic_steps_per_gen_step is None or 0, then we just train both the critic and the generator
        # at each training batch. But when it's greater than 0, we always train the critic first, then train the
        # generator as the last step in our outer training loop.
        train_critic = not critic_step_ratio or outer_idx < outer_loop_len - 1
        train_generator = not critic_step_ratio or outer_idx == outer_loop_len - 1

        x = batch['token_ids']
        latents = self.get_latents(x)

        fake_z = self.fake_latents_like(latents)
        critic_loss = self.critic.loss(latents, fake_z)
        encoder_loss = -critic_loss

        # When the discriminator is utterly confused, it will output roughly 0.5 for all inputs. This will
        # result in a BCE loss of ln(0.5) = -0.6931471805599. We normalize the adversarial loss term so that
        # it is 0 when the discriminator's BCE loss is equal to this value.
        if not self.critic.wasserstein:
            encoder_loss += log(2.0)

        log_dict = {}
        pbar_dict = {'adv_loss': encoder_loss}
        return_dict = {'loss': encoder_loss, 'log': log_dict, 'progress_bar': pbar_dict}

        if train_critic:
            self.training_status = f'critic ({outer_idx + 1} of {outer_loop_len - 1})'

            # Compute discriminator loss w/ encoder.requires_grad == False
            with critic_opt.toggle_model():
                if self.critic.wasserstein and not self.hparams.critic_weight_clip_threshold:
                    grad_penalty = self.critic.compute_gradient_penalty(latents, fake_z)
                    log_dict['critic_grad_penalty'] = grad_penalty
                    self.manual_backward(grad_penalty)

                # We may want to retain the graph here to reuse it for the generator
                self.manual_backward(critic_loss, retain_graph=train_generator)  # noqa

        if train_generator:
            self.training_status = 'gen'

            # Check the actual moments of the latent distribution to see how well the discriminator is doing.
            all_latents = torch.cat([z.flatten() for z in latents])
            z_var, z_mean = torch.var_mean(all_latents)
            normalized = (all_latents - z_mean) / z_var.sqrt()
            z_skew = (normalized ** 3.0).mean()
            z_kurtosis = (normalized ** 4.0).mean()

            with gen_opt.toggle_model():
                reconstruction = self.get_reconstruction(latents)
                target = x if (labels := batch.get('labels')) is None else labels
                nll = -reconstruction.log_prob(target).mean()

                pbar_dict['nll'] = nll
                return_dict['logits'] = reconstruction.logits

                if self.hparams.abs_trick:
                    encoder_loss = encoder_loss.abs()

                self.manual_backward(nll + encoder_loss * self.hparams.adv_weight)  # noqa

            log_dict.update(pbar_dict)
            log_dict.update(z_var=z_var, z_mean=z_mean, z_skew=z_skew, z_kurtosis=z_kurtosis)

        if inner_idx == inner_loop_len - 1:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e3)

            if train_critic:
                critic_opt.step(); critic_opt.zero_grad()
                if threshold := self.hparams.critic_weight_clip_threshold:
                    self.critic.clip_weights(max_weight_magnitude=threshold)

            if train_generator:
                gen_opt.step(); gen_opt.zero_grad()

        return return_dict

    def on_validation_start(self):
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        if self.skip_validation:
            return None

        try:
            result = self.reconstruct(batch)
        except RuntimeError as ex:
            if 'CUDA out of memory' not in str(ex):
                raise

            self.print("Warning: Validation caused an OOM error. Skipping validation.")
            self.skip_validation = True
            return

        enc_loss, nll = result['adv_loss'], result['nll']

        result.update(log={'val_nll': nll, 'val_adv_loss': enc_loss})
        result.update(progress_bar={'nll': result['nll'], 'adv_loss': result['nll']})
        return result

    def markov_chain_sample(self, max_length: int = 40, num_samples: int = 1, k: int = 100, num_iter: int = 100,
                            masked_tokens_per_iter: int = 1, anneal_temperature: bool = False):
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.funnel
        stride = prod(funnel_hparams.scaling_factors)
        length = max_length // stride

        self.decoder.attention_state.configure_for_input(
            seq_len=length * stride,  # May not be the same as max_length if it doesn't divide evenly
            padding_mask=None,
            upsampling=True
        )

        depth = self.hparams.latent_depth
        latents = self.fake_latent_sample([num_samples, length, depth])
        latents = [upsampler(z) for upsampler, z in zip(self.latent_upsamplers, latents)]

        logits = self.output_layer(self.decoder(latents[0], cross_attn_iter=latents[1:]).final_state).logits
        tokens = logits.argmax(dim=-1)
        yield tokens

        mask_token = self.tokenizer.get_vocab()['[MASK]']
        temperature = 1.0

        for i in range(num_iter):
            indices_to_mask = torch.randint(max_length, [num_samples * masked_tokens_per_iter], device=tokens.device)
            tokens.index_fill_(-1, indices_to_mask, mask_token)

            batch = {'token_ids': tokens, 'padding_mask': None}
            logits, ids = self.reconstruct(batch, compute_loss=False)['logits'].topk(k)
            probs = logits.div(temperature).softmax(dim=-1)

            id_indices = probs.view(-1, k).multinomial(num_samples=1).view(num_samples, max_length, 1)
            tokens = ids.gather(dim=-1, index=id_indices).squeeze(-1)

            # Linear temperature annealing schedule
            if anneal_temperature:
                temperature = 1 - (i + 1) / num_iter

            yield tokens

    def sample(self, max_length: int, count: int = 1, **kwargs):
        return list(self.markov_chain_sample(max_length, num_samples=count))

    def compute_latents(self, batch: Dict[str, Any]) -> Any:
        pass


class Generator(nn.Module):
    def __init__(self, latent_depth: int, encoder: FunnelTransformer):
        super(Generator, self).__init__()

        funnel_hparams = deepcopy(encoder.hparams)
        d_model = latent_depth * 4
        blocks = funnel_hparams.block_sizes

        funnel_hparams.d_model = d_model
        funnel_hparams.scaling_factors = funnel_hparams.scaling_factors[::-1]
        funnel_hparams.upsampling = True

        self.base_dist = AutoregressiveGaussian(d_model)
        self.downsamplers = nn.ModuleList([
            nn.Linear(d_model, latent_depth)
            for _ in range(len(blocks))
        ])

        funnel = FunnelTransformer(funnel_hparams)
        funnel.attention_state = encoder.attention_state
        self.funnel = funnel

    def forward(self, batch: int, seq_len: int) -> List[Tensor]:
        z = self.base_dist(batch, seq_len)

        attn_state = self.funnel.attention_state
        attn_state.upsampling = True

        latents = self.funnel(z).hidden_states
        attn_state.rewind()
        return [downsampler(z) for downsampler, z in zip(self.downsamplers, latents)]


class Critic(nn.Module):
    def __init__(self, encoder_hparams: DictConfig, layers_per_scale: int, num_scales: int, latent_depth: int,
                 wasserstein: bool = False, grad_penalty_weight: float = 0.1):
        super(Critic, self).__init__()

        d_model = latent_depth * 2
        self.dropout = nn.Dropout(p=0.5)
        self.transformer = Transformer(
            num_layers=layers_per_scale * num_scales,
            d_model=d_model,
            num_heads=encoder_hparams.num_heads
        )
        self.upsamplers = nn.ModuleList([
            nn.Linear(latent_depth, d_model)
            for _ in range(num_scales)
        ])
        for i in range(num_scales - 1):
            self.transformer.layers[(i + 1) * layers_per_scale].use_cross_attention = True

        pred_linear = nn.Linear(d_model, 1)
        self.prediction_head = nn.Sequential(
            LambdaLayer(lambda output: output[..., 0, :]),  # Get [CLS] token
            pred_linear  # Get binary logits
        )

        self.wasserstein = wasserstein
        self.grad_penalty_weight = grad_penalty_weight

        # Weight initialization from DCGAN paper- this is important for training stability!
        for param in self.parameters():
            param.data.normal_(0, 0.02)

    def forward(self, all_z: List[Tensor]) -> Tensor:
        all_z = [self.dropout(upsampler(z)) for upsampler, z in zip(self.upsamplers, all_z)]

        x = self.transformer(all_z[0], cross_attn_target=all_z[1:])
        x = self.prediction_head(x)

        # With a Wasserstein loss, the output of the critic is interpreted as being a distance and
        # therefore should be constrained to be non-negative. We use a leaky ReLU here though, following ARAE,
        # to make sure gradients always flow through and the critic doesn't get stuck in the degenerate state of
        # always returning 0.0 for everything.
        if self.wasserstein:
            x = F.leaky_relu(x, negative_slope=0.2)

        return x

    def loss(self, real_zs: List[Tensor], fake_zs: List[Tensor]) -> Tensor:
        inputs = [torch.cat([real, fake], dim=0) for real, fake in zip(real_zs, fake_zs)]
        predictions = self.forward(inputs)

        # Using a Wasserstein optimal transport loss- maximize the distance assigned to the samples from the generator
        # and minimize the distance assigned to samples from the prior. In general *this loss should be negative*;
        # if it's positive then the critic is seriously misaligned.
        if self.wasserstein:
            predictions = predictions.view(2, -1)
            prior_distance = predictions[0].mean()
            posterior_distance = predictions[1].mean()
            return -(posterior_distance - prior_distance)

        # Using a standard GAN binary cross entropy loss
        batch_size, device = real_zs[0].shape[0], real_zs[0].device
        labels = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            torch.ones(batch_size, 1, device=device)
        ], dim=0)
        return F.binary_cross_entropy_with_logits(predictions, labels)

    def clip_weights(self, max_weight_magnitude: float = 0.01):
        for param in self.parameters():
            param.data.clamp_(-max_weight_magnitude, max_weight_magnitude)

    # Used with the Wasserstein optimal transport loss. Computes Monte Carlo estimate of the expected value of the norm
    # of the derivative of the critic's output wrt *how fake its input is*, for all degrees of fakeness.
    def compute_gradient_penalty(self, real_zs: List[Tensor], fake_zs: List[Tensor]) -> Tensor:
        # The alphas are randomly sampled from Unif[0, 1] and determine how much weight we place on the real latent,
        # as opposed to the fake one, for each sample in the batch.
        base_z = real_zs[0]
        alpha = torch.rand(base_z.shape[0], *([1] * (base_z.ndim - 1)), device=base_z.device, requires_grad=True)
        interpolations = [alpha * real.detach() + ((1 - alpha) * fake.detach()) for real, fake in zip(real_zs, fake_zs)]
        outputs = self.forward(interpolations)

        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=interpolations,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True  # So we can call .backward() on the penalty
        )
        gradients = torch.cat(gradients, dim=1)
        gradients = gradients.view(gradients.size(0), -1)

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.grad_penalty_weight
