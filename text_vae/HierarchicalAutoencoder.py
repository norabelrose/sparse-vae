from collections import defaultdict, deque
from dataclasses import asdict
from numpy import prod
from .Autoencoder import *
from .FunnelTransformer import *
from .GenerationUtils import *
import math
import torch.nn.functional as F
import torch


@dataclass
class HierarchicalAutoencoderHparams(AutoencoderHparams):
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        block_sizes=(10, 10, 10, 4, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2, 4, 4),    # How much the hidden state is downsampled between each encoder block
        d_model=256,                     # This is small to save on parameters while having a deep model
        num_heads=4
    )
    # Indicates how many groups of latent variables should be used. If None, the number will be set equal to
    # the number of Transformer layers in the encoder.
    num_latent_groups: Optional[int] = None
    tie_embedding_weights: bool = True
    use_autoregressive_decoder: bool = False
    use_encoder_residual_connections: bool = True
    use_pretrained_encoder: bool = False

    include_padding_positions: bool = True


class HierarchicalAutoencoder(Autoencoder):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        encoder_hparams = hparams.encoder
        num_layers = sum(encoder_hparams.block_sizes)
        num_latent_groups = hparams.num_latent_groups

        if not num_latent_groups:
            num_latent_groups = hparams.num_latent_groups = num_layers
        else:
            assert 1 <= num_latent_groups <= num_layers

        # Evenly space the latent groups to the extent possible, biasing them toward higher resolutions
        stride, leftover = divmod(num_layers, num_latent_groups)
        self.latent_groups = list(range(0, num_layers - leftover, stride))

        # save_hyperparameters() stores the hparams in self.hparams and ensures they are saved to disk during training.
        self.save_hyperparameters(hparams)

        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(encoder_hparams)
        attn_state.shared = True
        decoder_hparams = mutate(
            encoder_hparams,
            block_sizes=encoder_hparams.block_sizes[::-1],  # Reverse the order of the blocks
            scaling_factors=encoder_hparams.scaling_factors[::-1],
            upsampling=True
        )
        self.encoder = FunnelTransformer(encoder_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)
        self.encoder.attention_state = attn_state
        self.decoder.attention_state = attn_state

        if hparams.tie_embedding_weights:
            self.decoder.output_layer[0].weight = self.encoder.input_layer[0].weight

        if hparams.use_autoregressive_decoder:
            hidden_size = hparams.encoder.d_model * 2

            # This is used to upsample one of the latent tensors to be the same dimensionality as the RNN hidden state
            # so we can use it as the *initial* hidden state of the RNN
            self.rnn_latent_upsample = nn.Linear(hparams.latent_depth, hidden_size)
            self.rnn_output_downsample = nn.Linear(hidden_size, hparams.encoder.d_model)
            self.rnn_decoder = nn.RNN(
                input_size=hparams.encoder.d_model + hparams.latent_depth,
                hidden_size=hidden_size,
                batch_first=True
            )

        # Whether we should feed selected encoder states into the decoder to help it along
        overt_depth, latent_depth = encoder_hparams.d_model, hparams.latent_depth
        if self.hparams.use_encoder_residual_connections:
            self.encoder.hparams.return_block_outputs = True
            posterior_input_depth = overt_depth * 2  # Concatenate the encoder and decoder states depthwise
        else:
            posterior_input_depth = overt_depth  # Just the decoder state

        def get_linear_with_gelu(input_dim, output_dim, zero_initialized: bool = False):
            linear = nn.Linear(input_dim, output_dim)
            if zero_initialized:
                linear.bias.data.zero_()
                linear.weight.data.zero_()

            return nn.Sequential(nn.GELU(), linear)

        def linears_with_gelu(input_dim, output_dim, layers: List[int], zero_initialized: bool = False):
            return nn.ModuleDict({
                str(layer_idx): get_linear_with_gelu(input_dim, output_dim, zero_initialized)
                for layer_idx in layers
            })

        self.distributions = nn.ModuleDict({
            # Output contains mu & log sigma. Note that initializing the weights of the prior to zero means that at
            # the start of training, it will output 0 mean, 0 log sigma (i.e. 1 sigma) independent of the input
            'p(z)': linears_with_gelu(latent_depth, latent_depth * 2, self.latent_groups[1:], zero_initialized=True),
            'q(z|x)': linears_with_gelu(posterior_input_depth, latent_depth * 2, self.latent_groups)
        })
        self.latent_upsample = linears_with_gelu(latent_depth, overt_depth, self.latent_groups)

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

    def forward(self, batch: Dict[str, Any], **kwargs) -> FunnelTransformerOutput:
        x = batch['token_ids']
        self.encoder.attention_state.configure_for_input(x.shape[1], x.dtype, x.device, batch['padding_mask'])
        return self.encoder(batch['token_ids'], padding_mask=batch['padding_mask'], **kwargs)

    def compute_loss_for_step(self, result: Dict[str, Tensor], step: str):
        # Figure out the correct denominator for the loss
        if self.hparams.include_padding_positions:
            num_tokens = result['logits'].shape[1]
        else:
            # Each sample in the batch will have a different number of nonpadding tokens
            num_tokens = self.decoder.attention_state.padding_mask.sum(dim=-1)

        # Normalize the KL, NLL, and p(x) and q(z|x) terms by dividing them by the number of tokens in the input.
        # Then take the average over all samples in the batch.
        for key, value in result.items():
            if key in ('kl', 'nll', 'p(z)', 'q(z|x)', 'loss'):
                result[key] = (value / num_tokens).mean()

        # When the KL weight is at its default 1.0 setting, this is equal to the negative ELBO (evidence lower bound)
        loss = self.hparams.kl_weight * result['kl'] + result['nll']
        self.log_dict({'kl': result['kl'], 'nll': result['nll'], step + '_loss': loss})
        return loss

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'train')

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_threshold)
        self.log('grad_norm', grad_norm, on_step=True)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        result = self.reconstruct(batch)
        self.log('iw_loss', self.get_loss_monte_carlo(batch))
        return self.compute_loss_for_step(result, 'val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        result = self.reconstruct(batch)
        return self.compute_loss_for_step(result, 'test')

    # Runs a forward pass through the encoder and the decoder and returns a dict with the sum total KL and NLL
    # for each sample in the batch. Note that these tensors must be normalized before computing the loss.
    def reconstruct(self, batch: Dict[str, Any], return_logits: bool = False, **kwargs) -> Dict[str, Tensor]:
        # Encoder forward pass
        is_autoregressive = self.hparams.use_autoregressive_decoder
        encoder_out = self.forward(batch, return_embedded_input=is_autoregressive, return_logits=not is_autoregressive)

        # Help out the decoder by giving it some of the states of the encoder
        if self.hparams.use_encoder_residual_connections:
            kwargs['encoder_states'] = encoder_out.hidden_states

        output = self.decoder_forward(
            encoder_out.final_state,
            return_latents=is_autoregressive,
            **kwargs
        )
        # If we're using autoregressive decoding, now is the time to run the RNN
        if is_autoregressive:
            # Use the last low-resolution latent group as the initial hidden state for the RNN
            low_res_latent = min(reversed(output['latents']), key=lambda z: z.shape[1]).mean(dim=1)
            last_latent = output['latents'][-1]
            rnn_input = torch.cat([encoder_out.embedded_input, last_latent], dim=-1)
            rnn_input = rnn_input[:, :-1]  # Remove [SEP] token to keep the input the same length as the expected output

            rnn_states, _ = self.rnn_decoder(rnn_input, self.rnn_latent_upsample(low_res_latent).unsqueeze(0))
            output['logits'] = self.decoder.output_layer(self.rnn_output_downsample(rnn_states))
            ce_target = batch['token_ids'][:, 1:]  # Remove [CLS] token, thereby shifting everything to the left
        else:
            ce_target = batch['token_ids']

        if return_logits:
            return output['logits']

        nll_target = output['logits'].transpose(-2, -1)  # cross_entropy wants the dimensions permuted for some reason
        ignore_idx = -100 if self.hparams.include_padding_positions else 0
        raw_nll = F.cross_entropy(nll_target, target=ce_target, ignore_index=ignore_idx, reduction='none')
        output['nll'] = raw_nll.sum(dim=-1)

        return output

    # Called both by reconstruct() and sample()
    def decoder_forward(self, x: Optional[Tensor],
                        padding_mask: Tensor = None,
                        batch_dims: List[int] = None,
                        clamped_latents: Mapping[int, Tensor] = None,
                        encoder_states: Mapping[int, Tensor] = None,
                        temperature: float = 1.0,
                        return_latents: bool = False,
                        return_log_probs: bool = False
                        ) -> Dict[str, Any]:
        # We use a coroutine so that we can unobtrusively insert ourselves into Funnel Transformer's forward pass
        # and replace its hidden state with samples from our prior or posterior distributions
        self.decoder.attention_state.upsampling = True
        coroutine = self.decoder.forward_coroutine(
            x,
            padding_mask=padding_mask,
            return_logits=not self.hparams.use_autoregressive_decoder,
            states_to_yield=self.latent_groups
        )

        latents = deque(maxlen=None if return_latents else 1)
        num_blocks = len(self.decoder.hparams.block_sizes)
        stats = defaultdict(float)

        for block_idx, layer_idx, decoder_state in coroutine:
            # Final output- decoder_state is a FunnelTransformerOutput here
            if block_idx == -1:
                output = {**decoder_state.__dict__, **stats}
                if return_latents:
                    output['latents'] = list(latents)

                return output

            # In the first stochastic layer, the prior is just a standard diagonal Gaussian:
            if not latents:
                # Sampling unconditionally
                if decoder_state is not None:
                    batch_dims = list(decoder_state.shape[:2])

                new_shape = batch_dims + [self.hparams.latent_depth]
                prior = self.get_base_prior().expand(new_shape)  # noqa

            # But in subsequent layers, it is conditioned on the previous layer of latent variables:
            else:
                # We may have to upsample the previous latent in order to condition the prior on it
                prev_latent = latents[-1]
                if prev_latent.shape != decoder_state.shape:
                    scale_factor = decoder_state.shape[1] // prev_latent.shape[1]
                    prev_latent = prev_latent.repeat_interleave(scale_factor, dim=1)

                prior = self.get_distribution_for_tensor('p(z)', layer_idx, prev_latent, temperature=temperature)

            if encoder_states is None and decoder_state is None:  # Sample unconditionally
                z = clamped_latents.get(layer_idx) if clamped_latents else None
                if z is None:
                    z = prior.rsample()

                posterior = None
            else:
                # Corresponding block in the encoder, counting in reverse
                if block_idx != 0 and encoder_states is not None:
                    encoder_block_idx = num_blocks - block_idx - 1
                    encoder_state = encoder_states[encoder_block_idx]
                    posterior_input = torch.cat([decoder_state, encoder_state], dim=-1)

                # We're just given the final output of the encoder
                else:
                    posterior_input = torch.cat([torch.zeros_like(decoder_state), decoder_state], dim=-1)

                # Sample conditioned on the input from the encoder
                posterior = self.get_distribution_for_tensor('q(z|x)', layer_idx, posterior_input,
                                                             temperature=temperature)
                z = posterior.rsample()

            self.update_stats_dict(prior, posterior, z, stats, return_log_probs=return_log_probs)

            # Needed by the next stochastic layer up to parameterize its prior distribution
            latents.append(z)

            reconstruction = self.latent_upsample[str(layer_idx)](z)
            if decoder_state is not None:
                out_state = decoder_state + reconstruction

                coroutine.send(out_state)
            else:
                coroutine.send(reconstruction)

    def update_stats_dict(self, prior: Normal, posterior: Normal, z: Tensor, stats: Dict[str, Any],
                          return_log_probs: bool = False):
        # Marginalize over successive layers of latent variables
        if return_log_probs:
            stats['p(z)'] += prior.log_prob(z).sum(dim=(1, 2))
            stats['q(z|x)'] += posterior.log_prob(z).sum(dim=(1, 2))

        # Update the running totals of the KL divergences
        if posterior:
            kl_tensor = torch.distributions.kl_divergence(prior, posterior)

            if not self.hparams.include_padding_positions:
                # Gives us the appropriately scaled mask for the current block
                padding_mask = self.decoder.attention_state.get_padding_mask().unsqueeze(-1)
                kl_tensor.masked_fill_(padding_mask, 0.0)

            stats['kl'] += kl_tensor.sum(dim=(1, 2))  # Sum over seq len and latent dim

    # Used for both the prior 'p(z)' and the posterior 'q(z|x)'
    def get_distribution_for_tensor(self, name: str, layer_idx: int, x: Tensor, temperature: float) -> Normal:
        dist_linear = self.distributions[name][str(layer_idx)]
        mu, logsigma = dist_linear(x).chunk(2, dim=-1)
        stddev = logsigma.exp()

        if temperature != 1.0:
            stddev *= temperature

        return Normal(mu, stddev)

    # Get a tighter estimate for the KL and NLL of some input using Monte Carlo importance sampling
    def get_loss_monte_carlo(self, batch: Dict[str, Any], num_samples: int = 10):
        batch_size, seq_len = batch['token_ids'].shape

        marginal_prob_list = []
        for _ in range(num_samples):
            stats = self.reconstruct(batch, return_log_probs=True)

            log_joint_prob = stats['p(z)'] - stats['nll']  # p(z) * p(x|z) = p(x,z)
            log_marginal_prob = log_joint_prob - stats['q(z|x)']  # p(x,z) / p(z|x) = p(x)
            marginal_prob_list.append(log_marginal_prob)

        # Average over the Monte Carlo samples
        log_marginal_prob = torch.stack(marginal_prob_list, dim=0)
        avg_log_prob = log_marginal_prob.logsumexp(dim=0).sub(math.log(num_samples * batch_size))
        return (-avg_log_prob / seq_len).mean()

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, return_latents: bool = False,
               clamped_latents: Optional[Mapping[int, Tensor]] = None,
               temperature: float = 1.0) -> Tensor:
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
        is_autoregressive = self.hparams.use_autoregressive_decoder
        decoder_output = self.decoder_forward(
            x=None,
            padding_mask=None,
            batch_dims=[count, seed_length],
            clamped_latents=clamped_latents or {},  # Allow for style transfer-type experiments
            return_latents=return_latents or is_autoregressive,
            temperature=temperature
        )

        if is_autoregressive:
            low_res_latent = min(reversed(decoder_output['latents']), key=lambda z: z.shape[1]).mean(dim=1)

            return autoregressive_decode(
                strategy=GenerationStrategy.Greedy,
                rnn=self.rnn_decoder,
                z=decoder_output['final_state'],
                embedding=self.encoder.input_layer[0],
                logit_callable=lambda x: self.decoder.output_layer(self.rnn_output_downsample(x)),
                initial_hidden_state=self.rnn_latent_upsample(low_res_latent),
                max_length=max_length
            )
        else:
            return nonautoregressive_decode(decoder_output['logits'])

    def decoder_requires_grad_(self, requires_grad: bool):
        self.decoder.requires_grad_(requires_grad)
        # self.distributions.requires_grad_(requires_grad)
