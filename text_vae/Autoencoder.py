from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from numpy import prod
from omegaconf import OmegaConf
from torch import nn
from torch import Tensor
from .Utilities import *
from .AttentionState import AttentionState
from .FunnelTransformer import FunnelTransformer, FunnelTransformerHparams
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


@dataclass
class AutoencoderHparams:
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        block_sizes=(4, 4, 4),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2)  # How much the hidden state is downsampled between each encoder block
    )
    latent_depth: int = 16  # Depth of the latent tensors (dimensionality per token)
    use_pretrained_encoder: bool = True
    copy_encoder_weights_to_decoder: bool = True
    use_autoregressive_decoding: bool = False

    batch_size: int = 0    # This is here just for compatibility with pl.Trainer's auto_scale_batch_size feature
    grad_clip_threshold: float = 200.0
    grad_skip_threshold: float = 400.0
    kl_warmup_steps: int = 0  # For KL annealing
    lr: float = 1e-3
    warmup_steps: int = 1000
    weight_decay: float = 0.01


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams: OmegaConf):
        super().__init__()

        # save_hyperparameters() stores the hparams in self.hparams and ensures they are saved to disk during training.
        self.save_hyperparameters(hparams)

        encoder_hparams = hparams.encoder
        all_blocks = list(range(len(encoder_hparams.block_sizes)))
        encoder_hparams = mutate(encoder_hparams, block_outputs_to_return=all_blocks)
        
        # The encoder and decoder share an AttentionState object, which caches positional encodings and the
        # padding mask for each scale
        attn_state = AttentionState(encoder_hparams)
        attn_state.shared = True
        decoder_hparams = mutate(
            encoder_hparams,
            block_sizes=encoder_hparams.block_sizes[::-1],          # Reverse the order of the blocks
            scaling_factors=encoder_hparams.scaling_factors[::-1],
            upsampling=True
        )
        self.encoder = FunnelTransformer(encoder_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)
        self.encoder.attention_state = attn_state
        self.decoder.attention_state = attn_state
        
        # Initial input into the decoder
        overt_depth, latent_depth = encoder_hparams.d_model, hparams.latent_depth
        self.decoder_seed = nn.Parameter(torch.zeros(1, 1, overt_depth))

        # Construct Linear layers which generate p(z), q(z|x), and p(x|z) given the output of a Transformer layer
        def linears_with_gelu(input_dim, output_dim):
            return nn.ModuleList([
                # Note that we apply the activation function FIRST and then the Linear layer
                nn.Sequential(nn.GELU(), nn.Linear(input_dim, output_dim))
                for _ in range(sum(encoder_hparams.block_sizes))
            ])

        self.distributions = nn.ModuleDict({
            # Output contains mu & log sigma for p(z)
            'p(z)': linears_with_gelu(overt_depth, latent_depth * 2),
            # Input is the encoder and decoder states concatenated depthwise, output is mu & log sigma for q(z|x)
            'q(z|x)': linears_with_gelu(overt_depth * 2, latent_depth * 2)
        })
        self.latent_upsample = linears_with_gelu(latent_depth, overt_depth)

        # After each layer in the decoder, call decoder_layer_forward with the layer's output and the block and
        # layer indices
        absolute_idx = 0
        for block_idx, block in enumerate(self.decoder.blocks):
            for layer in block.layers:
                layer.output_transform = partial(self.decoder_layer_forward, absolute_idx, block_idx)
                absolute_idx += 1

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder.load_pretrained_weights()

            if hparams.copy_encoder_weights_to_decoder:
                # Only copy the first three blocks of the encoder because they are the ones that are pretrained
                encoder_blocks = self.encoder.blocks[:3]
                decoder_blocks = self.decoder.blocks[-3:]

                for enc_block, dec_block in zip(encoder_blocks, reversed(decoder_blocks)):
                    dec_block.load_state_dict(enc_block.state_dict())

    def configure_optimizers(self):
        # Cosine decay learning rate schedule with warmup steps
        def cosine_with_warmup(current_step, num_cycles=1):
            warmups = self.hparams.warmup_steps
            if current_step < warmups:
                return float(current_step) / float(max(1, warmups))

            total_steps = self.trainer.max_steps
            assert total_steps, "Max training steps must be known to use lr decay."

            progress = float(current_step - warmups) / float(max(1, total_steps - warmups))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        weight_decay, lr = self.hparams.weight_decay, self.hparams.lr
        adam = torch.optim.AdamW(weight_decay=weight_decay, lr=lr, params=self.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(adam, cosine_with_warmup)

        return [adam], [scheduler]

    def forward(self, batch: Dict[str, Any]) -> List[Tensor]:
        batch['input'] = batch['token_ids']

        self.encoder.attention_state.configure_for_input(batch)
        return self.encoder(batch)

    # Get a tighter estimate for the negative log likelihood of some input using Monte Carlo importance sampling
    def get_nll_monte_carlo(self, batch: Dict[str, Any], num_samples: int = 10):
        batch_size = batch['token_ids'].shape[0]
        batch['return_log_probs'] = True

        marginal_prob_list = []
        for _ in range(num_samples):
            decoder_output = self.reconstruct(batch)

            # These are all scalar tensors, marginalized over batches, tokens, and the latent depth
            log_prior_prob = decoder_output['p(z)']
            log_posterior_prob = decoder_output['q(z|x)']
            log_conditional_prob = -decoder_output['nll']     # p(x|z)

            log_joint_prob = log_prior_prob + log_conditional_prob  # p(z) * p(x|z) = p(x,z)
            log_marginal_prob = log_joint_prob - log_posterior_prob  # p(x,z) / p(z|x) = p(x)
            marginal_prob_list.append(log_marginal_prob)

        # Average over the Monte Carlo samples
        log_marginal_prob = torch.stack(marginal_prob_list, dim=0)
        avg_log_prob = log_marginal_prob.logsumexp(dim=0).sub(math.log(num_samples * batch_size))
        return -avg_log_prob

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, return_latents: bool = False,
               clamped_latents: Optional[Mapping[int, Tensor]] = None,
               temperature: Union[float, Mapping[int, float]] = 1.0) -> Tensor:
        # Allow for style transfer-type experiments
        decoder_input = dict(clamped_latents=clamped_latents or {})

        # The temperature parameter can either be a Mapping that assigns different temperatures to different layer
        # indices, or a single float for all layers
        if isinstance(temperature, Mapping):
            decoder_input['temperatures'] = defaultdict(lambda: 1.0)
            decoder_input['temperatures'].update(temperature)
        elif isinstance(temperature, float):
            decoder_input['temperatures'] = defaultdict(lambda: temperature)

        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder_hparams
        total_scaling = prod(funnel_hparams.scaling_factors)
        seed_length = max_length // total_scaling

        seed = self.decoder_seed.expand(count, seed_length, funnel_hparams.d_model)
        decoder_input['input'] = seed

        output_logits: Tensor = self.decoder(decoder_input)['logits']  # (batch, seq_len, vocab_size)
        output_ids = output_logits.topk(top_k, dim=-1)

        return output_ids

    # This is wrapped in a functools.partial() and called by FunnelTransformer once for each layer in the decoder
    def decoder_layer_forward(self, layer_index: int, block_index: int, layer_out: Dict[str, Any]) -> Dict[str, Any]:
        # Used for both the prior 'p(z)' and the posterior 'q(z|x)'
        def get_distribution_for_input(name: str, x: Tensor) -> torch.distributions.Normal:
            dist_linear = self.distributions[name][layer_index]
            mu, logsigma = dist_linear(x).chunk(2, dim=-1)
            stddev = logsigma.exp()

            # Really wish we had Swift-style optional chaining syntax here tbh
            temp = 1.0 if not (temps := layer_out.get('temperatures')) else temps[layer_index]
            if temp != 1.0:
                stddev *= temp

            return torch.distributions.Normal(mu, stddev)

        prior = get_distribution_for_input('p(z)', layer_out['kv'])

        # Sample conditioned on the encoder state
        if states := layer_out.get('encoder_states'):
            encoder_state = states[-1 - block_index]  # Traverse the list of encoder states backwards
            q_input = torch.cat([layer_out['q'], encoder_state], dim=-1)

            posterior = get_distribution_for_input('q(z|x)', q_input)
            z = posterior.rsample()

            if layer_out.get('return_log_probs'):
                for dist, name in ((prior, 'p(z)'), (posterior, 'q(z|x)')):
                    sample_log_prob = dist.log_prob(z).sum()

                    # Marginalize over successive layers of latent variables
                    running_total = layer_out.get(name) or torch.zeros_like(sample_log_prob)
                    layer_out[name] = running_total + sample_log_prob

            if self.training:
                # Update the running totals of the KL divergences and the number of nonpadding positions.
                kl_tensor = torch.distributions.kl_divergence(prior, posterior)

                # Gives us the appropriately scaled mask for the current block
                padding_mask = self.decoder.attention_state.get_padding_mask().unsqueeze(-1)
                kl_tensor = kl_tensor.masked_fill(padding_mask, 0.0)  # Ignore KL divergences for padding positions

                kl_running_total = layer_out.get('total_kl') or kl_tensor.new_zeros([])
                num_pos_running_total = layer_out.get('total_nonpadding') or padding_mask.new_zeros([])
                layer_out['total_kl'] = kl_running_total + kl_tensor.sum()
                layer_out['total_nonpadding'] = num_pos_running_total + (~padding_mask).sum()

        # Sample unconditionally
        else:
            clamped = layer_out.get('clamped_latents')
            z = clamped.get(layer_index) if clamped else None
            if z is None:
                prior.rsample()

        if layer_out.get('return_latents'):
            layer_out['latents'] = layer_out.get('latents', []) + [z]

        reconstruction = self.latent_upsample[layer_index]
        layer_out['kv'] = layer_out['kv'] + reconstruction(z)

        return layer_out

    # Runs a forward pass through the encoder and the decoder and returns a dict with the KL and reconstruction loss
    def reconstruct(self, batch: Dict[str, Any], loss_only: bool = True) -> Dict[str, Tensor]:
        batch['keep_masks'] = True
        batch = self(batch)  # Encoder forward pass

        encoder_states = batch.pop('hidden_states')
        batch['input'] = self.decoder_seed.expand_as(encoder_states[-1])
        batch['encoder_states'] = encoder_states

        self.decoder.attention_state.upsampling = True
        output = self.decoder(batch)

        output['nll'] = F.nll_loss(output['logits'].transpose(-2, -1), target=batch['token_ids'], ignore_index=0)
        if loss_only:
            del output['logits']

        if self.training:
            # We're careful to exclude padding positions from our KL divergence calculation
            output['kl'] = output.pop('total_kl') / (output.pop('total_nonpadding') * self.hparams.latent_depth)

        return output

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        result = self.reconstruct(batch)

        self.log_dict(select(result, 'kl', 'nll'))
        return result['kl'] + result['nll']  # Negative ELBO (evidence lower bound)

    # Implements gradient skipping to stabilize training as described in 'Very Deep VAEs'
    def on_after_backward(self):
        # Don't clip at all if .grad_clip_threshold is falsy
        if not (clip_threshold := self.hparams.grad_clip_threshold):
            return

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_threshold)
        if grad_norm > self.hparams.grad_skip_threshold:
            self.zero_grad()

        self.log('grad_norm', grad_norm)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.get_nll_monte_carlo(batch)

    def validation_epoch_end(self, losses: List[Tensor]):
        self.log('val_loss', torch.mean(torch.stack(losses)))

    def test_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.get_nll_monte_carlo(batch, num_samples=100)

    def test_epoch_end(self, losses: List[Tensor]):
        self.log('test_loss', torch.mean(torch.stack(losses)))
