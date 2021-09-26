from .core import (
    Attention, GenerationState,
    ConditionalGaussian, ContinuousVAE, ContinuousVAEHparams, PaddedTensor, Perceiver,
    Transformer, TransformerLanguageModel, TransformerHparams, marginal_kl
)
from copy import deepcopy
from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import *
import torch


@dataclass
class TransformerVAEHparams(TransformerHparams, ContinuousVAEHparams):
    latent_depth: int = 64
    pretrained_encoder: bool = False
    pretrained_decoder: bool = False
    use_gpt2: bool = False
    early_stopping_metric: str = 'val_nll'


class TransformerVAE(TransformerLanguageModel, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_input_layer = deepcopy(self.input_layer)
        self.encoder_input_layer[0].weight = self.input_layer[0].weight
        self.q_of_z_given_x = ConditionalGaussian(hparams.d_model, hparams.latent_depth)

        # if hparams.use_gpt2:
        #     from transformers import GPT2LMHeadModel, GPT2Config
#
        #     if hparams.pretrained_decoder:
        #         # Distilled 6-layer version of GPT-2
        #         self.decoder = GPT2LMHeadModel.from_pretrained('distilgpt2')
        #         if hparams.sparse_self_attention:
        #             from deepspeed.ops.sparse_attention import SparseAttentionUtils
#
        #             SparseAttentionUtils.extend_position_embedding(self.decoder, max_position=25_008)
        #             SparseAttentionUtils.replace_model_self_attention_with_sparse_self_attention(
        #                 self.decoder, max_position=25_008,
        #                 sparsity_config=SlidingWindowSparsityConfig(num_heads=8, window_size=hparams.attn_window_size)
        #             )
        #     else:
        #         self.decoder = GPT2LMHeadModel(GPT2Config.from_pretrained('distilgpt2'))
        # else:
        #     self.decoder = Transformer(num_layers=hparams.num_layers, d_model=hparams.d_model)

        self.encoder = Perceiver(
            num_layers=hparams.num_layers // 2, num_latents=64, d_model=hparams.d_model, bottleneck_width=1
        )
        self.z_projections = nn.ModuleList([
            nn.Linear(hparams.latent_depth, hparams.d_model)
            for _ in range(hparams.num_layers)
        ])

    def training_step(self, batch: Dict[str, PaddedTensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids'].long()
        if original.numel() > 2 ** 16:
            breakpoint()

        x = self.input_layer(original)
        encoder_out = self.encoder(x)

        z, kl, posterior = self.sample_z(encoder_out, token_counts=batch['num_tokens'], stage=stage)

        logits = self.reconstruct(x, z)[..., :-1, :]
        nll = self.get_nll(
            logits, original[..., 1:], stage=stage,
            bytes_per_token=batch['num_bytes'] / batch['num_tokens'] if stage == 'val' else None
        )
        loss = nll + self.hparams.kl_weight * kl

        # There appears to be a minor bug in our batch creation code- there really shouldn't be any
        # batches that have only one document in them, but there seem to be
        if original.shape[0] > 1:
            mutual_info = kl - marginal_kl(posterior)
            self.log(stage + '_mc_mutual_info', mutual_info)

        if stage == 'train':
            return {'loss': loss, 'posterior': posterior}
        elif stage == 'val':
            self.log('val_loss', nll + kl)

    def validation_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def test_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        original = batch['token_ids'].long()
        x = self.input_layer(original)

        posterior = self.q_of_z_given_x(self.encoder(x))
        log_prob = self.estimate_log_prob_iw(posterior, x, original, num_samples=100, num_iter=100) / batch['num_tokens']
        nll_iw = -log_prob.mean()
        self.log('nll_iw', nll_iw, on_step=True)
        return nll_iw

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x = self.input_layer(batch['token_ids'].long())
        return self.q_of_z_given_x(self.encoder(x), get_kl=False)

    def reconstruct(self, x, z) -> Tensor:
        should_checkpoint = self.hparams.grad_checkpointing and x.requires_grad
        # z = self.z_to_hidden(z)
        for i, layer in enumerate(self.decoder_layers):
            z_hidden = self.z_projections[i](z)
            x = torch.cat([z_hidden, x[..., 1:, :]], dim=-2)
            x = layer(x) if not should_checkpoint else checkpoint(layer, x)

        return self.output_layer(x)

    @torch.cuda.amp.autocast()
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        # Unconditional samples will be mostly garbage when we haven't annealed to the full KL weight
        if self.hparams.kl_weight < 1.0:
            return None

        # noinspection PyUnreachableCode
        z = kwargs.pop('z', None)
        if z is None:
            z = torch.randn(batch_size, 1, self.hparams.latent_depth, device=self.device)

        # z = self.z_to_hidden(z)
        state = GenerationState(
            max_length, batch_size, self.start_token, self.end_token, device=self.device, **kwargs
        )
        state.current_index = 1
        state.output_ids[:, 0] = self.start_token

        with Attention.kv_cache(max_length):
            while not state.should_stop():
                inputs = state.prev_tokens()
                x = self.input_layer(inputs)

                for i, layer in enumerate(self.decoder_layers):
                    if state.current_index == 1:
                        x = torch.cat([self.z_projections[i](z), x[..., 1:, :]], dim=-2)

                    x = layer(x)

                next_logits = self.output_layer(x.squeeze(1))
                continuing_mask = state.process_logits(next_logits)

                Attention.update_kv_cache(continuing_mask)

        return state.final_output()
