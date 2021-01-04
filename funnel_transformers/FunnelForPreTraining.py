from .FunnelTransformer import FunnelTransformer, FunnelBlock
from .RemoteModels import *
from ..HparamUtils import *
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from torch import nn, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


# Funnel Transformer with a decoder block at the end for ELECTRA pretraining
class FunnelForPreTraining(pl.LightningModule):
    default_hparams = AttributeDict(
        funnel_hparams=FunnelTransformer.default_hparams,   # Discriminator and generator hparams copied from this
        train_generator=False,
        use_pretrained_adam_state=False,    # Whether to use the saved Adam `m` and `v` from the pretrained checkpoint
        use_pretrained_weights=True,

        # See Funnel Transformer paper, page 15
        lr=1e-4,
        weight_decay=0.01,
        adam_eps=1e-6
    )

    def __init__(self, hparams: Mapping[str, Any]):
        super(FunnelForPreTraining, self).__init__()

        hparams = merge(self.default_hparams, hparams)
        self.save_hyperparameters(hparams)

        discriminator_hparams = merge(hparams.funnel_hparams, dict(has_decoder_block=True, return_block_outputs=[0]))
        generator_hparams = deepcopy(discriminator_hparams)
        generator_hparams.d_model //= 4
        generator_hparams.num_heads //= 4

        # Generator at index 0, discriminator at index 1
        self.encoders = [FunnelTransformer(generator_hparams), FunnelTransformer(discriminator_hparams)]
        self.decoders = [FunnelBlock(generator_hparams, 3), FunnelBlock(discriminator_hparams, 3)]

        self.mlm_head = nn.Sequential(
            nn.Linear(generator_hparams.d_model, generator_hparams.d_model),
            nn.GELU(),
            nn.LayerNorm([generator_hparams.d_model]),
            nn.Linear(generator_hparams.d_model, generator_hparams.vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.discriminator_head = nn.Sequential(
            nn.Linear(discriminator_hparams.d_model, discriminator_hparams.d_model),
            nn.GELU(),
            nn.Linear(discriminator_hparams.d_model, 1)
        )

        # Freeze the generator weights if indicated
        if not hparams.train_generator:
            self.encoders[0].requires_grad_(False)
            self.decoders[0].requires_grad_(False)

        if hparams.use_pretrained_weights:
            url = remote_model_url_for_hparams(discriminator_hparams, suffix="-FULL-TF")
            self._load_weights_from_tf_ckpt(str(load_remote_model(url) / "model.ckpt"), True)

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, optimizer_index: int) -> Tensor:
        # Either generator or discriminator forward pass
        def _encoder_decoder_forward(inputs: Tensor, model_index: int) -> Tensor:
            encoder, decoder = self.encoders[model_index], self.decoders[model_index]

            encoder_output, hidden_states = encoder(inputs)
            decoder_input = encoder_output + hidden_states[0]  # Residual connection
            return decoder(decoder_input)

        # Train generator
        if optimizer_index == 0:
            raise NotImplementedError

        # Train discriminator
        else:
            masked_text, labels = batch['text'], batch['labels']

            # Probability distribution over tokens: (batch, seq_len, vocab_size)
            generator_logits = _encoder_decoder_forward(masked_text, 0)

            # Sample from the distribution (Gumbel softmax). Greedy sampling, plus some noise.
            noise = torch.rand_like(generator_logits)
            noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
            samples = torch.argmax(generator_logits + noise, -1)    # (batch, seq_len)

            # 1 where the generator output is equal to the ground truth, 0 elsewhere. Note that this will be 1 where
            # a word was masked out and the generator correctly predicted the original token. The discriminator
            # is only asked to find the generator's mistakes.
            is_groundtruth = torch.eq(samples, labels)

            # For each token, the probability that matches the ground truth input.
            discriminator_logits = _encoder_decoder_forward(samples, 1)  # (batch, seq_len)
            weights = 1.0 - mask if (mask := batch.get('padding_mask')) else None
            return F.binary_cross_entropy_with_logits(discriminator_logits, is_groundtruth, weight=weights)

    def configure_optimizers(self):
        # See Funnel Transformer paper, page 15
        adam_hparams = transmute(self.hparams, 'weight_decay', 'lr', eps='adam_eps')
        discriminator_params = chain(self.discriminator.parameters(), self.discriminator_decoder.parameters())
        discriminator_opt = torch.optim.AdamW(**adam_hparams, params=discriminator_params)

        if self.hparams.use_pretrained_adam_state:
            discriminator_opt.state = self.optimizer_state
            del self.optimizer_state

        if self.train_generator:
            generator_params = chain(self.generator.parameters(), self.generator_decoder.parameters())
            generator_opt = torch.optim.AdamW(**adam_hparams, params=generator_params)
            return [generator_opt, discriminator_opt], []
        else:
            return discriminator_opt

    def _load_weights_from_tf_ckpt(self, path: str, strict: bool):
        import tensorflow as tf

        print("Loading model from TensorFlow checkpoint...")
        reader = tf.train.load_checkpoint(path)

        def copy_tf_param(key_string: str, param, layer_index=None, prefix=None):
            # 'attention.v_head.bias' -> 'layer_2/rel_attn/v/bias'

            # 'feedforward.pffn.0.bias' -> 'feedforward.layer_1.bias'
            # 'feedforward.layer_norm.weight' -> 'feedforward.layer_norm.gamma'
            key_string = replace_all(key_string, {
                'layer_norm.weight': 'layer_norm.gamma',
                'layer_norm.bias': 'layer_norm.beta',
                'pffn.0': 'layer_1',
                'pffn.3': 'layer_2',
                'r_kernel': 'r.kernel',  # r_kernel is a parameter of a separate Dense layer in TF
                'lm_loss.weight': 'lm_loss/weight',
                '.weight': '.kernel'
            })

            keys = key_string.split('.')
            if layer_index is not None:
                keys.insert(0, "layer_" + str(layer_index))

            if len(keys) > 1:
                keys[1] = replace_all(keys[1], {  # 'layer_17.feedforward.layer_1.bias' -> 'layer_17.ff.layer_1.bias'
                    'attention': 'rel_attn',
                    'feedforward': 'ff'
                })

            if len(keys) > 2:
                keys[2] = replace_all(keys[2], {  # 'layer_17.rel_attn.v_head.bias' -> 'layer_17.rel_attn.v.bias'
                    '_head': '',
                    'post_proj': 'o'
                })

            # 'layer_17.rel_attn.v.bias' -> 'model/encoder/layer_17/rel_attn/v/bias'
            tf_name = prefix + '/'.join(keys)
            param.data = get_tf_param(tf_name, param)

        def get_tf_param(tf_name: str, param: nn.Parameter):
            weights = reader.get_tensor(tf_name)
            if weights is None and strict:
                return None

            if self.hparams.use_pretrained_adam_state:
                adam_m = reader.get_tensor(tf_name + '/adam_m')
                adam_v = reader.get_tensor(tf_name + '/adam_v')

                if adam_m is not None and adam_v is not None:
                    self.optimizer_state[param] = dict(exp_avg=adam_m, exp_avg_sq=adam_v, step=1)

            # Don't forget to transpose the Dense layer weight matrices when converting to PyTorch
            if 'kernel' in tf_name:
                weights = weights.T

            return torch.from_numpy(weights)

        def copy_tf_params_with_prefix(param_iterator: Iterator[Tuple[str, torch.nn.Parameter, int]], prefix: str):
            for key_string, param, abs_index in param_iterator:
                copy_tf_param(key_string, param, abs_index, prefix)

        def copy_params_with_mapping(module: nn.Module, mapping: Dict[str, str], prefix=""):
            for name, param in module.named_parameters():
                if tf_name := mapping.get(name):
                    # noinspection PyTypeChecker
                    param.data = get_tf_param(prefix + tf_name, param)

        # Store the Adam optimizer state in a temporary attribute until configure_optimizers() is called
        if self.hparams.use_pretrained_adam_state:
            self.optimizer_state = defaultdict(dict)

        copy_params_with_mapping(self.encoders[1].input_layer, prefix='model/input/', mapping={
            '0.weight': 'word_embedding/lookup_table',
            '1.bias': 'layer_norm/beta',
            '1.weight': 'layer_norm/gamma'
        })
        copy_params_with_mapping(self.mlm_head, prefix='generator/', mapping={
            '0.bias': 'lm_proj/dense/bias',
            '0.weight': 'lm_proj/dense/kernel',
            '2.bias': 'lm_proj/layer_norm/beta',
            '2.weight': 'lm_proj/layer_norm/gamma',
            '3.bias': 'lm_loss/bias'
        })
        copy_params_with_mapping(self.discriminator_head, prefix='model/', mapping={
            '0.bias': 'binary_proj/dense/bias',
            '0.weight': 'binary_proj/dense/kernel', 
            '2.bias': 'binary_loss/bias',
            '2.weight': 'binary_loss/weight',
        })

        num_encoder_layers = sum(self.hparams.funnel_hparams.block_sizes)
        def decoder_param_iterator(decoder: FunnelBlock):
            for i, layer in enumerate(decoder.layers):
                for var_name, param in layer.named_parameters():
                    yield var_name, param, i + num_encoder_layers

        copy_tf_params_with_prefix(self.encoders[1].enumerate_parameters_by_layer(), 'model/encoder/')
        copy_tf_params_with_prefix(self.encoders[0].enumerate_parameters_by_layer(), 'generator/encoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.decoders[1]), 'model/decoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.decoders[0]), 'generator/decoder/')

        print("Finished.")
