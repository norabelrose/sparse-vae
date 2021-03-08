from .FunnelWithDecoder import *
from .RemoteModels import *
from ..core import LanguageModel, LanguageModelHparams
from copy import deepcopy
from dataclasses import *
from numpy import prod
from torch import nn, Tensor
import torch.nn.functional as F
import torch


@dataclass
class ElectraModelHparams(LanguageModelHparams):
    # Discriminator and generator hparams copied from this
    funnel: FunnelTransformerHparams = FunnelTransformerHparams(
        positional_attention_type='factorized',
        separate_cls=True,
        use_segment_attention=True
    )
    discriminator_loss_weight: float = 50.0
    train_generator: bool = True
    train_discriminator: bool = True
    use_pretrained_adam_state: bool = False  # Whether to use the saved Adam `m` and `v` from the pretrained checkpoint
    use_pretrained_weights: bool = True

    # See Funnel Transformer paper, page 15
    adam_eps: float = 1e-6


# Funnel Transformer with a decoder block at the end for ELECTRA pretraining
class ElectraModel(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(ElectraModel, self).__init__(hparams)

        discriminator_hparams = hparams.funnel
        generator_hparams = deepcopy(discriminator_hparams)
        generator_hparams.d_embedding = discriminator_hparams.d_model   # Generator shares embeddings w/ discriminator
        generator_hparams.d_model //= 4
        generator_hparams.num_heads //= 4

        self.generator = FunnelWithDecoder(generator_hparams, num_decoder_layers=2)
        self.discriminator = FunnelWithDecoder(discriminator_hparams, num_decoder_layers=2)

        self.mlm_head = nn.Sequential(
            nn.Linear(generator_hparams.d_model, generator_hparams.d_embedding),
            nn.GELU(),
            nn.LayerNorm(generator_hparams.d_embedding),
            nn.Linear(generator_hparams.d_embedding, generator_hparams.vocab_size)
        )
        self.discriminator_head = nn.Sequential(
            nn.Linear(discriminator_hparams.d_model, discriminator_hparams.d_model),
            nn.GELU(),
            nn.Linear(discriminator_hparams.d_model, 1)
        )
        if not hparams.train_discriminator:
            self.discriminator.requires_grad_(False)
            self.discriminator_head.requires_grad_(False)

        # Tie the embedding weight matrices (and embedding layer norm parameters)
        self.generator.encoder.input_layer[0] = self.discriminator.encoder.input_layer[0]
        self.generator.encoder.input_layer[1] = self.discriminator.encoder.input_layer[1]
        self.mlm_head[-1].weight = self.generator.encoder.input_layer[0].weight

        # Freeze the generator weights if indicated
        if not hparams.train_generator:
            self.generator.requires_grad_(False)

        self.has_loaded_from_checkpoint = False

    def forward(self, batch: Dict[str, Tensor]):
        generator_output = self.generator(batch['token_ids']).final_state
        return self.mlm_head(generator_output)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self.has_loaded_from_checkpoint = True

    def on_train_start(self):
        if not self.has_loaded_from_checkpoint and self.hparams.use_pretrained_weights:
            url = remote_model_url_for_hparams(self.hparams.funnel, suffix="-FULL-TF")
            self._load_weights_from_tf_ckpt(str(load_remote_model(url) / "model.ckpt"), True)

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:  # noqa
        masked_text, labels = batch['token_ids'], batch['labels']
        seg_ids, padding = batch['segment_ids'], batch['padding_mask']

        # Probability distribution over tokens: (batch, seq_len, vocab_size)
        generator_output = self.generator(masked_text, padding_mask=padding, segment_ids=seg_ids).final_state
        generator_logits = self.mlm_head(generator_output)

        # Get MLM loss for the generator
        if self.hparams.train_generator:
            gen_loss = F.cross_entropy(generator_logits.flatten(end_dim=-2), labels.flatten())
            self.log('gen_loss', gen_loss, prog_bar=True)
        else:
            gen_loss = 0.0

        if self.current_epoch > 1:
            # Sample from the distribution (Gumbel softmax). Greedy sampling, plus some noise.
            noise = torch.rand_like(generator_logits)
            noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)     # noqa
            samples = (generator_logits.detach() + noise).argmax(-1)    # (batch, seq_len)

            # 1 where the generator output is different from the ground truth, 0 elsewhere. Note that this will be 1
            # where a word was masked out and the generator correctly predicted the original token. The discriminator
            # is only asked to find the generator's mistakes.
            is_groundtruth = samples.ne(labels).float()

            # For each token, the probability that matches the ground truth input.
            discriminator_output = self.discriminator(samples, padding_mask=padding, segment_ids=seg_ids).final_state
            discriminator_logits = self.discriminator_head(discriminator_output).squeeze(-1)

            discr_losses = F.binary_cross_entropy_with_logits(discriminator_logits, is_groundtruth, reduction='none')
            discr_losses *= ~padding.bool()
            discr_loss = discr_losses.mean()

            self.log('discr_loss', discr_loss, prog_bar=True)
            return self.hparams.discriminator_loss_weight * discr_loss + gen_loss  # noqa
        else:
            return gen_loss

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index)

    def _load_weights_from_tf_ckpt(self, path: str, strict: bool):
        import tensorflow as tf

        print("\nLoading model from TensorFlow checkpoint...")
        reader = tf.train.load_checkpoint(path)
        param_list = {param: name for name, param in self.named_parameters()}
        # var_list = [var_tuple[0] for var_tuple in tf.train.list_variables(path)]

        copy_adam_state = self.hparams.use_pretrained_adam_state
        adam_state = self.optimizers().state if copy_adam_state else None
        device = self.device

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
                # 'lm_loss.bias': 'lm_loss/bias',
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
            del param_list[param]

        def realign(tf_tensor, pt_shape, name):
            # Align the shapes if need be
            tf_shape = tf_tensor.shape
            if tf_shape != pt_shape:
                assert tf_tensor.numel() == prod(pt_shape),\
                    f"{name} of shape {tf_shape} cannot be coerced into shape {pt_shape}"

                singleton_dims = [i for i, x in enumerate(pt_shape) if x == 1]  # Indices of singleton dims
                tf_tensor.squeeze_()

                pt_shape2, tf_shape2 = [x for x in pt_shape if x != 1], tf_tensor.shape
                try:
                    tf_tensor = tf_tensor.permute(*[tf_shape2.index(x) for x in pt_shape2])
                except ValueError:
                    raise ValueError(f"Cannot permute() {name} from {tf_shape} to {pt_shape} since the "
                                     f"nonsingleton dimensions differ.")

                for i in singleton_dims:
                    tf_tensor.unsqueeze_(i)

            return tf_tensor

        def get_tf_param(tf_name: str, param: nn.Parameter):
            weights = reader.get_tensor(tf_name)
            # var_list.remove(tf_name)
            if weights is None and strict:
                return None

            if copy_adam_state:
                adam_m = reader.get_tensor(tf_name + '/adam_m')
                adam_v = reader.get_tensor(tf_name + '/adam_v')

                if adam_m is not None and adam_v is not None:
                    # assert list(param.shape) == list(adam_m.shape) == list(adam_v.shape)
                    adam_state[param] = dict(
                        exp_avg=realign(torch.from_numpy(adam_m).to(device), param.shape, tf_name),
                        exp_avg_sq=realign(torch.from_numpy(adam_v).to(device), param.shape, tf_name),
                        step=1
                    )

            tensor = torch.from_numpy(weights).to(device)
            return realign(tensor, param.shape, tf_name)

        def copy_tf_params_with_prefix(param_iterator: Iterator[Tuple[str, torch.nn.Parameter, int]], prefix: str):
            for key_string, param, abs_index in param_iterator:
                copy_tf_param(key_string, param, abs_index, prefix)

        def copy_params_with_mapping(module: nn.Module, mapping: Dict[str, str], prefix=""):
            for name, param in module.named_parameters():
                if tf_name := mapping.get(name):
                    # noinspection PyTypeChecker
                    param.data = get_tf_param(prefix + tf_name, param)
                    del param_list[param]

        gen_input, discr_input = self.generator.encoder.input_layer, self.discriminator.encoder.input_layer
        copy_params_with_mapping(discr_input, prefix='model/input/', mapping={
            '0.weight': 'word_embedding/lookup_table',
            '1.bias': 'layer_norm/beta',
            '1.weight': 'layer_norm/gamma'
        })
        copy_params_with_mapping(gen_input, prefix='generator/encoder/', mapping={
            '2.bias': 'input_projection/bias',
            '2.weight': 'input_projection/kernel'
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

        num_encoder_layers = sum(self.hparams.funnel.block_sizes)
        def decoder_param_iterator(decoder):
            for i, layer in enumerate(decoder):
                for var_name, param in layer.named_parameters():
                    yield var_name, param, i + num_encoder_layers

        copy_tf_params_with_prefix(self.discriminator.encoder.enumerate_parameters_by_layer(), 'model/encoder/')
        copy_tf_params_with_prefix(self.generator.encoder.enumerate_parameters_by_layer(), 'generator/encoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.discriminator.decoder), 'model/decoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.generator.decoder), 'generator/decoder/')

        print("Uninitialized parameters: ", list(param_list.values()))
        print("Finished.")
