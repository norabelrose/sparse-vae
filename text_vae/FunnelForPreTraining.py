from .FunnelWithDecoder import *
from .RemoteModels import *
from .HparamUtils import *
from collections import defaultdict
from copy import deepcopy
from dataclasses import *
from torch import nn, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


@dataclass
class FunnelForPreTrainingHparams:
    # Discriminator and generator hparams copied from this
    funnel: FunnelTransformerHparams = field(default_factory=FunnelTransformerHparams)
    train_generator: bool = False,
    use_pretrained_adam_state: bool = False,  # Whether to use the saved Adam `m` and `v` from the pretrained checkpoint
    use_pretrained_weights: bool = True,

    # See Funnel Transformer paper, page 15
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    adam_eps: float = 1e-6


# Funnel Transformer with a decoder block at the end for ELECTRA pretraining
class FunnelForPreTraining(pl.LightningModule):
    def __init__(self, hparams: Union[FunnelForPreTrainingHparams, OmegaConf]):
        super(FunnelForPreTraining, self).__init__()

        if isinstance(hparams, FunnelTransformerHparams):
            hparams = OmegaConf.structured(FunnelTransformerHparams)

        self.save_hyperparameters(hparams)

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
            nn.LayerNorm([generator_hparams.d_embedding]),
            nn.Linear(generator_hparams.d_embedding, generator_hparams.vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.discriminator_head = nn.Sequential(
            nn.Linear(discriminator_hparams.d_model, discriminator_hparams.d_model),
            nn.GELU(),
            nn.Linear(discriminator_hparams.d_model, 1)
        )
        # Tie the embedding weight matrices
        self.mlm_head[-2].weight.data = self.generator.encoder.input_layer[0].weight.data

        # Freeze the generator weights if indicated
        if not hparams.train_generator:
            self.generator.requires_grad_(False)

        if hparams.use_pretrained_weights:
            url = remote_model_url_for_hparams(discriminator_hparams, suffix="-FULL-TF")
            self._load_weights_from_tf_ckpt(str(load_remote_model(url) / "model.ckpt"), True)

    def configure_optimizers(self):
        # See Funnel Transformer paper, page 15
        adam_hparams = transmute(self.hparams, 'weight_decay', 'lr', eps='adam_eps')
        discr_opt = torch.optim.AdamW(**adam_hparams, params=self.discriminator.parameters())

        if self.hparams.use_pretrained_adam_state:
            discr_opt.state = self.optimizer_state
            del self.optimizer_state

        if self.hparams.train_generator:
            generator_opt = torch.optim.AdamW(**adam_hparams, params=self.generator.parameters())
            return [generator_opt, discr_opt], []
        else:
            return discr_opt

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int, optimizer_index: int = 0) -> Tensor:
        # Train generator
        if optimizer_index == 0 and self.hparams.train_generator:
            raise NotImplementedError

        # Train discriminator
        else:
            masked_text, labels, padding_mask = batch['token_ids'], batch['labels'], batch['padding_mask']
            nonpadding_mask = (~padding_mask).float()
            padding_mask = padding_mask.float()

            # Probability distribution over tokens: (batch, seq_len, vocab_size)
            generator_output = self.generator(masked_text, input_mask=padding_mask)['output']
            generator_logits = self.mlm_head(generator_output)

            # Sample from the distribution (Gumbel softmax). Greedy sampling, plus some noise.
            noise = torch.rand_like(generator_logits)
            noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
            samples = torch.argmax(generator_logits + noise, -1)    # (batch, seq_len)

            # 1 where the generator output is equal to the ground truth, 0 elsewhere. Note that this will be 1 where
            # a word was masked out and the generator correctly predicted the original token. The discriminator
            # is only asked to find the generator's mistakes.
            is_groundtruth = samples.eq(labels).float()

            # For each token, the probability that matches the ground truth input.
            discriminator_output = self.discriminator(samples, input_mask=padding_mask)['output']
            discriminator_logits = self.discriminator_head(discriminator_output)
            return F.binary_cross_entropy_with_logits(discriminator_logits, is_groundtruth, weight=nonpadding_mask)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        del checkpoint['hyper_parameters']['__builtins__']

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index, optimizer_index=0)

    def validation_epoch_end(self, losses: List[Tensor]):
        self.log('val_loss', torch.mean(torch.stack(losses)))

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

            tensor = torch.from_numpy(weights)

            # Align the shapes if need be
            pt_shape, tf_shape = param.shape, tensor.shape
            if pt_shape != tf_shape:
                assert param.numel() == tensor.numel(), f"{tf_name} of shape {tf_shape} cannot be coerced into shape "\
                                                        f"{pt_shape}"

                singleton_dims = [i for i, x in enumerate(pt_shape) if x == 1]  # Indices of singleton dims
                tensor.squeeze_()

                pt_shape2, tf_shape2 = param.squeeze().shape, tensor.shape
                try:
                    tensor = tensor.permute(*[tf_shape2.index(x) for x in pt_shape2])
                except ValueError:
                    raise ValueError(f"Cannot permute() {tf_name} from {tf_shape} to {pt_shape} since the "\
                                     f"nonsingleton dimensions differ.")

                for i in singleton_dims:
                    tensor.unsqueeze_(i)

            return tensor

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

        gen_input, discr_input = self.generator.encoder.input_layer, self.discriminator.encoder.input_layer
        copy_params_with_mapping(discr_input, prefix='model/input/', mapping={
            '0.weight': 'word_embedding/lookup_table',
            '1.bias': 'layer_norm/beta',
            '1.weight': 'layer_norm/gamma'
        })
        copy_params_with_mapping(gen_input, prefix='generator/encoder/', mapping={
            '1.bias': 'input_projection/bias',
            '1.weight': 'input_projection/kernel'
        })
        gen_input[0].weight.data = discr_input[0].weight.data   # Tie embedding weights

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
        def decoder_param_iterator(decoder: nn.Module):
            for i, layer in enumerate(decoder.layers):
                for var_name, param in layer.named_parameters():
                    yield var_name, param, i + num_encoder_layers

        copy_tf_params_with_prefix(self.discriminator.encoder.enumerate_parameters_by_layer(), 'model/encoder/')
        copy_tf_params_with_prefix(self.generator.encoder.enumerate_parameters_by_layer(), 'generator/encoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.discriminator.decoder), 'model/decoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.generator.decoder), 'generator/decoder/')

        print("Finished.")