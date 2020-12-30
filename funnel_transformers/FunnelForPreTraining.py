from FunnelTransformer import FunnelTransformer, FunnelBlock
from Utilities import *
from copy import deepcopy
from itertools import chain
import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict


# Funnel Transformer with a decoder block at the end for ELECTRA pretraining
class FunnelForPreTraining(pl.LightningModule):
    def __init__(self,
                 discriminator_hparams: AttributeDict,
                 generator_hparams: Optional[AttributeDict] = None,
                 train_generator: bool = False,
                 pretrained_path: Optional[str] = None,
                 strict: bool = False
                 ):
        super(FunnelForPreTraining, self).__init__()
        discriminator_hparams.has_decoder_block = True
        discriminator_hparams.return_block_outputs = [0]  # We just need the first hidden state for the res connection

        generator_hparams = generator_hparams or deepcopy(discriminator_hparams)
        generator_hparams.d_model /= 4
        generator_hparams.num_heads /= 4

        # Generator at index 0, discriminator at index 1
        self.encoders = [FunnelTransformer(generator_hparams), FunnelTransformer(discriminator_hparams)]
        self.decoders = [FunnelBlock(generator_hparams, 3), FunnelBlock(discriminator_hparams, 3)]

        # Freeze the generator weights if indicated
        self.train_generator = train_generator
        if not train_generator:
            for param in chain(self.encoders[0].parameters(), self.decoders[0].parameters()):
                param.requires_grad = False

        if pretrained_path:
            self._load_weights_from_tf_ckpt(pretrained_path, strict)

    # Returns the loss
    def training_step(self, batch: Tensor, batch_index: int, optimizer_index: int) -> Tensor:
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
            generator_output = _encoder_decoder_forward(batch, 0)
            discriminator_output = _encoder_decoder_forward(generator_output, 1)


    def configure_optimizers(self):
        # See Funnel Transformer paper, page 15
        hyperparams = dict(lr=1e-4, weight_decay=0.01, eps=1e-6)
        discriminator_params = chain(self.discriminator.parameters(), self.discriminator_decoder.parameters())
        discriminator_opt = torch.optim.AdamW(**hyperparams, params=discriminator_params)

        if self.train_generator:
            generator_params = chain(self.generator.parameters(), self.generator_decoder.parameters())
            generator_opt = torch.optim.AdamW(**hyperparams, params=generator_params)
            return [generator_opt, discriminator_opt], []
        else:
            return discriminator_opt

    def _load_weights_from_tf_ckpt(self, path: str, strict: bool):
        import tensorflow as tf

        print("Loading model from TensorFlow checkpoint...")
        reader = tf.train.load_checkpoint(path)

        def copy_tf_params_with_prefix(param_iterator: Iterator[Tuple[str, torch.nn.Parameter, int]], prefix: str):
            for key_string, param, abs_index in param_iterator:
                # 'attention.v_head.bias' -> 'layer_2/rel_attn/v/bias'

                # 'feedforward.pffn.0.bias' -> 'feedforward.layer_1.bias'
                # 'feedforward.layer_norm.weight' -> 'feedforward.layer_norm.gamma'
                key_string = replace_all(key_string, [
                    ('layer_norm.weight', 'layer_norm.gamma'),
                    ('layer_norm.bias', 'layer_norm.beta'),
                    ('pffn.0', 'layer_1'),
                    ('pffn.3', 'layer_2'),
                    ('r_kernel', 'r.kernel'),  # r_kernel is a parameter of a separate Dense layer in TF
                    ('weight', 'kernel')
                ])

                keys = key_string.split('.')
                keys.insert(0, "layer_" + str(abs_index))

                keys[1] = replace_all(keys[1], [  # 'layer_17.feedforward.layer_1.bias' -> 'layer_17.ff.layer_1.bias'
                    ('attention', 'rel_attn'),
                    ('feedforward', 'ff')
                ])
                keys[2] = replace_all(keys[2], [  # 'layer_17.rel_attn.v_head.bias' -> 'layer_17.rel_attn.v.bias'
                    ('_head', ''),
                    ('post_proj', 'o')
                ])

                # 'layer_17.rel_attn.v.bias' -> 'model/encoder/layer_17/rel_attn/v/bias'
                tf_name = prefix + '/'.join(keys)

                weights = reader.get_tensor(tf_name)
                if weights is None and strict:
                    return None

                # Don't forget to transpose the Dense layer weight matrices when converting to PyTorch
                if 'kernel' in tf_name:
                    weights = weights.T

                param.data = torch.from_numpy(weights)

        copy_tf_params_with_prefix(self.discriminator.enumerate_parameters_by_layer(), 'model/encoder/')
        copy_tf_params_with_prefix(self.generator.enumerate_parameters_by_layer(), 'generator/encoder/')

        def decoder_param_iterator(decoder: FunnelBlock):
            for i, layer in enumerate(decoder.layers):
                for var_name, param in layer.named_parameters():
                    yield var_name, param, i

        copy_tf_params_with_prefix(decoder_param_iterator(self.discriminator_decoder), 'model/decoder/')
        copy_tf_params_with_prefix(decoder_param_iterator(self.generator_decoder), 'generator/decoder/')

        print("Finished.")
