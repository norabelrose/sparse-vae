from functools import partial
from pathlib import Path
from text_vae import batch_generate_samples, get_checkpoint_path_for_name, select_best_gpu, Transformer
from tokenizers import Tokenizer
import sys


def main(args):
    model_name = args[1]
    gpu = int(args[2]) if len(args) > 2 else select_best_gpu()

    path = get_checkpoint_path_for_name('transformer-lm', model_name)
    transformer = Transformer.load_from_checkpoint(path).to('cuda:' + str(gpu))
    transformer.start_token = 2
    transformer.end_token = 3

    sample_func = partial(transformer.sample, max_length=512, batch_size=1000, beam_size=2)
    outputs = batch_generate_samples(sample_func, num_samples=2_000, max_length=512, end_token=3)

    print("Decoding tokens...")
    tokenizer_path = Path.cwd() / 'text-vae-pretrained' / 'tokenizers' / 'yelp_polarity.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    outputs = tokenizer.decode_batch([x.tolist() for x in outputs])


if __name__ == "__main__":
    main(sys.argv)
