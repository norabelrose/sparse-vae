import sys
from text_vae import *


def try_type_conversion(value, target_type):
    try:
        return target_type(value)
    except ValueError:
        print(f"Invalid input- expected value of type {target_type.__name__}.")
        return None


def main(args):
    version_name = args[1]
    sampler = QuantizedVAESampler.for_vae(version_name)

    vocab_path = Path.cwd() / 'text-vae-pretrained' / 'tokenizers' / 'yelp_polarity.json'
    assert vocab_path.exists(), f"Couldn't find pretrained tokenizer for yelp_polarity"

    tokenizer = Tokenizer.from_file(str(vocab_path))
    options = QuantizedVAESamplingOptions()

    print("Type 's' to generate a sample, or 'q' to quit. Type 'help' for a list of other commands.")
    while True:
        command = input()

        if command.startswith('set '):
            rest = command[4:]
            parts = rest.split('=')

            if len(parts) != 2:
                print("Expected a command of the form 'set max_length=500'")
                continue

            parts = [part.strip() for part in parts]
            key, value = parts

            # For moving the model between devices
            if key == 'gpu':
                if value.lower() == 'none':
                    sampler = sampler.to('cpu')
                    print("Model moved to the CPU.")
                else:
                    idx = try_type_conversion(value, int)
                    if idx is not None:
                        print(f"Moving model to GPU {idx}...")
                        sampler = sampler.to('cuda:' + str(idx))
                        print("Done.")

            # For loading different VAE versions
            elif key == 'version':
                try:
                    new_sampler = QuantizedVAESampler.for_vae(value)
                except AssertionError as ex:
                    print(f"Creating a sampler for VAE '{value}' failed with error: {ex}")
                else:
                    sampler = new_sampler

            # For changing the sampling options
            else:
                if not hasattr(options, key):
                    print(f"'{key}' is not a valid configuration option.")
                    continue

                key_type = type(getattr(options, key))
                value = try_type_conversion(value, key_type)
                if value is not None:
                    setattr(options, key, value)

        elif command == 's':
            breakpoint()
            output = sampler.sample(options)
            samples = tokenizer.decode_batch(output.tolist())
            for sample in samples:
                print(sample)

        elif command == 'q':
            return

        elif command == 'config':
            print("Current sampling options:")
            print(asdict(options))

        else:
            print("Not a recognized command. ")


if __name__ == "__main__":
    main(sys.argv)
