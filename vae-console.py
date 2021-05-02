import sys
from sparse_vae import *


user_env = {}


def encode(user_string):
    tokens = user_env['tokenizer'].encode(user_string).ids
    user_env['posterior'] = user_env['vae'].encode(torch.tensor(tokens))

def load(version_name):
    ckpt_path = get_checkpoint_path_for_name('transformer-vae', version_name)
    user_env['vae'] = TransformerVAE.load_from_checkpoint(ckpt_path)
    print(f"Loaded transformer VAE from {ckpt_path}.")

def print_help():
    print(list(commands.keys()))


commands = {
    'encode': encode,
    'load': load,
    'help': print_help
}


def main(args):
    load(args[1])

    vocab_path = Path.cwd() / 'sparse-vae-pretrained' / 'tokenizers' / 'yelp_polarity.json'
    assert vocab_path.exists(), f"Couldn't find pretrained tokenizer for yelp_polarity"

    user_env['tokenizer'] = Tokenizer.from_file(str(vocab_path))

    print("This is an augmented Python console. Type 'help' to get a list of commands.")
    while True:
        command = input()

        for cmd_name, func in commands.items():
            if command.startswith(cmd_name):
                prefix_len = len(cmd_name) + 1
                if len(command) <= prefix_len:
                    func()
                    break

                cmd_input = command[prefix_len:]
                try:
                    py_object = exec(cmd_input, user_env)
                except SyntaxError:
                    print(f"'{cmd_input}' is not a valid Python expression.")
                else:
                    func(py_object)
                finally:
                    break

        # The user didn't enter any of our commands- just execute the input as a Python statement
        try:
            py_object = exec(command, user_env)
        except SyntaxError:
            print(f"'{command}' is not a valid Python expression.")
        else:
            if py_object is not None:
                print(py_object)


if __name__ == "__main__":
    main(sys.argv)
