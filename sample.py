from text_vae import *
import sys


def main(args):
    model_type, model_name = args[1:]
    gpu = select_best_gpu()

    path = get_checkpoint_path_for_name(model_type, model_name)
    model_class = TransformerVAE if model_type == 'transformer-vae' else LSTMVAE
    model = model_class.load_from_checkpoint(path).to('cuda:' + str(gpu))
    model.eval()
    model.start_token = 2
    model.end_token = 3

    sample_func = partial(model.sample, max_length=512, batch_size=1000)
    outputs = batch_generate_samples(sample_func, num_samples=700_000, max_length=512, end_token=3)
    # generator = model.sample_generator(512, 700_000, batch_size=1000)
    # outputs = batch_generate_samples2(generator, num_samples=700_000, max_length=512, end_token=3)

    print("Saving to disk...")
    dataset_path = Path.cwd() / 'text-vae-datasets' / 'samples' / model_name
    dataset = Dataset.from_dict({'text': outputs})
    dataset = dataset.train_test_split(test_size=50_000)
    dataset.save_to_disk(str(dataset_path))
    print("Done.")


if __name__ == "__main__":
    main(sys.argv)
