from sparse_vae import *
from datasets import concatenate_datasets
import sys


def main(args):
    model_str, model_name = args[1:]
    model = load_checkpoint_for_name(model_str, model_name)
    model.freeze()
    model.eval()

    dm = TextDataModule(TextDataModuleHparams())
    dm.prepare_data()
    dataset, tokenizer = dm.dataset, dm.tokenizer
    dataset = concatenate_datasets([dataset['train'], dataset['test']])
    titles = {title: idx for idx, title in enumerate(dataset['title'])}
    gpu_idx = select_best_gpu(min_free_memory=4.0)
    model = model.to(gpu_idx)

    print("Type the title of an article to get a reconstruction. Type q to quit.\nType i to switch to interpolation mode.")
    while True:
        query = input("Article: ")
        if query == 'q':
            return

        article_idx = titles.get(query)
        if article_idx is None:
            print("No article found with that title. Try again.")
        else:
            text = dataset[article_idx]['text']
            latent = model.predict({'token_ids': torch.tensor([text], device=gpu_idx)}, 0).loc
            reconstruction = model.sample(1024, 1, z=latent, temperature=0.7)
            reconstruction = tokenizer.decode(reconstruction.squeeze().tolist())
            print("Reconstruction:\n\n" + reconstruction)


if __name__ == "__main__":
    main(sys.argv)
