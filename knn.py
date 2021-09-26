from sparse_vae import *
from datasets import Dataset
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
import sys


def main(args):
    model_str, model_name = args[1:]
    save_path = Path.cwd() / 'sparse-vae-datasets' / 'latents' / model_str / model_name
    dataset = Dataset.load_from_disk(str(save_path))
    titles = {title: idx for idx, title in enumerate(dataset['title'])}

    gpu_idx = select_best_gpu(min_free_memory=4.0)
    dataset.set_format('torch', device=gpu_idx)
    posteriors = Normal(loc=dataset['latent'], scale=dataset['scale'])
    dataset.reset_format()

    print("Type the title of an article to get the nearest neighbors. Type q to quit.")
    while (query := input("Article: ")) != 'q':
        article_idx = titles.get(query)

        if article_idx is None:
            print("No article found with that title. Try again.")
        else:
            posterior = Normal(loc=posteriors.loc[article_idx], scale=posteriors.scale[article_idx])

            print("\nL2 distance of means:")
            dists = torch.sum((posterior.mean - posteriors.mean) ** 2, dim=-1)
            dists, hit_indices = dists.topk(10, largest=False)

            # HF docs guarantee this will return a dictionary when passed a NumPy array like this
            hits = cast(Dict[str, List[str]], dataset[hit_indices.cpu().numpy()])

            max_len = max(len(x) for x in hits['title'])
            for dist, title in zip(dists, hits['title']):
                print(title + " " * (max_len - len(title)) + f" - {dist}")

            print("\nCosine similarity:")
            affinities = F.cosine_similarity(posterior.mean[None], posteriors.mean).squeeze()
            dists, hit_indices = affinities.topk(10, largest=True)
            hits = cast(Dict[str, List[str]], dataset[hit_indices.cpu().numpy()])

            max_len = max(len(x) for x in hits['title'])
            for dist, title in zip(dists, hits['title']):
                print(title + " " * (max_len - len(title)) + f" - {dist}")

            print("\nKL divergence:")
            kls = kl_divergence(posterior, posteriors).sum(dim=-1)
            dists, hit_indices = kls.topk(10, largest=False)
            dists = dists.cpu().numpy()
            hits = cast(Dict[str, List[str]], dataset[hit_indices.cpu().numpy()])

            max_len = max(len(x) for x in hits['title'])
            for dist, title in zip(dists, hits['title']):
                print(title + " " * (max_len - len(title)) + f" - {dist}")

            print('\n')


if __name__ == "__main__":
    main(sys.argv)
