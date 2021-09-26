from sparse_vae import *
from datasets import Dataset
# from itertools import chain
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from tqdm import tqdm
import sys


def main(args):
    model_str, model_name = args[1:]
    save_path = Path.cwd() / 'sparse-vae-datasets' / 'latents' / model_str / model_name
    dataset = Dataset.load_from_disk(str(save_path))
    latents = np.array(dataset['latent'])

    try:
        from tsnecuda import TSNE
    except ImportError:
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise RuntimeError("Error: Either t-SNE-CUDA or sklearn must be installed for t-SNE plots")
        else:
            print("Warning: Couldn't import t-SNE-CUDA, using sklearn CPU implementation. This could take a while.")
            tsne = TSNE(n_jobs=min(20, cpu_count() - 1))
    else:
        tsne = TSNE(device=select_best_gpu(min_free_memory=4.0))

    print("Fitting t-SNE embedding...")
    embeddings = tsne.fit_transform(latents.squeeze())
    print("Done.")

    print("Plotting random subset of 1,000 points in monochrome")
    subset = np.random.choice(embeddings.shape[0], 1000, replace=False)
    plt.scatter(embeddings[subset, 0], embeddings[subset, 1])
    plt.savefig('sparse-vae-tsne.png')

    try:
        from gensim.models.ldamulticore import LdaMulticore
        from gensim.corpora import Dictionary
    except ImportError:
        print("Gensim isn't available, so we can't fit an LDA model to color the t-SNE plot")
        return

    dm = TextDataModule(TextDataModuleHparams())
    dm.prepare_data()
    vocab = dm.tokenizer.get_vocab()
    dataset = dm.dataset.shuffle()

    lda = LdaMulticore(num_topics=10, workers=min(10, cpu_count() - 1), id2word=dict(zip(vocab.values(), vocab.keys())))
    pbar = tqdm(desc='Fitting LDA model', total=len(dataset['train']), unit='doc')
    for start in range(0, len(dataset['train']), 2_000):
        batch = dataset['train'][start:start + 2_000]['text']
        bow = [[(tok, count) for tok, count in zip(*np.unique(doc, return_counts=True))] for doc in batch]
        lda.update(bow)
        pbar.update(2_000)

        if start % 20_000 == 0:
            loss = -lda.log_perplexity(bow).mean()  # noqa
            pbar.set_postfix(loss=loss)

    print(lda)
    lda.save('lda')


if __name__ == "__main__":
    main(sys.argv)
