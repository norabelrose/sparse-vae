from sparse_vae import *
from datasets import Dataset
from itertools import chain
import sys


def main(args):
    model_str, model_name = args[1:]
    model = load_checkpoint_for_name(model_str, model_name)
    model.freeze()

    device = 'cuda:' + str(select_best_gpu())
    model = model.to(device)

    hparams = TextDataModuleHparams()
    data = TextDataModule(hparams)
    data.prepare_data()
    data.setup('predict')

    dls = data.predict_dataloader()
    total = sum(len(dl) for dl in dls)
    pbar = tqdm(
        chain.from_iterable(dls), desc="Gathering latents", unit='batch', total=total
    )
    latents, scales, titles = [], [], []
    for i, batch in enumerate(pbar):
        q_of_z = model.predict({k: v.to(device) for k, v in batch.items() if isinstance(v, Tensor)}, i)
        mean, scale = q_of_z.mean, q_of_z.scale
        latents.extend(mean.cpu().numpy().squeeze())
        scales.extend(scale.cpu().numpy().squeeze())
        titles.extend([''.join(x) for x in batch['title']])
        if i >= total:
            pbar.close()
            break

    print("Saving to disk...")
    save_path = Path.cwd() / 'sparse-vae-datasets' / 'latents' / model_str / model_name
    dataset = Dataset.from_dict({'title': titles, 'latent': latents, 'scale': scales})
    dataset.save_to_disk(str(save_path))
    print("Done.")


if __name__ == "__main__":
    main(sys.argv)
