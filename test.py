from text_vae import *
import sys


def main(args):
    model_str, model_name = args[1:]
    path = get_checkpoint_path_for_name(model_str, model_name)

    if model_str == 'lstm-vae':
        model_class = LSTMVAE
    elif model_str == 'lstm-lm':
        model_class = LSTMLanguageModel
    elif model_str == 'transformer-lm':
        model_class = Transformer
    elif model_str == 'transformer-vae':
        model_class = TransformerVAE
    elif model_str == 'vq-vae':
        model_class = QuantizedVAE
    else:
        print(f"Unrecognized model type '{model_str}'.")
        return

    model = model_class.load_from_checkpoint(path)
    model.freeze()
    model.start_token = 2
    model.end_token = 3

    device = 'cuda:' + str(select_best_gpu())
    model = model.to(device)

    hparams = TextDataModuleHparams()
    data = TextDataModule(hparams)
    data.prepare_data()
    data.setup('test')

    dataloader = data.test_dataloader()
    pbar = tqdm(dataloader, desc="Testing", unit='batch')
    losses = []
    for i, batch in enumerate(pbar):
        nll = model.test_step({k: v.to(device) for k, v in batch.items()}, i).item()
        losses.append(nll)

        pbar.set_postfix(last=nll, avg=sum(losses) / len(losses))

    print("Average test loss: ", sum(losses) / len(losses))


if __name__ == "__main__":
    main(sys.argv)
