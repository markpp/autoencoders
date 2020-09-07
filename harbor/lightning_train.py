import argparse
import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='trained_models/'
)


if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str,
                    default='configs/vae.yaml', help="path to the config file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    from lightning_model import LightningAutoencoder
    model = LightningAutoencoder(config)

    trainer = pl.Trainer(gpus=1, max_epochs=config['trainer_params']['max_epochs'])#, checkpoint_callback=checkpoint_callback)#, profiler=True)

    trainer.fit(model)

    torch.save(model.state_dict(), "trained_models/model.pt")
