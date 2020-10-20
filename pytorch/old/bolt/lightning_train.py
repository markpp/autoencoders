import argparse
import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import SprayDataModule


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
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)



    from pl_bolts.models.autoencoders import VAE

    from pl_bolts.datamodules import MNISTDataModule, ImagenetDataModule

    '''
    datamodule = SprayDataModule(dataset_parms=config['exp_params'],
                                 trainer_parms=config['trainer_params'],
                                 model_params=config['model_params'])
    model = VAE(hidden_dim = 128,
                latent_dim = 32,
                input_channels = config['model_params']['in_channels'],
                input_width = config['exp_params']['img_size'],
                input_height = config['exp_params']['img_size'],
                batch_size = config['exp_params']['batch_size'],
                learning_rate = config['exp_params']['LR'],
                datamodule = datamodule)
    '''
    datamodule = MNISTDataModule(data_dir='./')
    #datamodule = ImagenetDataModule(data_dir='./')

    model = VAE(datamodule=datamodule)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config['trainer_params']['max_epochs'],
                         checkpoint_callback=checkpoint_callback)#, profiler=True)

    trainer.fit(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir,"model.pt"))
