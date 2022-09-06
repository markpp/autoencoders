import os, sys
import numpy as np

import torch
import pytorch_lightning as pl

from config import hparams

if __name__ == "__main__":
    """
    Trains an autoencoder.

    Command:
        python train.py
    """

    # name the model using parameters from the configuration file
    model_name="model_{}_{}x{}_{}".format(hparams.image_size,hparams.input_dim,hparams.in_channels,hparams.latent_dim)
    print("Training model: {}".format(model_name))

    # initialize datamodule
    from dataset import ImageFolderDataset
    from transforms import train_transforms, aug_transforms
    trainset = ImageFolderDataset(hparams.train_root, hparams.folders, train_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim), aug_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim))
    valset = ImageFolderDataset(hparams.val_root, hparams.folders, train_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim), aug_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim))

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)
    val_dataloader = DataLoader(valset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)

    # initialize model and training, validation and test code
    from autoencoder import Autoencoder
    model = Autoencoder(hparams)

    # initialize logging
    loggers = []
    from pytorch_lightning.loggers import TensorBoardLogger
    loggers.append(TensorBoardLogger(save_dir='logs/', name=model_name, default_hp_metric=False))

    # initialize callbacks
    callbacks = []
    #from pytorch_lightning.callbacks import ModelCheckpoint
    #callbacks.append(ModelCheckpoint(dirpath=os.path.join("trained_models",model_name), monitor='avg_val_loss'))

    from pytorch_lightning import Trainer
    trainer = pl.Trainer(gpus=hparams.gpus,
                         benchmark=True,
                         #precision=16,
                         max_epochs=hparams.max_epochs,
                         logger=loggers,
                         callbacks=callbacks)
                         
    trainer.fit(model, train_dataloader, val_dataloader)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.encoder, os.path.join(output_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(output_dir,"decoder.pt"))
