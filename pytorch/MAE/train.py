import os, sys
import numpy as np

import torch
import pytorch_lightning as pl

from config import hparams

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb
# https://github.com/IcarusWizard/MAE/blob/main/train_classifier.py
# https://github.com/FlyEgle/MAE-pytorch/blob/master/train_mae.py
if __name__ == "__main__":
    """
    Trains an autoencoder.

    Command:
        python train.py
    """

    # name the model using parameters from the configuration file
    model_name="model_{}x{}_{}".format(hparams.image_size,hparams.in_channel,hparams.encoder_emb_dim)
    print("Training model: {}".format(model_name))

    # initialize datamodule
    from dataset import ImageFolderDataset
    from transforms import train_transforms
    trainset = ImageFolderDataset(hparams.train_root, hparams.folders, train_transforms(frame_size=484, crop_size=hparams.image_size))#, mean=0.5, std=0.5, norm=True))
    valset = ImageFolderDataset(hparams.val_root, hparams.folders, train_transforms(frame_size=484, crop_size=hparams.image_size))#, mean=0.5, std=0.5, norm=True))

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)
    val_dataloader = DataLoader(valset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)

    # initialize model and training, validation and test code
    from autoencoder import Autoencoder
    ae = Autoencoder(hparams)

    # initialize logging
    loggers = []
    from pytorch_lightning.loggers import TensorBoardLogger
    loggers.append(TensorBoardLogger(save_dir='logs/', name=model_name, default_hp_metric=False))

    # initialize callbacks
    callbacks = []
    #from pytorch_lightning.callbacks import ModelCheckpoint
    #callbacks.append(ModelCheckpoint(dirpath=os.path.join("trained_models",model_name), monitor='avg_val_loss'))
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor())

    from pytorch_lightning import Trainer
    trainer = pl.Trainer(gpus=hparams.gpus,
                         benchmark=True,
                         precision=16,
                         max_epochs=hparams.max_epochs,
                         logger=loggers,
                         callbacks=callbacks)
                         
    trainer.fit(ae, train_dataloader, val_dataloader)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(ae.model, os.path.join(output_dir,"model.pt"))
