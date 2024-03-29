import argparse
import os, sys
import cv2
import numpy as np
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
#from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder

from config import hparams

from torch.utils.data import DataLoader

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    # initialize datamodule
    from dataset import ImageFolderDataset
    from transforms import train_transforms, aug_transforms
    trainset = ImageFolderDataset(hparams.train_root, hparams.folders, train_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim), aug_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim))
    valset = ImageFolderDataset(hparams.val_root, hparams.folders, train_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim), aug_transforms(frame_size=hparams.image_size, crop_size=hparams.input_dim))

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)
    val_dataloader = DataLoader(valset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True, num_workers=hparams.n_workers)

    # initialize logging
    loggers = []
    from pytorch_lightning.loggers import TensorBoardLogger
    loggers.append(TensorBoardLogger(save_dir='logs/', name=hparams.name, default_hp_metric=False))

    model = Autoencoder(hparams)
    # print detailed summary with estimated network size
    #summary(model, (config['model_params']['in_channels'], config['exp_params']['image_size'], config['exp_params']['image_size']), device="cpu")
    trainer = Trainer(gpus=hparams.gpus, 
                      benchmark=True,
                      precision=16,
                      logger=loggers,
                      max_epochs=hparams.max_epochs)

    trainer.fit(model, train_dataloader, val_dataloader)
    #trainer.test(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #torch.save(model.net.state_dict(), os.path.join(output_dir,"net.pt"))
