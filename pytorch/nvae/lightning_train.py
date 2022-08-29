import argparse
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

from nvae.utils import add_sn
from nvae.lightning_model import NVAE


def create_datamodule(config):
    sys.path.append('../datasets')

    if config['exp_params']['dataset']=='sewer':
        from sewer.datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=config['exp_params']['data'],
                             batch_size=config['exp_params']['batch_size'],
                             image_size=config['exp_params']['image_size'],
                             img_channels=config['model_params']['img_channels'],
                             n_workers=config['trainer_params']['n_workers'])
        dm.setup()
        return dm
    elif config['exp_params']['dataset']=='celeba':
        from celeba.datamodule import CelebaDataModule
        dm = CelebaDataModule(data_dir=config['exp_params']['data'],
                              batch_size=config['exp_params']['batch_size'],
                              image_size=config['exp_params']['image_size'],
                              img_channels=config['model_params']['img_channels'],
                              n_workers=config['trainer_params']['n_workers'])
        dm.setup()
        return dm
    else:
        print("no such dataset: {}".format(config['exp_params']['data']))
        return None


if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str,
                    default='nvae.yaml', help="path to the config file")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dm = create_datamodule(config)
    if dm == None:
        print("failed to create datamodule")
        exit()

    #logger = loggers.TensorBoardLogger(config['logging_params']['save_dir'], name="{}_{}_{}".format(config['exp_params']['data'], config['exp_params']['image_size'],config['model_params']['in_channels']))
    model = NVAE(config=config, n_train=dm.n_training())
    # print detailed summary with estimated network size
    #summary(model, (config['model_params']['in_channels'], config['exp_params']['image_size'], config['exp_params']['image_size']), device="cpu")

    trainer = Trainer(gpus=config['trainer_params']['gpus'],
                      max_epochs=config['trainer_params']['max_epochs'],
                      #plugins='deepspeed',
                      #precision=16,
                      #stochastic_weight_avg=True
                      )

    trainer.fit(model, dm)
    #trainer.test(model)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(model.encoder, os.path.join(output_dir,"encoder.pt"))
    torch.save(model.decoder, os.path.join(output_dir,"decoder.pt"))
