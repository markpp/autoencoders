import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchvision.utils as vutils

import pytorch_lightning as pl

import os, sys
from models import *


class LightningAutoencoder(pl.LightningModule):

    def __init__(self, config, n_train, n_val):
        super().__init__()
        self.model_parms = config['model_params']
        self.exp_parms = config['exp_params']
        self.trainer_parms = config['trainer_params']
        self.model = vae_models[config['model_params']['name']](**self.model_parms)

        self.n_train, self.n_val = n_train, n_val

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.exp_parms['LR'], weight_decay=self.exp_parms['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.exp_parms['scheduler_gamma'])
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input = batch
        output = self.forward(input)
        loss = self.model.loss_function(*output,
                                        M_N = self.exp_parms['batch_size']/self.n_train,
                                        batch_idx = batch_idx)
        self.log('train_loss', loss['loss'], on_step=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rec_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD = torch.stack([x['KLD'] for x in outputs]).mean()
        #self.log('train_loss', avg_loss)
        #self.log('train_rec_loss', avg_rec_loss)
        #self.log('train_KLD', avg_KLD)

    def validation_step(self, batch, batch_idx):
        input = batch
        output = self.forward(input)
        loss = self.model.loss_function(*output,
                                        M_N = self.exp_parms['batch_size']/self.n_val,
                                        batch_idx = batch_idx)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(output[1], output[0], "val_input_vs_reconstruction")

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rec_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD = torch.stack([x['KLD'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_rec_loss', avg_rec_loss)
        self.log('val_KLD', avg_KLD)

    def save_images(self, x, output, name, n=8):
        """
        Saves a plot of n images from input and output batch
        """

        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n)
        grid_bottom = vutils.make_grid(output[:n,:,:,:], nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
