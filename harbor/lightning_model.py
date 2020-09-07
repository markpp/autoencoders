import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from data_loader import prepare_data_from_list, standard_dataloader

import sys
sys.path.append('../')
from models import *


class LightningAutoencoder(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.model_parms = config['model_params']
        self.exp_parms = config['exp_params']
        self.trainer_parms = config['trainer_params']
        self.model = vae_models[config['model_params']['name']](**self.model_parms)

    def forward(self, x):
        return self.model(x)

    # create folds from training set
    def prepare_data(self):
        # split the dataset in train and test set
        self.data_train = prepare_data_from_list(self.exp_parms['train_list'], img_size=self.exp_parms['img_size'])
        self.data_val = prepare_data_from_list(self.exp_parms['val_list'], img_size=self.exp_parms['img_size'])

    def train_dataloader(self):
        self.num_train_imgs = len(self.data_train)
        return standard_dataloader(self.data_train, batch_size=self.exp_parms['batch_size'], shuffle=True, num_workers=self.trainer_parms['n_workers'])

    def val_dataloader(self):
        self.num_val_imgs = len(self.data_val)
        return standard_dataloader(self.data_val, batch_size=self.exp_parms['batch_size'], num_workers=self.trainer_parms['n_workers'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.exp_parms['LR'])
        return optimizer

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.model.loss_function(*output,
                                        M_N = self.exp_parms['batch_size']/ self.num_train_imgs,
                                        batch_idx = batch_idx)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rec_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD = torch.stack([x['KLD'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss,
                            'train_rec_loss': avg_rec_loss,
                            'train_KLD': avg_KLD}
        return {'train_loss': avg_loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        input, target = batch
        #print(input.shape)
        output = self.forward(input)
        loss = self.model.loss_function(*output,
                                        M_N = self.exp_parms['batch_size']/ self.num_val_imgs,
                                        batch_idx = batch_idx)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_rec_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        avg_KLD = torch.stack([x['KLD'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_rec_loss': avg_rec_loss,
                            'val_KLD': avg_KLD}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
