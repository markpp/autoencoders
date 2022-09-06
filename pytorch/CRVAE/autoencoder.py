import os, sys

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crvae import CRVAE
from models import CNNVAE

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.learning_rate = hparams.learning_rate
        self.crvae = CRVAE() # CRVAE(gamma=hparams.gamma, beta_1=hparams.beta_1, beta_2=hparams.beta_2)
        self.model = CNNVAE(input_dim=hparams.input_dim, in_channels=hparams.in_channels, latent_dim=hparams.latent_dim)

    def forward(self, x):
        z = self.model.encode(x)
        return self.model.decode(z)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, x_aug, l = batch
        loss, log, recon, z = self.crvae.calculate_loss(self.model, x, x_aug)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, recon, "train_input_output_diff")

        #self.log('loss', loss, on_step=True, prog_bar=False)
        self.log_dict(log)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, x_aug, l = batch
        loss, log, recon, z = self.crvae.calculate_loss(self.model, x, x_aug)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, recon, "val_input_output_diff")
            self.logger.experiment.add_figure("val_latent_dist", self.plot_latent_distribution(z.cpu().numpy(), l.cpu().numpy()), self.current_epoch)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)

    ###
    def plot_latent_distribution(self, z, l):
        """
        Saves a plot of the main principal components of the latent space
        """
        PC = pca.fit_transform(z)
        df = pd.DataFrame(list(zip(PC[:,0], PC[:,1], l)), columns =['x', 'y', 'label'])
        ax = sns.scatterplot(data=df, x="x", y="y", hue="label")
        ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8))
        return ax.get_figure()

    def save_images(self, x, r, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n, normalize=True, value_range=(0,1))
        grid_middle = vutils.make_grid(r[:n,:,:,:], nrow=n, normalize=True, value_range=(0,1))
        diff = x - r
        diff = torch.abs(diff)
        grid_bottom = vutils.make_grid(diff[:n,:,:,:], nrow=n, normalize=True, value_range=(0,1))
        grid = torch.cat((grid_top, grid_middle, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
