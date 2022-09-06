import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam

from vqvae2 import VQVAE2

class Autoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # network init
        self.net = VQVAE2(in_channels=config.in_channels,
                          hidden_channels=config.hidden_channels,
                          embed_dim=config.embed_dim,
                          nb_entries=config.nb_entries,
                          nb_levels=config.nb_levels,
                          scaling_rates=config.scaling_rates)

        # loss weighting
        self.beta =config.beta

        # optmizer parms
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2

    def forward(self, x):
        y, diffs, encs, decs = self.net(x)
        return y, diffs, encs, decs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

    def training_step(self, x, batch_idx):
        y, d, _, _ = self(x)
        r_loss, l_loss = y.sub(x).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        self.log('r_loss', r_loss, on_step=True, prog_bar=False)
        self.log('l_loss', l_loss, on_step=True, prog_bar=False)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, x, batch_idx):
        y, d, _, _ = self(x)
        r_loss, l_loss = y.sub(x).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        self.log('val_r_loss', r_loss, on_step=False, prog_bar=False)
        self.log('val_l_loss', l_loss, on_step=False, prog_bar=False)
        self.log('val_loss', loss, on_step=False, prog_bar=True)
        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, y, "val_input_output")
        return {"loss": loss}

    #def validation_epoch_end(self, outputs):
    #    avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #    self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)

    def save_images(self, x, y, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n, normalize=True, value_range=(0,1))
        grid_bottom = vutils.make_grid(y[:n,:,:,:], nrow=n, normalize=True, value_range=(0,1))
        grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
