import os, sys

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import Adam
import pytorch_lightning as pl
import einops

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

class Autoencoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.max_epochs = hparams.max_epochs
        self.warmup_epochs = hparams.warmup_epochs
        self.mask_ratio=hparams.mask_ratio
        self.base_learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay

        from model import MAE_ViT
        self.model = MAE_ViT(image_size=hparams.image_size,
                             in_channel=hparams.in_channel,
                             patch_size=hparams.patch_size,
                             emb_dim=hparams.encoder_emb_dim,
                             encoder_layer=hparams.encoder_layer,
                             encoder_head=hparams.encoder_head,
                             decoder_layer=hparams.decoder_layer,
                             decoder_head=hparams.decoder_head,
                             mask_ratio=hparams.mask_ratio)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_learning_rate * self.batch_size / 256, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        
        lr_func = lambda epoch: min((epoch + 1) / (self.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.max_epochs * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, l = batch
        predicted_img, mask  = self.model(x)
        loss = torch.mean((predicted_img - x) ** 2 * mask) / self.mask_ratio

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            tmp = einops.rearrange(predicted_img[:16,:,:,:], 'b c h w -> c h (b w)')
            self.logger.experiment.add_image("train_output", tmp, self.current_epoch)

        self.log('loss', loss)
        #self.log_dict(log)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, l = batch
        predicted_img, mask  = self.model(x)
        loss = torch.mean((predicted_img - x) ** 2 * mask) / self.mask_ratio

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            x, predicted_img, mask = x[:32], predicted_img[:32], mask[:32]
            tmp = torch.cat([x * (1 - mask), predicted_img * mask + x * (1 - mask), x], dim=0)
            tmp = einops.rearrange(tmp, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            self.logger.experiment.add_image("val_input_output_img", tmp, self.current_epoch)
            #self.logger.experiment.add_figure("val_latent_dist", self.plot_latent_distribution(z.cpu().numpy(), l.cpu().numpy()), self.current_epoch)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)

    ###
    '''
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
    '''
