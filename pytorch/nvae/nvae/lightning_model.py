import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
from nvae.decoder import Decoder
from nvae.encoder import Encoder
from nvae.losses import recon, kl
from nvae.utils import reparameterize
from nvae.utils import add_sn
from nvae.losses import WarmupKLLoss

import robust_loss_pytorch
import numpy as np

class NVAE(pl.LightningModule):
    def __init__(self, config, n_train):
        super().__init__()
        self.lr = config['exp_params']['LR']
        self.encoder = Encoder(config['model_params']['latent_dim'], config['model_params']['img_channels'])
        self.decoder = Decoder(config['model_params']['latent_dim'], config['model_params']['img_channels'])

        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device="cpu")

        # apply Spectral Normalization
        self.apply(add_sn)

        self.warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                                 steps=[4500, 3000, 1500],
                                 M_N=config['exp_params']['batch_size'] / n_train,
                                 eta_M_N=5e-6,
                                 M_N_decay_step=36000)
        print('M_N=', self.warmup_kl.M_N, 'ETA_M_N=', self.warmup_kl.eta_M_N)

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        # Treat p(x|z) as discretized_mix_logistic distribution cost so much, this is an alternative way
        # witch combine multi distribution.
        #recon_loss = F.mse_loss(decoder_output, x)
        recon_loss = torch.mean(self.adaptive_loss.lossfun(
            torch.mean(F.binary_cross_entropy(decoder_output, x, reduction='none'), dim=[1, 2, 3])[:, None]))

        kl_loss = kl(mu, log_var)

        return decoder_output, recon_loss, [kl_loss] + losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch
        image_recon, recon_loss, kl_losses = self(x)
        kl_loss = self.warmup_kl.get_loss(self.global_step, kl_losses)
        loss = recon_loss + kl_loss
        self.log('recon_loss', recon_loss, on_step=True, prog_bar=False)
        self.log('kl_loss', kl_loss, on_step=True, prog_bar=False)
        self.log('loss', loss, on_step=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        #avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            z = torch.randn((16, 512, 4, 4)).to(self.device)
            gen_imgs, _ = self.decoder(z)
            #gen_imgs = gen_imgs.permute(0, 2, 3, 1).cpu() * 255

            self.save_images(gen_imgs,"latent_space_imgs")
    '''
    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        loss = F.mse_loss(output, x)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.save_images(x, output, "val_input_output")

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)
    '''

    def save_images(self, output, name, n=16):
        """
        Saves a plot of n images from input and output batch
        """
        # make grids and save to logger
        #grid_top = vutils.make_grid(x[:n,:,:,:], nrow=n)
        grid_bottom = vutils.make_grid(output[:n,:,:,:], nrow=n)
        #grid = torch.cat((grid_top, grid_bottom), 1)
        self.logger.experiment.add_image(name, grid_bottom, self.current_epoch)
