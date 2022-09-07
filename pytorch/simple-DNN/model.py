import pytorch_lightning as pl

from torch import nn, optim


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 392),  # 28*28/2 = 392
                nn.ReLU(),
                nn.Linear(392, 130),
                nn.ReLU(),
                nn.Linear(130, 10),
                nn.ReLU()               # Should this last activation exist?
        )
        self.decoder = nn.Sequential(
                nn.Linear(10, 130),
                nn.ReLU(),
                nn.Linear(130, 392),
                nn.ReLU(),
                nn.Linear(392, 28*28),
                nn.ReLU()
        )

    def training_step(self, batch, batch_idx):
        x, y  = batch
        x     = x.view(x.size(0), -1)
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        loss  = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        x     = x.view(x.size(0), -1)
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        loss  = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y  = batch
        x     = x.view(x.size(0), -1)
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        loss  = nn.functional.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

