import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet

from data_loader import prepare_data_from_folder, balanced_dataloader, standard_dataloader

class LightningDetector(pl.LightningModule):

    def __init__(self, hparams, pretrained=False):
      super().__init__()

      self.hparams = hparams

      # init model
      if pretrained:
          self.model = EfficientNet.from_pretrained(self.hparams.net)
      else:
          self.model = EfficientNet.from_name(self.hparams.net)

      self.model._fc = nn.Linear(self.model._fc.in_features, self.hparams.n_classes)

      self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    # create folds from training set
    def prepare_data(self):
        # split the dataset in train and test set
        self.data_train = prepare_data_from_folder(self.hparams.train_list, mode="train")
        self.data_val = prepare_data_from_folder('/home/markpp/datasets/bo/val/240x2', mode="val")

    def train_dataloader(self):
        return balanced_dataloader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        return balanced_dataloader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers) # change to standard dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        #loss = F.binary_cross_entropy_with_logits(output, target)
        loss = self.criterion(output, target)
        #tensorboard_logs = {'loss': loss}
        #return {'loss': loss, 'log': tensorboard_logs}
        return {'loss': loss} # somehow becomes batch_loss

    def training_epoch_end(self, outputs):
        epoch_loss = torch.stack([x['batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'epoch_loss': epoch_loss}
        return {'epoch_loss': epoch_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        batch_val_loss = self.criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        batch_val_correct = pred.eq(target.view_as(pred)).sum().item()/self.hparams.batch_size
        return {'batch_val_loss': batch_val_loss, 'batch_val_correct': batch_val_correct}#, 'target': target.cpu(), 'probs': output.cpu()}

    def validation_epoch_end(self, outputs):
        epoch_val_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        epoch_val_correct = np.stack([x['batch_val_correct'] for x in outputs]).mean()
        tensorboard_logs = {'epoch_val_loss': epoch_val_loss,
                            'epoch_val_correct': epoch_val_correct}

        return {'epoch_val_loss': epoch_val_loss, 'log': tensorboard_logs}
