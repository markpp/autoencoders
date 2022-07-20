import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os, sys
from glob import glob
import cv2
from PIL import Image

sys.path.append('../')
from sewer.dataset import SewerDataset

class SewerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                #transforms.Grayscale(),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = SewerDataset(os.path.join(self.data_dir,'Train'), transform=self.transform)
            self.data_val = SewerDataset(os.path.join(self.data_dir,'Val'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':

    dm = SewerDataModule(data_dir='/home/markpp/datasets/sewer/',
                         batch_size=32,
                         image_size=64)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if not os.path.exists(output_root):
        #shutil.rmtree(output_root)
        os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        imgs = batch
        for img in imgs:
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
        if batch_id > 0:
            break
