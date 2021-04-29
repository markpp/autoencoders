import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os, sys
import cv2

import albumentations as Augment

def basic_transforms(img_height, img_width, img_channels=3):
    augmentations = []
    #if img_channels == 1:
    #    augmentations.append(Augment.ToGray(p=1.0))
    augmentations.append(Augment.Resize(img_height, img_width, interpolation=cv2.INTER_NEAREST, always_apply=True))
    #augmentations.append(Augment.HorizontalFlip(p=0.5))
    #augmentations.append(Augment.RandomBrightnessContrast(p=1.0))
    return Augment.Compose(augmentations)

sys.path.append('../')
from celeba.dataset import CelebaDataset

class CelebaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size,img_channels, n_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.img_channels = img_channels
        self.n_workers = n_workers
    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = CelebaDataset(os.path.join(self.data_dir,'train'),
                                            transform=basic_transforms(img_height=self.image_size,
                                                                       img_width=self.image_size,
                                                                       img_channels=self.img_channels),
                                           img_channels=self.img_channels,
                                           )#noise_transform=extra_transforms())
            self.data_val = CelebaDataset(os.path.join(self.data_dir,'val'),
                                          transform=basic_transforms(self.image_size,self.image_size,self.img_channels),
                                          img_channels=self.img_channels)
            #self.data_train = CelebaDataset(os.path.join(self.data_dir,'train'), transform=self.transform)
            #self.data_val = CelebaDataset(os.path.join(self.data_dir,'val'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def n_training(self):
        return len(self.data_train)

if __name__ == '__main__':

    dm = CelebaDataModule(data_dir='/home/datasets/celeba/imgs',
                          batch_size=16,
                          image_size=128,
                          img_channels=1)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        imgs = batch
        for img in imgs:
            print(img.shape)
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
        if batch_id > 1:
            break
