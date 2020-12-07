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
from celeba.dataset import CelebaDataset

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensor

def basic_transforms(img_height, img_width, image_pad=0):
    return Augment.Compose([#Augment.ToGray(p=1.0),
                            Augment.Resize(img_height+image_pad, img_width+image_pad, interpolation=cv2.INTER_NEAREST, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            Augment.HorizontalFlip(p=0.5),
                            Augment.RandomBrightnessContrast(p=1.0),
                            ])#ToTensor()

def extra_transforms():
    return Augment.Compose([Augment.GaussNoise(p=0.75),
                            Augment.CoarseDropout(p=0.5),])

class CelebaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        '''
        self.transform = transforms.Compose(
            [
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                #transforms.RandomCrop(image_size),
                #transforms.Grayscale(),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
        '''
    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = CelebaDataset(os.path.join(self.data_dir,'train'),
                                            transform=basic_transforms(img_height=self.image_size,
                                                                      img_width=self.image_size,
                                                                      image_pad=0),
                                           )#noise_transform=extra_transforms())
            self.data_val = CelebaDataset(os.path.join(self.data_dir,'val'),
                                          transform=basic_transforms(self.image_size,self.image_size))
            #self.data_train = CelebaDataset(os.path.join(self.data_dir,'train'), transform=self.transform)
            #self.data_val = CelebaDataset(os.path.join(self.data_dir,'val'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':

    dm = CelebaDataModule(data_dir='/home/markpp/datasets/celeba/',
                          batch_size=16,
                          image_size=64)

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
