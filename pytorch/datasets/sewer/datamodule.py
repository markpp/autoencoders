import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os, sys
from glob import glob
import cv2

import albumentations as Augment
from albumentations.pytorch.transforms import ToTensor

import torchvision.transforms as transforms

def basic_transforms(img_height, img_width, image_pad=0):
    return Augment.Compose([#Augment.ToGray(p=1.0),
                            Augment.SmallestMaxSize(max_size=img_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                            #Augment.Resize(img_height+image_pad, img_width+image_pad, interpolation=cv2.INTER_LINEAR, always_apply=True),
                            Augment.RandomCrop(img_height, img_width, always_apply=True),
                            #Augment.HorizontalFlip(p=0.5),
                            #Augment.RandomBrightnessContrast(p=0.5),
                            ])#ToTensor()

def extra_transforms():
    return Augment.Compose([Augment.GaussNoise(p=0.75),
                            Augment.CoarseDropout(p=0.5),])

sys.path.append('../')
from sewer.dataset import SewerDataset

class SewerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, image_size, img_channels=3, n_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.img_channels = img_channels
        self.n_workers = n_workers
        '''
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(image_size+image_pad),
                transforms.RandomCrop(image_size),
                #transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.noise_transform = transforms.Compose(
            [
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                #transforms.Grayscale(),
            ]
        )
        '''
    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = SewerDataset(os.path.join(self.data_dir,'train'),
                                           transform=basic_transforms(img_height=self.image_size,
                                                                      img_width=self.image_size),
                                           img_channels=self.img_channels
                                           )#noise_transform=extra_transforms())
            self.data_val = SewerDataset(os.path.join(self.data_dir,'val'),
                                         transform=basic_transforms(self.image_size,self.image_size),
                                         img_channels=self.img_channels)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def n_training(self):
        return len(self.data_train)

if __name__ == '__main__':

    dm = SewerDataModule(data_dir='/home/datasets/sewer/imgs',
                         batch_size=16,
                         image_size=256)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.train_dataloader()):
        if len(batch) == 2:
            imgs, imgs_ = batch
            for img, img_ in zip(imgs,imgs_):
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                img_ = img_.mul(255).permute(1, 2, 0).byte().numpy()
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                filename = "id-{}_.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img_)
                sample_idx = sample_idx + 1
            if batch_id > 1:
                break
        else:
            imgs = batch
            for img in imgs:
                img = img.mul(255).permute(1, 2, 0).byte().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                output_dir = os.path.join(output_root,str(batch_id).zfill(6))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = "id-{}.png".format(str(sample_idx).zfill(6))
                cv2.imwrite(os.path.join(output_dir,filename),img)
                sample_idx = sample_idx + 1
            if batch_id > 1:
                break
