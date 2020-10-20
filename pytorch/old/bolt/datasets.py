import os
import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import random
import math

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


class SprayDataModule(pl.LightningDataModule):

    def __init__(self, dataset_parms, trainer_parms, model_params, data_dir = './'):
        super().__init__()
        self.dataset_parms = dataset_parms
        self.trainer_parms = trainer_parms

        self.dims = (model_params['in_channels'], dataset_parms['img_size'], dataset_parms['img_size'])
        #self._has_prepared_data = True
        #self._has_setup_fit = True

        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        print("prepare_data")
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
        if stage == 'fit' or stage is None:
            self.data_train = SprayDataset(self.dataset_parms['train_list'],
                                           crop_size=self.dataset_parms['img_size'])

            self.data_val = SprayDataset(self.dataset_parms['val_list'],
                                         crop_size=self.dataset_parms['img_size'])
        '''
         # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.dataset_parms['batch_size'],
                          shuffle=True,
                          num_workers=self.trainer_parms['n_workers'])

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.dataset_parms['batch_size'],
                          shuffle=False,
                          num_workers=self.trainer_parms['n_workers'],
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.dataset_parms['batch_size'],
                          shuffle=False,
                          num_workers=self.trainer_parms['n_workers'],
                          pin_memory=True)

class SprayDataset(torch.utils.data.Dataset):
    def __init__(self, list_path, crop_size=128):
        self.crop_size = crop_size
        self.transforms = self.get_transform()

        with open(list_path) as f:
            self.label_list = f.read().splitlines()

    def load_sample(self, label_path):

        with open(label_path) as f:
            line = f.read().splitlines()[0]
            values = []
            #point_t, point_l, point_r, line_l, line_r
            for set in line.rstrip().split(';'):
                # x,y or a,b
                for str_val in set.split(':')[1].split(','):
                    values.append(float(str_val))
        label = values[:]

        image_path = label_path.replace('txt','jpg')
        #img = Image.open(self.img_list[idx]).convert("RGB")
        img = cv2.imread(image_path)
        self.image_h, self.image_w, _ = img.shape
        img = cv2.cvtColor(img[:self.image_h,:self.image_h], cv2.COLOR_BGR2RGB)

        return img, label

    def get_transform(self):
        tfms = []
        tfms.append(transforms.ToTensor())
        #tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        return transforms.Compose(tfms)

    def rot_and_crop(self, img, values, show=False):
        image_h, image_w, _ = img.shape
        cX, cY = image_w//2, image_h//2

        angle = random.randint(0,180)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        # compute the new bounding dimensions of the image
        #cos = np.abs(M[0, 0])
        #sin = np.abs(M[0, 1])
        #nW = int((image_h * sin) + (image_w * cos))
        #nH = int((image_h * cos) + (image_w * sin))

        # adjust the rotation matrix to take into account translation
        #M[0, 2] += (nW / 2) - cX
        #M[1, 2] += (nH / 2) - cY

        # original
        top_x, top_y = int(values[0]*self.image_w), int(values[1]*self.image_h)
        if show:
            img = cv2.circle(img, (top_x, top_y), 2, (0,255,255), 5)

        # rotate image
        img = cv2.warpAffine(img, M, img.shape[:2])

        # rotate points
        # method 1
        top = M.dot(np.array((top_x, top_y, 1)))
        top_x, top_y = int(top[0]), int(top[1])
        # method 2
        #radian = math.radians(angle)
        #top_x_r = int(cX + math.cos(radian) * (top_x - cX) - math.sin(radian) * (top_y - cY))
        #top_y_r = int(cY + math.sin(radian) * (top_x - cX) + math.cos(radian) * (top_y - cY))

        # crop around new rotated top point
        crop_offset = 20
        x = max(0, top_x - self.crop_size//2 + random.randint(-crop_offset,crop_offset))
        y = max(0, top_y - self.crop_size//2 + random.randint(-crop_offset,crop_offset))
        x = min(x, self.image_h - self.crop_size)
        y = min(y, self.image_h - self.crop_size)
        crop = img[y:y+self.crop_size, x:x+self.crop_size]

        if show:
            # draw new top point in crop
            crop = cv2.circle(crop, (top_x-x, top_y-y), 2, (0,0,255), 3)

        #left_x, left_y = int(values[2]*image_w), int(values[3]*image_h)
        #crop = cv2.circle(crop, (left_x-x, left_y-y), 2, (0,0,255), 3)

        #right_x, right_y = int(values[4]*image_w), int(values[5]*image_h)
        #crop = cv2.circle(crop, (right_x-x, right_y-y), 2, (0,0,255), 3)

        if show:
            cv2.imshow("crop", crop)
            cv2.waitKey()

        return crop

    def __getitem__(self, idx):
        img, target = self.load_sample(self.label_list[idx])
        img = self.rot_and_crop(img, target)

        #img = self.transforms(img)

        img = img.transpose((2, 0, 1))
        img = img / 255.0
        #img[0] = (img[0] - 0.485)/0.229
        #img[1] = (img[1] - 0.456)/0.224
        #img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        img = img.float()
        return img, img#, target

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    dataset = SprayDataset('/home/markpp/datasets/teejet/iphone_data/train_list.txt')


    print(dataset[10])
