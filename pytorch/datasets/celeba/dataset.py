import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os
from glob import glob
import cv2
import numpy as np

class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, img_channels):
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.jpg'))])
        if not len(self.image_list)>0:
            print("did not find any files")
        self.img_channels = img_channels
        self.transform = transform

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        if self.img_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        if h > w:
            top_h = int((h - w) / 2)
            img = img[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            img = img[:, left_w:left_w + h]
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])

        sample = {'image':img}
        if self.transform:
            sample = self.transform(**sample)
            img = sample["image"]

        img = img.transpose((2, 0, 1))
        return torch.as_tensor(img)/255.0

    def __len__(self):
        return len(self.image_list)
