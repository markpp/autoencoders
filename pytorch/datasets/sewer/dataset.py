import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os
from glob import glob
import cv2
from PIL import Image

class SewerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.png'))])
        if not len(self.image_list)>0:
            print("did not find any files")

        self.transform = transform

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        #self.image_h, self.image_w, _ = img.shape
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])
        #img = img.transpose((2, 0, 1))
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_list)
