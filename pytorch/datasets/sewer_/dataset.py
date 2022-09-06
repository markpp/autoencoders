import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2

class SewerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, noise_transform=None):
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.png'))])
        if not len(self.image_list)>0:
            print("did not find any files")
        self.transform = transform
        self.noise_transform = noise_transform

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])

        sample = {'image':img}
        if self.transform:
            sample = self.transform(**sample)
            img = sample["image"]
            if self.noise_transform:
                sample = self.noise_transform(**sample)
                noisy_img = sample["image"]
                img = img.transpose((2, 0, 1))#[:1,:,:]
                noisy_img = noisy_img.transpose((2, 0, 1))#[:1,:,:]
                return torch.as_tensor(img)/255.0, torch.as_tensor(noisy_img)/255.0
        img = img.transpose((2, 0, 1))
        return torch.as_tensor(img)/255.0

    def __len__(self):
        return len(self.image_list)
