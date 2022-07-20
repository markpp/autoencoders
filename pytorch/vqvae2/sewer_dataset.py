import torch
import torchvision

import os
from glob import glob
import cv2
import random
import numpy as np

class SewerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms, n_channels=3):
        self.image_list = sorted([y for y in glob(os.path.join(root_dir, '*.png'))])
        if not len(self.image_list)>0:
            print("did not find any files")
        self.n_channels = n_channels
        self.transforms = transforms

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        if self.n_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])

        if self.transforms:
            sample = self.transforms(**{'image':img})
            x = sample["image"]

        x = x.transpose((2, 0, 1))
        return torch.as_tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.image_list)
