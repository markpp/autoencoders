import os
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms.transforms import RandomCrop, ToTensor, Normalize, Compose

import cv2
import random
import math

from views import views

class HarborDataset(torch.utils.data.Dataset):
    def __init__(self, list_path, crop_size=64):
        self.crop_size = crop_size
        self.transforms = self.get_transform()

        with open(list_path) as f:
            self.image_list = f.read().splitlines()

    def load_sample(self, image_path):

        #img = Image.open(self.img_list[idx]).convert("RGB")

        #read flow image
        #flow_x = cv2.imread(image_path, -1)

        thermal = cv2.imread(image_path)[:,:,0]
        self.image_h, self.image_w = thermal.shape

        return thermal, image_path.split('/')[-3]

    def get_transform(self):
        tfms = []
        tfms.append(ToTensor())
        return Compose(tfms)

    def crop(self, img, view, show=False):

        x, y = views[view][0]['x'], views[view][0]['y']
        w, h = self.crop_size, self.crop_size
        crop = img[y:y+h, x:x+w]

        return crop

    def __getitem__(self, idx):
        img, view = self.load_sample(self.image_list[idx])
        img = self.crop(img, view)

        #img = self.transforms(img)

        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = img.unsqueeze(0)
        return img, img

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    dataset = HarborDataset('/home/markpp/datasets/harbour_frames/2/view1.txt')

    input, ref = dataset[0]


    input = input.mul(255).byte().numpy()
    cv2.imwrite("input.png",input)
