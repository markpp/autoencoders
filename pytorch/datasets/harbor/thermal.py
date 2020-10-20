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
    def __init__(self, list_path, crop_size=64, crop_idx=0, USE_FLOW=False):
        self.crop_size = crop_size
        self.crop_idx = crop_idx
        self.transforms = self.get_transform()
        self.USE_FLOW = USE_FLOW

        with open(list_path) as f:
            self.image_list = f.read().splitlines()

    def load_sample(self, image_path):
        thermal = cv2.imread(image_path)
        self.image_h, self.image_w, _ = thermal.shape
        if self.USE_FLOW:
            thermal[:,:,1] = cv2.imread(image_path.replace('img_','flow_x_'), -1)
            thermal[:,:,2] = cv2.imread(image_path.replace('img_','flow_y_'), -1)
        else:
            thermal = thermal[:,:,0]
        return thermal, image_path.split('/')[-3]

    def crop(self, img, view, show=False):
        x, y = views[view][self.crop_idx]['x'], views[view][self.crop_idx]['y']
        w, h = self.crop_size, self.crop_size
        crop = img[y:y+h, x:x+w]
        return crop

    def __getitem__(self, idx):
        img, view = self.load_sample(self.image_list[idx])
        img = self.crop(img, view)
        img = img / 255.0
        img = torch.from_numpy(img)
        #img = img.float()
        if not self.flow:
            img = img.unsqueeze(0)
        return img, img

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    set = 'test'
    view = 'view1'

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    view_dir = os.path.join(output_dir,view)
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)

    for crop_idx in range(len(views[view])):
        crop_dir = os.path.join(view_dir,"crop{}".format(crop_idx))
        if not os.path.exists(crop_dir):
            os.mkdir(crop_dir)

        set_dir = os.path.join(crop_dir,set)
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)

        dataset = HarborDataset('/home/markpp/datasets/harbour_frames/2/{}_{}.txt'.format(set,view),
                                crop_size = 64,
                                crop_idx = crop_idx)

        print(len(dataset))
        for i, data in enumerate(dataset):
            input = data
            input = input.mul(255).byte().numpy()
            cv2.imwrite(os.path.join(set_dir,"{}.png".format(str(i).zfill(5))),input)
