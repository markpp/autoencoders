import os
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms.transforms import RandomCrop, ToTensor, Normalize, Compose

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import random
import math


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
        tfms.append(ToTensor())
        #tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        return Compose(tfms)

    def rot_and_crop(self, img, values, rot=False, show=False):
        image_h, image_w, _ = img.shape
        cX, cY = image_w//2, image_h//2

        # original
        top_x, top_y = int(values[0]*self.image_w), int(values[1]*self.image_h)
        if show:
            img = cv2.circle(img, (top_x, top_y), 2, (0,255,255), 5)

        if rot:
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
        crop_offset = 0
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
        return img#, img#, target

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    dataset = SprayDataset('/home/markpp/datasets/teejet/iphone_data/val_list.txt', crop_size=256)

    #for i in range(len(dataset)):
    for i, data in enumerate(dataset):
        img = data.mul(255).permute(1, 2, 0).byte().numpy()
        cv2.imwrite("output/val/{}.png".format(str(i).zfill(4)),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        #break
