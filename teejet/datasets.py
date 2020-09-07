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

'''
def draw_line():

    offset = 20

    a, b = label[:2]
    x, y = self.crop_size//2, self.crop_size//2
    left_x = x + offset
    left_y = int(a * left_x + b)
    crop = cv2.circle(crop, (left_x, left_y), 2, (0,255,0), 3)
    crop = cv2.line(crop,
                   (x, y),
                   (left_x, left_y),
                   (0,255,0), 1)


    a, b = label[2:4]
    x, y = self.crop_size//2, self.crop_size//2
    right_x = x + offset #tmp.shape[1]
    right_y = int(a * right_x + b)
    crop = cv2.circle(crop, (right_x, right_y), 2, (0,0,255), 3)
    crop = cv2.line(crop,
                   (x, y),
                   (right_x, right_y),
                   (0,0,255), 1)
'''

class SprayDataset(torch.utils.data.Dataset):
    def __init__(self, list_path, crop_size=512):
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
        tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        return Compose(tfms)

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
        img = cv2.warpAffine(img, M, (image_h, image_w))

        # rotate points
        # method 1
        top = M.dot(np.array((top_x, top_y, 1)))
        top_x_r, top_y_r = int(top[0]), int(top[1])
        # method 2
        #radian = math.radians(angle)
        #top_x_r = int(cX + math.cos(radian) * (top_x - cX) - math.sin(radian) * (top_y - cY))
        #top_y_r = int(cY + math.sin(radian) * (top_x - cX) + math.cos(radian) * (top_y - cY))

        # crop around new rotated top point
        x = max(0, top_x_r - self.crop_size//2 + random.randint(-20,20))
        y = max(0, top_y_r - self.crop_size//2 + random.randint(-20,20))
        w, h = self.crop_size, self.crop_size
        crop = img[y:y+h, x:x+w]

        if show:
            # draw new top point in crop
            crop = cv2.circle(crop, (top_x_r-x, top_y_r-y), 2, (0,0,255), 3)

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
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        img = img.float()

        return img, img#, target

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    dataset = SprayDataset('/home/markpp/datasets/teejet/iphone_data/val.txt')


    print(dataset[0])
