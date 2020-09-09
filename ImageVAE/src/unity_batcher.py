'''
Data sampling/augmentation class that will produce a batch for VAE
training from a list of image and points of interest
'''
import numpy as np
import argparse
import os
import cv2
import random
from pyntcloud import PyntCloud
import json
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

decodeDot = [1.0, 1/255.0, 1/65025.0, 1/16581375.0]

class unity_batcher:
    def __init__(self,path_list,batch_size,crop_size,reload_samples=False):
        self.samples = []
        self.centers = []
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.obs = []
        self.batch_y = []

        if reload_samples:
            with open(path_list) as f:
                depth_paths = f.read().splitlines()[:]
                for count, depth_path in enumerate(depth_paths):
                    print('loading sample {}/{}'.format(count+1,len(depth_paths)),end='\r')
                    # Load depth
                    enc = cv2.imread(depth_path, -1)/255.0
                    depth = self.create_depth(enc)
                    # load color
                    color_path = depth_path[:-9]+"rgb.jpg"
                    color = cv2.imread(color_path)/255.0

                    #np_orig = np.abs(np.nan_to_num(cloud.points.values[:CROP_W*CROP_H,:3]))

                    # merge
                    np_sample = np.dstack((color,depth))
                    self.samples.append(np_sample)

                    # Load anno
                    pos_path = depth_path[:-9] + "pos.txt"
                    for line in open(pos_path):
                        center_ = line.split(" ")
                        center = (int(center_[1]),int(center_[2]))

                    self.centers.append(center)
            np.save("obs_array.npy", np.array(self.samples))
            np.save("labels_array.npy", np.array(self.centers))
            print('done loading {} samples from files'.format(len(depth_paths)))
        else:
            self.samples = np.load("obs_array.npy")
            self.centers = np.load("labels_array.npy")
            print('done loading {} samples from .npy'.format(self.samples.shape))

    def enc2float(self,enc):
        return np.dot(enc,decodeDot)

    def create_depth(self,enc):
        # build depth image
        out = np.zeros((424,512), np.float32)
        for y in range(424):
            for x in range(512):
                out[y,x] = self.enc2float(enc[y,x])

        return out

    def crop_sanity_correction(self,x_min,y_min,size):
        if x_min < 0:
            x_min = 0
        if x_min+size > 511:
            x_min = 511-size
        if y_min < 0:
            y_min = 0
        if y_min+size > 423:
            y_min = 423-size
        return (x_min,y_min)

    def augment_sample(self,sample,center):
        x, y = center

        # Augmentation
        x = x + np.random.randint(-20,20)
        y = y + np.random.randint(-20,20)

        # cropping
        x_min = x - self.crop_size//2
        y_min = y - self.crop_size//2
        x_min, y_min = self.crop_sanity_correction(x_min, y_min, self.crop_size)
        crop = sample[y_min:y_min+self.crop_size,x_min:x_min+self.crop_size,:3]

        crop = cv2.resize(crop, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        self.obs.append(crop)

    def make_batch(self):
        while len(self.obs) < self.batch_size:
            idx = random.randint(0,len(self.samples)-1)
            self.augment_sample(self.samples[idx], self.centers[idx])

'''
def show_example_from_batch(self,idx):
    tmp = self.imgs[idx].copy()
    p1, p2 = self.roi_from_poi(self.pois[idx][0])
    cv2.rectangle(tmp, p1,p2,(0,255,0),1)
    cv2.imshow("example",tmp)

def plt_im_plot(self,np_pc):
    plt.imshow(np_pc)
    plt.show()
'''
if __name__ == "__main__":
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="list of encoded depth files")
    args = vars(ap.parse_args())

    ub = unity_batcher(args["list"], 100, 128, reload_samples=False) # list, batch size, crop size
    ub.make_batch()

    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)

    while(True):
        ran_idx = random.randint(0,ub.batch_size-1)
        example_x = ub.obs[ran_idx]
        print("red min: {}, max: {}".format(np.min(example_x[:,:,0]),np.max(example_x[:,:,0])))
        print("green min: {}, max: {}".format(np.min(example_x[:,:,1]),np.max(example_x[:,:,1])))
        print("blue min: {}, max: {}".format(np.min(example_x[:,:,2]),np.max(example_x[:,:,2])))
        print("depth min: {}, max: {}".format(np.min(example_x[:,:,3]),np.max(example_x[:,:,3])))
        print("sample {}, {}".format(ran_idx,example_x.shape))

        cv2.imshow("rgb",example_x[:,:,:3]) # show random example from batch
        cv2.imshow("depth",example_x[:,:,3]) # show random example from batch

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
