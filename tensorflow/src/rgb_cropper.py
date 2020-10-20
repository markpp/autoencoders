'''
Data sampling/augmentation class that will produce a batch for VAE
training from a list of image and points of interest
'''
import numpy as np
import argparse
import os
import cv2
import time


class rgb_cropper:
    def __init__(self,list_path,crop_size):
        self.names = []
        self.crops = []

        with open(list_path) as f:
            img_paths = f.read().splitlines()[:]
            for count, ip in enumerate(img_paths):
                print('loading img {}/{}'.format(count+1,len(img_paths)),end='\r')
                # Load images
                bgr = cv2.imread(ip)

                # Load annotations
                poi_path = ip[:-3] + "csv"
                poi = []
                with open(poi_path) as poi_file:
                    points = []
                    for poi in poi_file.read().splitlines():
                        x, y = poi.split(",")
                        crop = self.crop_sample(bgr,int(x),int(y),crop_size)
                        if len(crop) > 0:
                            self.names.append(ip.split("/")[-1])
                            self.crops.append(crop)
        num_names = len(self.names)
        num_crops = len(self.names)
        print('done loading {} imgs'.format(len(img_paths)))
        np.save("../../output/names_{}.npy".format(num_names), np.array(self.names))
        np.save("../../output/crops_{}.npy".format(num_crops), np.array(self.crops))


    def crop_sample(self,img,x,y,crop_size):
        width, height, _ = img.shape

        x_min = x-int(crop_size/2)
        x_max = x_min+crop_size
        y_min = y-int(crop_size/2)
        y_max = y_min+crop_size
        #print("x_min: {}, x_max {}, y_min {}, y_max {}".format(x_min,x_max,x_min,y_max))
        if(x_min >= 0 and x_max < width and y_min >= 0 and y_max < height):
            crop = img[y_min:y_max,x_min:x_max,:]
            w, h, _ = crop.shape
            if w != crop_size or h != crop_size:
                crop = cv2.resize(crop, (crop_size, crop_size))
            return crop.astype(np.float)/255.0
        return []


if __name__ == "__main__":
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="list of img files")
    args = vars(ap.parse_args())

    rc = rgb_cropper(args["list"], 64)
