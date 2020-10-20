'''
Train VAE model on data created using an image crop batcher
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from vae import ConvVAE, reset_graph

import sys
sys.path.append('../tools')

from rgb_crop_batcher import rgb_crop_batcher


if __name__ == "__main__":
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--npy", type=str,
                    help="path to numpy archive")
    args = vars(ap.parse_args())


    crops = np.load(args["npy"])

    # parameters
    z_size=16
    batch_size, crop_size, _, _ = crops.shape

    print("batch_size {}".format(batch_size))
    print("crop_size {}".format(crop_size))

    cv2.imshow("crop",crops[0])
    cv2.waitKey(0)

    reset_graph()
    test_vae = ConvVAE(z_size=z_size,
                       batch_size=batch_size,
                       is_training=False,
                       reuse=False,
                       gpu_mode=True)

    # show reconstruction example
    test_vae.load_json("../../models/0/vae_{}.json".format(180000))
    z = test_vae.encode(crops)
    print(z.shape)

    rec = test_vae.decode(z)
    print(rec.shape)

    np.save("../../output/z_{}.npy".format(batch_size), z)
    np.save("../../output/rec_{}.npy".format(batch_size), rec)

    #for img_idx, img in enumerate(test_batch):
    vis = np.concatenate((crops[0], rec[0]), axis=1)
    cv2.imshow("org vs. rec",cv2.resize(vis,(0,0),fx=4.0,fy=4.0))

    key = cv2.waitKey(0) & 0xFF
