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
import threading
import time

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from src.rgb_vae import ConvVAE, reset_graph


if __name__ == "__main__":
  """
  Main function for executing the .py script.
  Command:
      -p path/<filename>.npy
  """
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-t", "--train", type=str,
                  help="list of train img files")
  ap.add_argument("-v", "--val", type=str,
                  help="list of val img files")
  args = vars(ap.parse_args())

  # Hyperparameters for ConvVAE
  z_size=16
  batch_size=64
  learning_rate=0.0001
  kl_tolerance=0.05

  # Parameters for training
  NUM_EPOCH = 100000

  model_save_path = "../models"
  if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

  # load train set
  train = []
  with open(args["train"]) as f:
    img_paths = f.read().splitlines()[:]
    for ip in img_paths:
      im = cv2.resize(cv2.imread(ip), (64,64), interpolation = cv2.INTER_AREA)
      train.append(im)
  n_train_samples = len(train)
  np_train = np.array(train).astype(np.float)/255.0

  # load val set
  val = []
  with open(args["train"]) as f:
    img_paths = f.read().splitlines()[:]
    for ip in img_paths:
        im = cv2.resize(cv2.imread(ip), (64,64), interpolation = cv2.INTER_AREA)
        val.append(im)
  np_val = np.array(val).astype(np.float)/255.0
  test_batch = np_val[:64]

  vis = test_batch[0]
  cv2.imshow("rgb",vis)
  cv2.waitKey(500)
  #cv2.destroyWindow('rgb')

  # split into batches:
  num_batches = int(np.floor(n_train_samples/batch_size))
  print("num_batches", num_batches)

  reset_graph()

  vae = ConvVAE(z_size=z_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                kl_tolerance=kl_tolerance,
                is_training=True,
                reuse=False,
                gpu_mode=True)

  test_vae = ConvVAE(z_size=z_size,
                     batch_size=1,
                     is_training=False,
                     reuse=False,
                     gpu_mode=False)

  time_step_ = []
  enc_loss_ = []
  rec_loss_ = []
  kl_loss_ = []

  key = 0
  # train loop:
  print("train", "step", "loss", "recon_loss", "kl_loss")
  for epoch in range(NUM_EPOCH):
    #np.random.shuffle(dataset)
    for idx in range(num_batches):
      sel = np.random.randint(0, n_train_samples, size=batch_size)
      train_batch = np_train[sel]
      feed = {vae.x: train_batch,}

      (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
      ], feed)
      if (train_step % 100 == 0):
        print("step", train_step, train_loss, r_loss, kl_loss)
        cv2.imshow("rgb",train_batch[0])
        if (train_step % 1000 == 0):
          #print("step", train_step, train_loss, r_loss, kl_loss)
          time_step_.append(train_step)
          enc_loss_.append(train_loss)
          rec_loss_.append(r_loss)
          kl_loss_.append(kl_loss)
          plt.plot(time_step_, enc_loss_, color='b')
          plt.plot(time_step_, rec_loss_, color='g')
          plt.plot(time_step_, kl_loss_, color='r')
          plt.yscale('log')
          plt.draw()
          plt.pause(0.01)
          #if (train_step % 2500 == 0):
          vae.save_json("../models/vae_{}.json".format(train_step))
          # show reconstruction example
          test_vae.load_json("../models/vae_{}.json".format(train_step))

          z = test_vae.encode(test_batch)
          rec = test_vae.decode(z)
          #print(rec)
          #for img_idx, img in enumerate(test_batch):
          vis = np.concatenate((vis, rec[0]), axis=1)
          cv2.imshow("org vs. rec",cv2.resize(vis,(0,0),fx=4.0,fy=4.0))
        key = cv2.waitKey(20) & 0xFF
      if key == 27:
        break
    if key == 27:
      break
  # finished, final model:
  np.save("../example.npy", vis)
  # finished, final model:
  vae.save_json("../models/vae_final.json")
