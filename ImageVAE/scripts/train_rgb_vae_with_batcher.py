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

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from src.rgb_vae import ConvVAE, reset_graph
#from rgbd_vae import ConvVAE, reset_graph
#from vae128 import ConvVAE, reset_graph

#import sys
#sys.path.append('../tools')
#from rgb_crop_batcher import rgb_crop_batcher


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

    # Hyperparameters for ConvVAE
    z_size=16
    batch_size=96
    learning_rate=0.0001
    kl_tolerance=0.05

    # Parameters for training
    NUM_EPOCH = 100000

    model_save_path = "../../models"
    if not os.path.exists(model_save_path):
      os.makedirs(model_save_path)

    '''
    batcher = rgb_crop_batcher(args["list"], batch_size, 64)
    test_batch = batcher.make_batch()
    '''
    vis = test_batch[0].copy()
    cv2.imshow("rgb",vis[:,:,:])
    cv2.waitKey(100)
    #cv2.destroyWindow('rgb')

    # split into batches:
    num_batches = int(np.floor(len(batcher.imgs)/batch_size))
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

    threads = [threading.Thread(target=batcher.make_batches, daemon=True) for x in range(3)]
    for t in threads:
        t.start()

    # train loop:
    print("train", "step", "loss", "recon_loss", "kl_loss")
    for epoch in range(NUM_EPOCH):
      #np.random.shuffle(dataset)
        for idx in range(num_batches):
            batcher.lock.acquire()
            if len(batcher.batches):
                #train_batch = batcher.make_batch()
                train_batch = batcher.batches.pop(-1)
                batcher.lock.release()
                #obs = np.array(batcher.batch/255.0, dtype='f')
                #obs = np.array(batcher.batch).astype(np.float)/255.0
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
                        if (train_step % 10000 == 0):
                            vae.save_json("../../models/vae_{}.json".format(train_step))
                            # show reconstruction example
                            test_vae.load_json("../../models/vae_{}.json".format(train_step))
                            z = test_vae.encode(test_batch)
                            rec = test_vae.decode(z)
                            #print(rec)
                            #for img_idx, img in enumerate(test_batch):
                            vis = np.concatenate((vis, rec[0]), axis=1)
                            cv2.imshow("org vs. rec",cv2.resize(vis,(0,0),fx=4.0,fy=4.0))
                    key = cv2.waitKey(1) & 0xFF
            else:
                batcher.lock.release()
                #key = cv2.waitKey(200) & 0xFF
                time.sleep(0.2)

            if key == 27:
                batcher.running = False
                break
        if key == 27:
          batcher.running = False
          break
    # finished, final model:
    np.save("../../example.npy", vis)
    # finished, final model:
    vae.save_json("../../models/vae_final.json")
