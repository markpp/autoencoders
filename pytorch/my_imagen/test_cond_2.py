# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

import sys
sys.path.insert(0,'../imagen-pytorch')

from imagen_pytorch import Unet, Imagen, ImagenTrainer

if __name__ == "__main__":
    """
    Trains an autoencoder.

    Command:
        python train.py
    """

    unet = Unet(
        dim = 96,
        memory_efficient = False,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    srunet = Unet(
        dim = 64,
        memory_efficient = True,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unet above
    imagen = Imagen(
        condition_on_text = True,  # this must be set to False for unconditional Imagen
        unets = (unet, srunet),
        channels = 1,
        image_sizes = (128, 448),
        timesteps = 1000,
        cond_drop_prob = 0.1
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        fp16 = True,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    trainer.load('checkpoint_2.pt')

    texts = ['J60-11003',
             'J60-11003',
             'J60-11004',
             'J60-11004',
             'J60-11006',
             'J60-11006',]

    images = trainer.sample(batch_size = len(texts), texts = texts, cond_scale = 3.)

    for i, image in enumerate(images):
        print(image.shape)
        cv2.imwrite('output/{}_gen_{}.jpg'.format(texts[i],i), (image.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
