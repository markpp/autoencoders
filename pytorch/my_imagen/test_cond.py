# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

import sys
sys.path.insert(0,'../imagen-pytorch')

if __name__ == "__main__":
    """
    Trains an autoencoder.

    Command:
        python train.py
    """


    from imagen_pytorch import Unet, Imagen, ImagenTrainer

    unet = Unet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    # imagen, which contains the unet above

    imagen = Imagen(
        condition_on_text = True,  # this must be set to False for unconditional Imagen
        unets = unet,
        channels = 3,
        image_sizes = 128,
        timesteps = 400
    )

    trainer = ImagenTrainer(imagen=imagen).cuda()

    trainer.load('checkpoint.pt')

    texts = ['spray',
             'spray',
             'spray',
             'don\'t',
             'dont',
             'dont',]

    images = trainer.sample(batch_size = len(texts), texts = texts, cond_scale = 3.)

    for i, image in enumerate(images):
        print(image.shape)
        cv2.imwrite('output/{}_gen_{}.jpg'.format(texts[i],i), (image.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
