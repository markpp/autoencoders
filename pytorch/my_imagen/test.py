# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

if __name__ == "__main__":
    """
    Trains an autoencoder.

    Command:
        python train.py
    """


    from imagen_pytorch import Unet, Imagen, ImagenTrainer
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen

    unet = Unet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    # imagen, which contains the unet above

    imagen = Imagen(
        condition_on_text = False,  # this must be set to False for unconditional Imagen
        unets = unet,
        channels = 1,
        image_sizes = 128,
        timesteps = 1000
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    trainer.load('checkpoint.pt')

    images = trainer.sample(batch_size = 16)

    for i, image in enumerate(images):
        print(image.shape)
        cv2.imwrite('output/gen_{}.jpg'.format(i), (image.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
