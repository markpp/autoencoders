# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

import sys
sys.path.insert(0,'../imagen-pytorch')

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from data import Dataset


if __name__ == "__main__":
    """
    Trains an Imagen.

    TODO: 
    Generate conditioned on product type, flow rate, defect type, and defect location.
    Later pressure, ?

    Command:
        python train.py
    """

    steps = [2]

    # https://github.com/lucidrains/imagen-pytorch/blob/797484d039ce8f62280f5e470ef17e3d291957f2/imagen_pytorch/imagen_pytorch.py#L1646
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

    from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME
    from functools import partial
    encode_text = partial(t5_encode_text, name = DEFAULT_T5_NAME)
    text_embeds, text_masks = encode_text(['J60-11006'], return_attn_mask = True)

    data_init = False
    for step in steps:

        if step == 0:
            trainer.save('checkpoint.pt')
            continue
        
        if step == 1 or step == 2 and not data_init:
            dataset = Dataset('/home/aau3090/Datasets/nozzle_dataset/spray_frames/', image_size=448)
            trainer.add_train_dataset(dataset, batch_size=64)
            data_init = True

        if step == 1:
            trainer.load('./checkpoint.pt', noop_if_not_exist = True)
            #dataset = Dataset('/home/aau3090/Datasets/nozzle_dataset/spray_frames/', image_size=128)
            #trainer.add_train_dataset(dataset, batch_size=64)
            max_batch_size = 4

        elif step == 2:
            trainer.load('./checkpoint_1.pt', noop_if_not_exist = True)
            max_batch_size = 4

        print("# {} sample in dataset".format(len(dataset)))
        example = (dataset[0][0].numpy()*255).transpose((1, 2, 0)).astype(np.uint8)
        cv2.imwrite('example_{}.jpg'.format(step), example)

        # working training loop
        for i in range(4001):
            loss = trainer.train_step(unet_number = step, max_batch_size = max_batch_size)
            if not (i % 10):
                print(f'loss: {loss}')

            if not (i % 50):
                valid_loss = trainer.valid_step(unet_number = step, max_batch_size = max_batch_size)
                print(f'valid loss: {valid_loss}')

            if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
                images = trainer.sample(stop_at_unet_number = step, batch_size = 1, return_pil_images = True, text_embeds=text_embeds, text_masks=text_masks)
                images[0].save(f'samples/sample-{i}.png')

            if not (i % 100) and trainer.is_main:
                trainer.save('checkpoint_{}.pt'.format(step))