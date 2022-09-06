# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

import sys
sys.path.insert(0,'../imagen-pytorch')

if __name__ == "__main__":
    """
    Trains an Imagen.

    TODO: 
    Generate conditioned on product type, flow rate, defect type, and defect location.
    Later pressure, ?

    Command:
        python train.py
    """
    image_size = 128

    from imagen_pytorch import Unet, Imagen, ImagenTrainer
    from data import Dataset

    # unets for unconditional imagen

    # https://github.com/lucidrains/imagen-pytorch/blob/797484d039ce8f62280f5e470ef17e3d291957f2/imagen_pytorch/imagen_pytorch.py#L1646
    unet = Unet(
        dim = 128,
        #text_embed_dim = 5,
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
        image_sizes = image_size,
        timesteps = 400 # 1000
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        #fp16 = True,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

    dataset = Dataset('/home/aau3090/Datasets/tmp/train_val/', image_size=image_size)
    #dataset = Dataset('/home/aau3090/Datasets/nozzle_dataset/spray_frames/', image_size=image_size)
    #dataset = Dataset('/home/aau3090/Datasets/celeba/train', image_size=image_size)

    print("# {} sample in dataset".format(len(dataset)))

    example = (dataset[0][0].numpy()*255).transpose((1, 2, 0)).astype(np.uint8)
    print("embedding: {}, mask {} ".format(dataset[0][1], dataset[0][2]))
    cv2.imwrite('example.jpg', example)

    trainer.add_train_dataset(dataset, batch_size=32)

    max_batch_size = 2

    from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME
    from functools import partial
    encode_text = partial(t5_encode_text, name = DEFAULT_T5_NAME)
    text_embeds, text_masks = encode_text(['spray'], return_attn_mask = True)


    # working training loop
    for i in range(200000):
        loss = trainer.train_step(unet_number = 1, max_batch_size = max_batch_size)
        print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = max_batch_size)
            print(f'valid loss: {valid_loss}')

        if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
            images = trainer.sample(batch_size = 1, return_pil_images = True, text_embeds=text_embeds, text_masks=text_masks) # returns List[Image]
            images[0].save(f'samples/sample-{i}.png')

        if not (i % 1000) and trainer.is_main:
            trainer.save('checkpoint.pt')

    #trainer.load('./path/to/checkpoint.pt')