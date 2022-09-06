# https://github.com/lucidrains/imagen-pytorch
import os, sys
import numpy as np

import torch
import cv2

if __name__ == "__main__":
    """
    Trains an Imagen.

    TODO: 
    Generate conditioned on product type, flow rate, defect type, and defect location.
    Later pressure, ?

    Command:
        python train.py
    """

    from imagen_pytorch import Unet, Imagen, ImagenTrainer
    from imagen_pytorch.data import Dataset

    # unets for unconditional imagen

    # https://github.com/lucidrains/imagen-pytorch/blob/797484d039ce8f62280f5e470ef17e3d291957f2/imagen_pytorch/imagen_pytorch.py#L1646
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
        fp16 = True,
        split_valid_from_train = True # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

    dataset = Dataset('/home/aau3090/Datasets/nozzle_dataset/spray_frames/J60-11003', image_size=128)
    #dataset = Dataset('/home/aau3090/Datasets/celeba/train', image_size=128)

    print("# {} sample in dataset".format(len(dataset)))

    cv2.imwrite('example.jpg', (dataset[0].numpy()*255).transpose((1, 2, 0)).astype(np.uint8))

    trainer.add_train_dataset(dataset, batch_size=64)

    max_batch_size = 4

    # working training loop
    for i in range(200000):
        loss = trainer.train_step(unet_number = 1, max_batch_size = max_batch_size)
        print(f'loss: {loss}')

        if not (i % 50):
            valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = max_batch_size)
            print(f'valid loss: {valid_loss}')

        if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
            images = trainer.sample(batch_size = 1, return_pil_images = True) # returns List[Image]
            images[0].save(f'samples/sample-{i}.png')

        if not (i % 1000) and trainer.is_main:
            trainer.save('checkpoint.pt')

    #trainer.load('./path/to/checkpoint.pt')