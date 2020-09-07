import argparse
import os
import yaml

import torch
import pytorch_lightning as pl

from datasets import SprayDataset
import cv2

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of RGB images.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    ap.add_argument("-c", "--config", type=str,
                    default='configs/vae.yaml', help="path to the config file")
    ap.add_argument("-d", "--data", type=str,
                    default='/home/markpp/datasets/teejet/iphone_data/val_list.txt', help="path to list of files")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    from lightning_model import LightningAutoencoder
    model = LightningAutoencoder(config)
    model.load_state_dict(torch.load(args['checkpoint']))

    #model = LightningAutoencoder(config).load_from_checkpoint(args['checkpoint'])

    model.eval()

    dataset = SprayDataset(args['data'], crop_size=64)

    rec, a, b, c = model(dataset[0][0].unsqueeze(0))

    print(rec.shape)

    from torchvision.transforms.transforms import Normalize

    unnormalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
    rec = unnormalize(rec[0])
    rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
    cv2.imwrite("rec.png",rec)

    input = unnormalize(dataset[0][0])
    input = input.mul(255).permute(1, 2, 0).byte().numpy()
    cv2.imwrite("input.png",input)
