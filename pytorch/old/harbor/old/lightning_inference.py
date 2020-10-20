import argparse
import os
import yaml

import torch
import pytorch_lightning as pl

from datasets import HarborDataset
import cv2

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    ap.add_argument("-c", "--config", type=str,
                    default='configs/vae.yaml', help="path to the config file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    from lightning_model import LightningAutoencoder
    model = LightningAutoencoder(config)
    model.load_state_dict(torch.load(args['checkpoint']))
    model.eval()

    dataset = HarborDataset('/home/markpp/datasets/harbour_frames/2/view1_val.txt', crop_size=512)

    rec, a, b, c = model(dataset[0][0].unsqueeze(0))

    print("aa {}".format(rec.shape))


    from torchvision.transforms.transforms import Normalize

    rec = rec.mul(255).byte().numpy()
    cv2.imwrite("rec.png",rec)

    input = dataset[0][0][0].mul(255).byte().numpy()
    cv2.imwrite("input.png",input)
