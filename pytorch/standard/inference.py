import argparse
import os
import yaml

import torch
import pytorch_lightning as pl
import cv2


class LightningAutoencoder(pl.LightningModule):

    def __init__(self, config, n_train, n_val):
        super().__init__()
        self.model_parms = config['model_params']
        self.exp_parms = config['exp_params']
        self.trainer_parms = config['trainer_params']

if __name__ == '__main__':
    """
    Trains an autoencoder from RGB images.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    ap.add_argument("-c", "--config", type=str,
                    default='config/vae_celeba.yaml', help="path to the config file")
    ap.add_argument("-i", "--image", type=str,
                    default='/home/markpp/datasets/celeba/val/000002.jpg', help="path image file")
    args = vars(ap.parse_args())

    #/home/markpp/datasets/celeba/val/000002.jpg
    #/home/markpp/datasets/celeba/train/010005.jpg
    #/home/markpp/datasets/sewer/test/00003557.png

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    from models import *
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.load_state_dict(torch.load(args['checkpoint']))
    model.eval()


    img = cv2.imread(args['image'])#[:575,:575]
    h,w = img.shape[:2]
    if h > w:
        scale = h/w
        img = cv2.resize(img, dsize=(64,int(64*scale)))
        offset = (int(64*scale) - 64) // 2
    else:
        scale = w/h
        img = cv2.resize(img, dsize=(int(64*scale),64))
        offset = (int(64*scale) - 64) // 2
    img = img[offset:offset+64,offset:offset+64]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.float() / 255.0

    out = model(img.unsqueeze(0))
    if len(out) == 4:
        rec, a, b, c = model(img.unsqueeze(0))
    if len(out) == 5:
        rec, a, b, c, _ = model(img.unsqueeze(0))

    print(rec.shape)

    rec = rec[0]
    rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
    rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
    cv2.imwrite("rec.png",rec)
    #'''
    #input = unnormalize(input)
    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("input.png",img)#cv2.cvtColor(input, cv2.COLOR_RGB2BGR))
