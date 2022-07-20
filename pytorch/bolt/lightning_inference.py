import argparse
import os
import yaml

import torch
import pytorch_lightning as pl

import cv2

if __name__ == '__main__':
    """
    Embeds and reconstructs images.

    Command:
        python lightning_inference.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    ap.add_argument("-c", "--config", type=str,
                    default='config/vae.yaml', help="path to the config file")
    ap.add_argument("-d", "--data", type=str,
                    default='val.txt', help="path to list of files")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    #from lightning_model import LightningAutoencoder
    #model = LightningAutoencoder(config)
    #model.load_state_dict(torch.load(args['checkpoint']))
    #model = LightningAutoencoder(config).load_from_checkpoint(args['checkpoint'])
    model = torch.load(args['checkpoint'])
    model.eval()

    with open(args['data']) as f:
        image_list = f.read().splitlines()

    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image_path in image_list[:]:
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        rec, input, mu, log_var = model(img.unsqueeze(0))
        print(rec.shape)
        print(mu)
        rec = rec[0]
        rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
        cv2.imwrite(os.path.join(output_dir,filename),rec)
