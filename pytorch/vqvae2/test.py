import argparse
import os, sys
import cv2
import numpy as np
import csv

import torch
import pytorch_lightning as pl
#from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms

from vqvae2 import VQVAE2

from config import hparams

from sewer_dataset import SewerDataset

import albumentations as Augment

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


test_t = Augment.Compose([Augment.SmallestMaxSize(max_size=hparams.image_height, interpolation=cv2.INTER_LINEAR, always_apply=True),
                          Augment.CenterCrop(hparams.image_height, hparams.image_height, always_apply=True),
                          Augment.Normalize(mean, std)
              ])


data_val = SewerDataset(os.path.join(hparams.data_path,'val'),
                             transforms=test_t,
                             n_channels=hparams.in_channels)

def largest_center_crop(img):
    h, w, c = img.shape
    if h > w:
        top_h = int((h - w) / 2)
        img = img[top_h:top_h + w]
    else:
        left_w = int((w - h) / 2)
        img = img[:, left_w:left_w + h]
    return img

'''
def normalize(img):
    img[0] = (img[0] - 0.5) / 0.5
    img[1] = (img[1] - 0.5) / 0.5
    img[2] = (img[2] - 0.5) / 0.5
    return img

def denormalize(img):
    img[0] = (img[0] * 0.5) + 0.5
    img[1] = (img[1] * 0.5) + 0.5
    img[2] = (img[2] * 0.5) + 0.5
    return img
'''

mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

normalize = transforms.Normalize(mean.tolist(), std.tolist())
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/net.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    data_dir = '/home/markpp/github/data/sewer/'

    file = open(os.path.join(data_dir,'ALL_GT.csv'))
    csv_reader = csv.reader(file)
    next(csv_reader)

    with torch.no_grad():

        net = VQVAE2(in_channels=hparams.in_channels,
                     hidden_channels=hparams.hidden_channels,
                     embed_dim=hparams.embed_dim,
                     nb_entries=hparams.nb_entries,
                     nb_levels=hparams.nb_levels,
                     scaling_rates=hparams.scaling_rates)

        net.load_state_dict(torch.load(args['checkpoint']))
        '''
        net = torch.load(args['checkpoint'])
        net.eval()
        '''
        for img in data_val:
        #for row in csv_reader:
            #img_path = os.path.join(data_dir,'val',row[0])
            #if os.path.isfile(img_path):
                #print(img_path)
            #img = cv2.imread(img_path)

            #x = largest_center_crop(img)
            #x = cv2.resize(x, (hparams.image_height, hparams.image_height))
            #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            #x = x.transpose((2, 0, 1))
            #x = torch.as_tensor(x, dtype=torch.float32)/255.0
            #x = normalize(x)
            #print(x.unsqueeze(0).shape)
            x = img
            rec, diffs, encs, recs = net(x.unsqueeze(0))
            print(rec.shape)
            '''
            print(len(diffs))
            print(diffs[0].shape)
            print(len(encs))
            print(encs[0].shape)
            print(len(recs))
            print(recs[0].shape)
            '''
            rec = unnormalize(rec[0])
            print(rec.shape)
            rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.imshow("rec",rec)

            #print(row[4])
            #cv2.putText(img, "gt {:.0f}, pred {:.0f}".format(gt,pred), (x+10,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            #cv2.imshow("test",img)
            key = cv2.waitKey()
            if key == 27:
                break
