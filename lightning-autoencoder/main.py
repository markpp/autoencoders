from argparse import ArgumentParser
import os
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary

from model_64_64 import create_encoder, create_decoder
from autoencoder import Autoencoder
from harbour_datamodule import list_frames_in_dir

def play(view, crop, set, model=None):
    frames = list_frames_in_dir(os.path.join('data/',view,crop,set), 'png')
    if frames is None:
        return

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    view_dir = os.path.join(output_dir,view)
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)

    set_dir = os.path.join(view_dir,set)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    print(len(frames))
    for i, path in enumerate(frames):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        intensity, dx, dy = cv2.split(img)

        if model:
            img = img.transpose((2, 0, 1))
            img = img / 255.0
            img[0] = (img[0] - 0.5)/0.5
            img[1] = (img[1] - 0.5)/0.5
            img[2] = (img[2] - 0.5)/0.5
            img = torch.from_numpy(img)
            img = img.float()
            rec = model(img.unsqueeze(0))[0]
            rec[0] = rec[0] * 0.5 - 0.5
            rec[1] = rec[1] * 0.5 - 0.5
            rec[2] = rec[2] * 0.5 - 0.5
            rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            intensity_, dx_, dy_ = cv2.split(rec)

            vis_org = np.concatenate((intensity, dx, dy), axis=1)
            vis_reg = np.concatenate((intensity_, dx_, dy_), axis=1)
            vis = np.concatenate((vis_org, vis_reg), axis=0)

        else:
            vis = np.concatenate((intensity, dx, dy), axis=1)

        vis = cv2.putText(vis, str(i).zfill(4), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
        vis = cv2.resize(vis, (vis.shape[1]*3,vis.shape[0]*3), interpolation = cv2.INTER_AREA)
        cv2.imshow(set,vis)
        cv2.imwrite("{}.png".format(str(i).zfill(5)),vis)
        key = cv2.waitKey(0)
        if key == 27:
            break


def train(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"bs{hparams.batch_size}_nf{hparams.nfe}")
    model = Autoencoder(hparams)
    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.image_size, hparams.image_size), device="cpu")
    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)
    trainer.test(model)

def test(hparams, path):
    model = Autoencoder.load_from_checkpoint(path)
    model.eval()
    play(view = 'view1', crop = 'crop0', set = 'test', model = model)


if __name__ == "__main__":
    parser = ArgumentParser()

    #parser.add_argument("--data_root", type=str, default="data/teejet", help="Train root directory")
    parser.add_argument("--data_root", type=str, default="data/view1/crop0", help="View root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=1, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=16, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    #train(args)
    model_path = '/home/markpp/github/autoencoders/lightning-autoencoder/logs/bs64_nf32/version_27/checkpoints/epoch=8.ckpt'
    test(args,model_path)
    #play(view = 'view1', crop = 'crop0', set = 'test')
