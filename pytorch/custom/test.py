from argparse import ArgumentParser
import os
import cv2
import numpy as np
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder
from views import views
from plotter import Plotter


def play_thermal(frame_list, hparams, output_dir, encoder=None, decoder=None, norm=True, save=True, n_channels=1):

    # Create a plotter class object
    plot = Plotter(192*2, 192*2)

    view_dir = os.path.join(output_dir,hparams.view)
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)

    crop_name = "crop{}".format(hparams.crop_idx)

    crop_dir = os.path.join(view_dir,crop_name)
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)

    inputs, recs, files, losses, latent = [], [], [], [], []
    print(len(frame_list))
    for i, frame_path in enumerate(frame_list):
        thermal = cv2.imread(frame_path)[:,:,0] #[:,:,:1]
        x, y = views[hparams.view][hparams.crop_idx]['x'], views[hparams.view][hparams.crop_idx]['y']
        crop = thermal[y:y+hparams.crop_size, x:x+hparams.crop_size]
        inputs.append(crop)
        files.append(frame_path)
        img = crop / 255.0
        if norm:
            img = (img - 0.5)/0.5
        img = torch.from_numpy(img)
        img = img.float()
        x = img.unsqueeze(0).unsqueeze(0)
        if encoder is not None:
            z = encoder(x)
        if decoder is not None:
            rec = decoder(z)[0][0]
        loss = F.mse_loss(rec, img)
        if norm:
            rec = rec * 0.5 - 0.5
        rec = rec.mul(255).byte().numpy()
        recs.append(rec)

        vis = np.concatenate((inputs[-1], recs[-1]), axis=0)
        losses.append(loss.item())
        latent.append(z[0].detach().numpy().flatten())
        vis = cv2.resize(vis, (vis.shape[1]*3,vis.shape[0]*3), interpolation = cv2.INTER_NEAREST)

        plot.plot(loss*10000)
        output = np.concatenate((cv2.merge((vis,vis,vis)), plot.plot_canvas), axis=1)

        output = cv2.putText(output, str(i).zfill(4), (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow(crop_name,output)
        if save:
            l = "{:.6f}".format(losses[-1])[-6:]
            cv2.imwrite(os.path.join(crop_dir,"{}_l-{}.png".format(str(i).zfill(5),l)),vis)
        key = cv2.waitKey(30)

        if key == 27:
            break
    np.save(os.path.join(view_dir,'{}_inputs.npy'.format(crop_name)), inputs)
    np.save(os.path.join(view_dir,'{}_recs.npy'.format(crop_name)), recs)
    np.save(os.path.join(view_dir,'{}_files.npy'.format(crop_name)), files)
    np.save(os.path.join(view_dir,'{}_losses.npy'.format(crop_name)), losses)
    np.save(os.path.join(view_dir,'{}_latent.npy'.format(crop_name)), latent)

def test(hparams):

    model = Autoencoder(hparams)

    model.encoder = torch.load("encoder.pt")
    model.decoder = torch.load("decoder.pt")

    model.encoder.eval()
    model.decoder.eval()

    folders = sorted([y for y in glob(os.path.join(hparams.data_root, '* - *'))])

    for folder in folders[2:3]:

        #create output folder
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        folder_dir = os.path.join(output_dir,folder.split('/')[-1])
        if not os.path.exists(folder_dir):
            os.mkdir(folder_dir)

        # list images
        frame_list = sorted([y for y in glob(os.path.join(folder, 'img_*.jpg'))])
        if not len(frame_list)>0:
            print("did not find any files")
            return

        play_thermal(frame_list, hparams, folder_dir, encoder=model.encoder.to('cpu'), decoder=model.decoder.to('cpu'), norm=hparams.norm, n_channels=hparams.nc)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/markpp/datasets/harbour_frames/2/view1/", help="View root directory")
    parser.add_argument('--view', type=str, help='which view folder', default='view1')
    parser.add_argument('--crop_idx', type=int, help='which crop?', default=0)
    parser.add_argument('--crop_size', type=int, help='crop size', default=64)
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=8, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")

    args = parser.parse_args()

    test(args)
