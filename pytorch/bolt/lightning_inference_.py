import argparse
import os
import yaml

import torch
import pytorch_lightning as pl

import cv2

if __name__ == '__main__':
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

    pl.seed_everything()

    from argparse import ArgumentParser
    args=None
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "stl10", "imagenet"])
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    from lightning_model import VAE

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = VAE(**vars(args))
    model.load_from_checkpoint("trained_models/-epoch=3.ckpt", input_height=args.input_height)
    #model.load_state_dict(torch.load("trained_models/model_cifar10.pt"))
    model.eval()

    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    '''
    for image_path in image_list[:]:
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        img = img.transpose((2, 0, 1))
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
    '''
    for img in dm.val_dataloader:
        rec, input, mu, log_var = model(img.unsqueeze(0))
        print(rec.shape)
        print(mu)
        rec = rec[0]
        rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
        cv2.imwrite(os.path.join(output_dir,filename),rec)
