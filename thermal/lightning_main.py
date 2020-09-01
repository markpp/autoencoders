import argparse
import os

import torch
import pytorch_lightning as pl

import config


if __name__ == '__main__':
    """
    Trains a patch classifier.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fine", type=str,
                    default='fine', help="Flag")
    args = vars(ap.parse_args())

    trainer = pl.Trainer(gpus=1, max_epochs=100)#, profiler=True)

    from lightning_model import LightningDetector
    model = LightningDetector(config.hparams)

    trainer.fit(model)
