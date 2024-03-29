import argparse
import os, sys
import yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='trained_models/'
)

def create_datamodule(config):
    sys.path.append('../datasets')

    if config['exp_params']['dataset']=='sewer':
        from sewer.datamodule import SewerDataModule
        dm = SewerDataModule(data_dir=config['exp_params']['data'],
                             batch_size=config['exp_params']['batch_size'],
                             image_size=config['exp_params']['image_size'])
        dm.setup()
        return dm
    else:
        print("no such dataset: {}".format(config['exp_params']['data']))
        return None

if __name__ == '__main__':
    """
    Trains an autoencoder from patches of thermal imaging.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str,
                    default='config/vae.yaml', help="path to the config file")
    ap.add_argument("-p", "--checkpoint", type=str,
                    default='trained_models/model.pt', help="path to the checkpoint file")
    args = vars(ap.parse_args())

    with open(args['config'], 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dm = create_datamodule(config)
    if dm == None:
        print("failed to create datamodule")
        exit()

    from lightning_model import VAE
    model = VAE(input_height=config['exp_params']['image_size'],
                enc_type='resnet50',
                first_conv=True,
                maxpool1=True,
                enc_out_dim=2048,
                kl_coeff=0.1,
                latent_dim=256,
                lr=1e-4)

    trainer = pl.Trainer(gpus=1, max_epochs=config['trainer_params']['max_epochs'], checkpoint_callback=checkpoint_callback)#, profiler=True)

    trainer.fit(model, dm)

    output_dir = 'trained_models/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.model, os.path.join(output_dir,"model.pt"))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(output_dir,"final.ckpt"))
