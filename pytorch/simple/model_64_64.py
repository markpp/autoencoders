import torch
import torch.nn as nn

def create_encoder(config, train=True):
    return nn.Sequential(
        # input (nc) x 64 x 64
        nn.Conv2d(config['model_params']['in_channels'], config['model_params']['nfe'], 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfe']),
        nn.LeakyReLU(True),
        # input (nfe) x 32 x 32
        nn.Conv2d(config['model_params']['nfe'], config['model_params']['nfe'] * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfe'] * 2),
        nn.LeakyReLU(True),
        # input (nfe*2) x 16 x 16
        nn.Conv2d(config['model_params']['nfe'] * 2, config['model_params']['nfe'] * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfe'] * 4),
        nn.LeakyReLU(True),
        # input (nfe*4) x 8 x 8
        nn.Conv2d(config['model_params']['nfe'] * 4, config['model_params']['nfe'] * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfe'] * 8),
        nn.LeakyReLU(True),
        # input (nfe*8) x 4 x 4
        nn.Conv2d(config['model_params']['nfe'] * 8, config['model_params']['latent_dim'], 4, 1, 0, bias=False),
        nn.BatchNorm2d(config['model_params']['latent_dim']),
        nn.LeakyReLU(True)
        # output (nz) x 1 x 1
    )

def create_decoder(config):
    return nn.Sequential(
        # input (nz) x 1 x 1
        nn.ConvTranspose2d(config['model_params']['latent_dim'], config['model_params']['nfd'] * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(config['model_params']['nfd'] * 8),
        nn.ReLU(True),
        # input (nfd*8) x 4 x 4
        nn.ConvTranspose2d(config['model_params']['nfd'] * 8, config['model_params']['nfd'] * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfd'] * 4),
        nn.ReLU(True),
        # input (nfd*4) x 8 x 8
        nn.ConvTranspose2d(config['model_params']['nfd'] * 4, config['model_params']['nfd'] * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfd'] * 2),
        nn.ReLU(True),
        # input (nfd*2) x 16 x 16
        nn.ConvTranspose2d(config['model_params']['nfd'] * 2, config['model_params']['nfd'], 4, 2, 1, bias=False),
        nn.BatchNorm2d(config['model_params']['nfd']),
        nn.ReLU(True),
        # input (nfd) x 32 x 32
        nn.ConvTranspose2d(config['model_params']['nfd'], config['model_params']['in_channels'], 4, 2, 1, bias=False),
        nn.Tanh()
        # output (nc) x 64 x 64
    )
