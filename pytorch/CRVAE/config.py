from argparse import Namespace

# Large
hparams = Namespace(**{# data
                       'train_root': '/home/aau3090/Datasets/celeba/train',
                       'val_root': '/home/aau3090/Datasets/celeba/val',
                       'folders': ['cat', 'dog'],
                       'image_size': 178,
                       'input_dim': 128,
                       'in_channels': 3,
                       'latent_dim': 128,
                       # model
                       #'nc':          1,
                       #'nz':          8,
                       #'nfe':         32,
                       #'nfd':         32,
                       # training
                       'gpus': 1,
                       'max_epochs': 500,
                       'learning_rate': 1e-5,
                       'batch_size': 96,
                       'n_workers': 4})
