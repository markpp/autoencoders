from argparse import Namespace

hparams = Namespace(**{# data
                       'name': 'VQVAE2',
                       'train_root': '/home/aau3090/Datasets/celeba/train',
                       'val_root': '/home/aau3090/Datasets/celeba/val',
                       'folders': ['cat', 'dog'],
                       'image_size': 178,
                       'input_dim': 128,
                       'image_width': 512,
                       'image_height': 512,
                       # model
                       'in_channels':              3,
                       'hidden_channels':          128,
                       'res_channels':             32,
                       'nb_res_layers':            2,
                       'embed_dim':                64,
                       'nb_entries':               512,
                       'nb_levels':                3,
                       'scaling_rates':            [4, 2, 2],
                       # training
                       'gpus': 1,
                       'max_epochs': 50,
                       'learning_rate': 1e-4,
                       'beta': 0.25,
                       'beta1': 0.9,
                       'beta2': 0.999,
                       'batch_size': 96,
                       'n_workers':8})
