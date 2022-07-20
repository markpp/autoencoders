from argparse import Namespace

hparams = Namespace(**{# data
                       'name': 'VQVAE2',
                       'dataset': 'sewer',
                       'data_path': '/home/markpp/github/autoencoders/pytorch/datasets/sewer/imgs',
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
                       'batch_size': 48,
                       'n_workers':8})
