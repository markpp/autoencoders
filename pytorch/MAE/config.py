from argparse import Namespace

# Large
hparams = Namespace(**{# data
                       #'train_root': '/home/aau3090/Datasets/celeba/train',
                       #'val_root': '/home/aau3090/Datasets/celeba/val',
                       'train_root': '/home/aau3090/Datasets/nozzle_dataset/spray_frames/generative/train',
                       'val_root': '/home/aau3090/Datasets/nozzle_dataset/spray_frames/generative/val_sample',
                       'folders': [],
                       'image_size': 224,#192,
                       'in_channel': 1,
                       # model
                       'patch_size': 16,
                       'encoder_emb_dim': 192,
                       'encoder_layer': 12,
                       'encoder_head': 4,
                       'decoder_emb_dim': 512,
                       'decoder_layer': 8,
                       'decoder_head': 16,
                       'mask_ratio': 0.75,
                       # training
                       'gpus': 1,
                       'max_epochs': 250,
                       'warmup_epochs': 20,
                       'learning_rate': 1e-4,
                       'weight_decay': 0.05,
                       'batch_size': 256,
                       'n_workers': 4})
