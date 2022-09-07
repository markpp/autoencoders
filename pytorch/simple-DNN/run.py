import torch
import pytorch_lightning as pl
#import matplotlib.pyplot  as     plt

from torchvision.datasets   import MNIST
#from torch.utils.data       import DataLoader, random_split
from torchvision.transforms import ToTensor

from transforms import MinMaxNormalize
from model import SimpleAutoencoder



def trainae():
    print("It's simpleae-ing time")

    trnsf = Compose([
        ToTensor(),
        MinMaxNormalize()
    ])

    mnist = MNIST(root='datasets/', train=True, download=True, transform=trnsf)

    train_size = int( 0.64 * len(mnist) )
    val_size   = int( 0.16 * len(mnist) )
    test_size  = len(mnist) - (train_size + val_size)
    train, val, test = random_split(mnist, [train_size, val_size, test_size])

    # Defining DataLoader for model
    train_DL = DataLoader(train, batch_size=32, shuffle=True)
    val_DL   = DataLoader(val, batch_size=32, shuffle=False)
    test_DL  = DataLoader(test, batch_size=32, shuffle=False)

    # Using model
    model = SimpleAutoencoder()
    trainer = pl.Trainer(limit_train_batches=1000, max_epochs=10)
    trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL)


def train():
    dataset = MNISTDataCarrier((0.64, 0.16, 0.2), 'datasets/', [ToTensor(), MinMaxNormalize()])
    train, val, test = dataset.to_train()

    model = SimpleAutoencoder()
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10)

    trainer.fit(model, train, val)
    trainer.test(model, test)

def run(args):
    if args.train:
        train()




