from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import os
from PIL import Image

from imagen_pytorch.t5 import t5_encode_text, DEFAULT_T5_NAME

# helpers functions

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# dataset and dataloader

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        convert_image_to_type = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        #self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        convert_fn = partial(convert_image_to, convert_image_to_type) if exists(convert_image_to_type) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        #self.enc_dims = 768

        self.encode_text = partial(t5_encode_text, name = DEFAULT_T5_NAME)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        texts = [os.path.basename(os.path.dirname(str(path)))] # weeds
        #texts = [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(str(path)))))] # nozzles
        text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)
        img = Image.open(path)
        return self.transform(img), text_embeds[0], text_masks[0]#, mask

def get_images_dataloader(
    folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = False,
    pin_memory = True
):
    ds = Dataset(folder, image_size)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    if cycle_dl:
        dl = cycle(dl)
    return dl
