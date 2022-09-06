import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.utils import shuffle
#from hparams import HParams
from PIL import Image
import cv2
#from torch.utils.distributed import DistributedSampler
from torch.utils.data import Sampler as DistributedSampler

#hparams = HParams.get_hparams_by_name("efficient_vdvae")
import config as hparams


class Normalize(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype

        img = np.floor(img / np.uint8(2 ** (8 - hparams.num_bits))) * 2 ** (8 - hparams.num_bits)
        img = img.astype(img_dtype)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Tensor
        """
        img = np.asarray(img)

        shift = scale = (2 ** 8 - 1) / 2
        img = (img - shift) / scale  # Images are between [-1, 1]
        #return torch.tensor(img).permute(2, 0, 1).contiguous().float()
        return torch.unsqueeze(torch.tensor(img), 0).contiguous().float()
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


if hparams.random_horizontal_flip:
    train_transform = transforms.Compose([
        Normalize(),
        transforms.RandomHorizontalFlip(),
        MinMax(),
    ])
else:
    train_transform = transforms.Compose([
        Normalize(),
        MinMax(),
    ])

valid_transform = transforms.Compose([
    Normalize(),
    MinMax(),
])


def create_filenames_list(path):
    filenames = sorted(os.listdir(path))
    files = [os.path.join(path, f) for f in filenames]
    print(path, len(files))
    return files, filenames


def read_resize_image(image_file):
    #img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (hparams.target_res, hparams.target_res), interpolation= cv2.INTER_LINEAR)
    #img = np.expand_dims(img, axis=2)
    #return torch.as_tensor(img/255.0, dtype=torch.float)
    #print(img.shape)
    #return Image.fromarray(img)
    #return Image.open(image_file).convert("RGB").resize((hparams.target_res, hparams.target_res), resample=Image.BILINEAR)
    #return Image.open(image_file).convert("LA").resize((hparams.target_res, hparams.target_res), resample=Image.BILINEAR)
    return Image.open(image_file).resize((hparams.target_res, hparams.target_res), resample=Image.BILINEAR)


class generic_dataset(torch.utils.data.Dataset):
    def __init__(self, files, filenames, mode):

        self.mode = mode
        if mode != 'encode':
            self.files, self.filenames = shuffle(files, filenames)
        else:
            self.files = files
            self.filenames = filenames

    def __getitem__(self, idx):
        if self.mode == 'train':
            img = read_resize_image(self.files[idx])
            img = train_transform(img)
            return img

        elif self.mode in ['val', 'div_stats', 'test']:
            img = read_resize_image(self.files[idx])
            img = valid_transform(img)
            return img

        elif self.mode == 'encode':
            filename = self.filenames[idx]
            img = read_resize_image(self.files[idx])
            img = valid_transform(img)
            return img, filename

        else:
            raise ValueError(f'Unknown Mode {self.mode}')

    def __len__(self):
        if self.mode in ['train', 'encode', 'test']:
            return len(self.files)
        elif self.mode == 'val':
            return hparams.n_samples_for_validation
        elif self.mode == 'div_stats':
            return round(len(self.files) * hparams.div_stats_subset_ratio)


def train_val_data_generic(train_images, train_filenames, val_images, val_filenames, world_size, rank):
    train_data = generic_dataset(train_images, train_filenames, mode='train')
    #train_sampler = DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(#sampler=train_sampler,
                                               dataset=train_data,
                                               batch_size=hparams.train_batch_size // hparams.num_gpus,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2,
                                               drop_last=True, prefetch_factor=3)

    val_data = generic_dataset(val_images, val_filenames, mode='val')
    #val_sampler = DistributedSampler(val_data)
    val_loader = torch.utils.data.DataLoader(#sampler=val_sampler,
                                             dataset=val_data,
                                             batch_size=hparams.val_batch_size // hparams.num_gpus,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             drop_last=True, prefetch_factor=3)

    return train_loader, val_loader


def synth_generic_data():
    synth_images, synth_filenames = create_filenames_list(hparams.synthesis_data_path)
    synth_data = generic_dataset(synth_images, synth_filenames, mode='test')
    synth_loader = torch.utils.data.DataLoader(
        dataset=synth_data,
        batch_size=hparams.synthesis_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True)
    return synth_loader


def encode_generic_data():
    images, filenames = create_filenames_list(hparams.train_data_path)
    data = generic_dataset(images, filenames, mode='encode')
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=hparams.synthesis_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2)

    return data_loader


def stats_generic_data():
    images, filenames = create_filenames_list(hparams.train_data_path)
    data = generic_dataset(images, filenames, mode='div_stats')
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=hparams.synthesis_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2)

    return data_loader
