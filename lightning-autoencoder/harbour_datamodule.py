import torch
import torchvision
from torchvision.transforms.transforms import RandomAffine, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
import os
import cv2

def list_frames_in_dir(dir, file_type):
    frame_list = []
    print("looking in dir: {}".format(dir))
    file_list = os.listdir(dir)
    if len(file_list)>0:
        for filename in file_list:
            name, postfix = filename.split('.')
            if (postfix == file_type):
                frame_list.append(os.path.join(dir, filename))
        return sorted(frame_list)
    else:
        print("did not find any files")
        return None

class HarbourDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, crop_size=64):
        self.crop_size = crop_size
        self.transforms = self.get_transform()


        self.image_files = list_frames_in_dir(data_dir, 'png')

    def load_sample(self, label_path):
        img = cv2.imread(image_path)
        self.image_h, self.image_w, _ = img.shape
        #img = cv2.cvtColor(img[:self.image_h,:self.image_h], cv2.COLOR_BGR2RGB)
        return img

    def get_transform(self):
        tfms = []
        tfms.append(ToTensor())
        #tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
        return Compose(tfms)

    def __getitem__(self, idx):
        img = self.load_sample(self.image_files[idx])

        #img = self.transforms(img)

        img = img.transpose((2, 0, 1))
        img = img / 255.0
        #img[0] = (img[0] - 0.485)/0.229
        #img[1] = (img[1] - 0.456)/0.224
        #img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        img = img.float()
        return img#, img#, target

    def __len__(self):
        return len(self.image_files)

'''
class HarbourDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def prepare_data_from_list(list, img_size=64):
        dataset = SprayDataset(list, img_size)
        return dataset

    def standard_dataloader(dataset, batch_size=16, num_workers=1):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    unnormalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
'''
if __name__ == '__main__':

    dataset = prepare_data_from_list('/home/markpp/datasets/teejet/iphone_data/val.txt')
    dataloader = standard_dataloader(dataset, batch_size=8)

    # cleanup output dir
    import os, shutil
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output")

    import numpy as np
    import cv2
    data = iter(dataloader)
    labels = []
    for batch in range(2):
        images,targets = next(data)
        for i, img_tar in enumerate(zip(images,targets)):
            img, tar = img_tar
            #labels.append(tar.item())
            #
            img = unnormalize(img)
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            tar = unnormalize(tar)
            tar = tar.mul(255).permute(1, 2, 0).byte().numpy()

            if batch == 0:
                cv2.imwrite("output/b{}_i{}_in.png".format(batch,i),img)
                cv2.imwrite("output/b{}_i{}_tar.png".format(batch,i),tar)
