import torch
import torchvision
from torchvision.transforms.transforms import RandomAffine, RandomVerticalFlip, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Subset
import os
import cv2
import random
import pytorch_lightning as pl
import numpy as np
'''
def get_transform(self):
    tfms = []
    tfms.append(RandomHorizontalFlip())
    tfms.append(RandomVerticalFlip())
    tfms.append(ToTensor())
    tfms.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
    return Compose(tfms)
'''
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

class BoODataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size=256, train_mode=False, crop_offset=28):
        super().__init__()
        self.root = root
        self.image_size = crop_size
        self.train_mode = train_mode
        self.crop_offset = crop_offset
        #self.transforms = self.get_transform()
        self.classes = ['defect','ok']
        self.img_list = []
        self.list_idx = 0

        items = sorted([f for f in os.listdir(root) if 'item' in f])
        for item in items[:]:
            for f in [f for f in os.listdir(os.path.join(self.root,item,"rgb")) if f.lower().endswith(('.jpg', '.jpeg'))]:
                self.img_list.append(os.path.join(self.root,item,"rgb",f))

        # count number of samples
        width, height, _ = cv2.imread(os.path.join(self.root, self.img_list[0])).shape
        self.n_samples = len(self.img_list) * (height // self.image_size) * (width // self.image_size)
        print("number of patches {}".format(self.n_samples))

        self.crops = []
        self.labels = []
        self.coords = []
        self.filenames = []
    def load_sample(self, idx):
        # generate crops from image, label as defect if it contains a defect
        if len(self.crops) < 1:
            rgb_path = os.path.join(self.root, self.img_list[self.list_idx])
            self.list_idx = self.list_idx + 1
            mask_path = rgb_path.replace('rgb','masks').replace('jpg','png')

            filename = os.path.basename(rgb_path).split('.')[0]
            img = cv2.imread(rgb_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.int8)

            # divide image into appropritate crops and detect defects
            for x in range(0,img.shape[1]-self.image_size,self.image_size):
                for y in range(0,img.shape[0]-self.image_size,self.image_size):
                    # data augmentation by jittering the crop coordinates
                    if self.train_mode:
                        x = x + random.randint(-crop_offset, crop_offset)
                        y = y + random.randint(-crop_offset, crop_offset)
                    crop = img[y:y+self.image_size, x:x+self.image_size]
                    mask_crop = mask[y:y+self.image_size, x:x+self.image_size]
                    nonzero = np.count_nonzero(mask_crop)
                    if nonzero:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                    self.crops.append(crop)
                    self.coords.append(np.array([x,y]))
                    self.filenames.append(filename)

        return self.crops.pop(), self.labels.pop(), self.coords.pop(), self.filenames.pop()

    def __getitem__(self, idx):
        img, target, coor, name = self.load_sample(idx)

        img = img.transpose((2, 0, 1))
        img = img / 255.0
        #img[0] = (img[0] - 0.485)/0.229
        #img[1] = (img[1] - 0.456)/0.224
        #img[2] = (img[2] - 0.406)/0.225
        img = torch.from_numpy(img)
        #img = img.float()

        return img, target, coor, name

    def __len__(self):
        return self.n_samples

class BoODataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = BoODataset(self.train_dir, train_mode=False)
            self.data_test_val = BoODataset(self.test_dir, train_mode=False)
            n_sample = len(self.data_test)
            end_val_idx = int(n_sample * 0.5)
            self.data_val = Subset(self.data_test_val, range(0, end_val_idx))
            self.data_test = Subset(self.data_test_val, range(end_val_idx + 1, n_sample))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)

if __name__ == '__main__':

    dm = BoODataModule(train_dir='/home/markpp/datasets/bo/train',
                       test_dir='/home/markpp/datasets/bo/val',
                       batch_size=128)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.test_dataloader()):
        imgs, labels, coords, names = batch
        for sample in zip(imgs, labels, coords, names):
        #for sample in batch:
            img, label, coord, name = sample
            # input should probably be changed to RGB
            #img[0] = img[0] * 0.229 - 0.485
            #img[1] = img[1] * 0.224 - 0.456
            #img[2] = img[2] * 0.225 - 0.406
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            output_dir = os.path.join(output_root,name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = "id-{}_x-{}_y-{}_l-{}.jpg".format(str(sample_idx).zfill(6),coord[0],coord[1],label)
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1


    '''
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
    '''
