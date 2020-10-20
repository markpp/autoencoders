import torchvision
from torchvision.transforms.transforms import RandomAffine, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from datasets import HarborDataset

def prepare_data_from_list(list, img_size=64):
    dataset = HarborDataset(list, img_size)
    return dataset

def standard_dataloader(dataset, batch_size=16, shuffle=False, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == '__main__':
    dataset = prepare_data_from_list('/home/markpp/datasets/harbour_frames/2/view1_train.txt')
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

            img = img.mul(255).byte().numpy()

            tar = tar.mul(255).byte().numpy()

            if batch == 0:
                cv2.imwrite("output/b{}_i{}_in.png".format(batch,i),img)
                cv2.imwrite("output/b{}_i{}_tar.png".format(batch,i),tar)
