import torchvision
from torchvision.transforms.transforms import RandomAffine, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from datasets import SprayDataset



def standard_dataloader(dataset, batch_size=16, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

unnormalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225])

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
