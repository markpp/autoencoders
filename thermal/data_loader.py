import torchvision
from torchvision.transforms.transforms import RandomAffine, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from datasets import BOItemDataset, BODataset, BOItemDataset

# define transformations
affine = RandomAffine(degrees=90, scale=(0.9,1.2), shear=15, resample=False, fillcolor=0)
crop = RandomCrop(224)
h_flip = RandomHorizontalFlip(0.5)
tensorize = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
unnormalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])

train_transforms = Compose([affine, crop, h_flip, tensorize, normalize])
val_transforms = Compose([crop, tensorize, normalize])
test_transforms = Compose([tensorize, normalize])

#
#def custom_data_from_npy(folder, augment=False):

#    return dataset

#def custom_dataloader(dataset, batch_size=16, num_workers=1):
#    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

def prepare_data_from_folder(folder, mode="train"):
    if mode=="train":
        dataset = torchvision.datasets.ImageFolder(root=folder,  transform=train_transforms)
    elif mode=="val":
        dataset = torchvision.datasets.ImageFolder(root=folder, transform=val_transforms)
    else:
        #dataset = torchvision.datasets.ImageFolder(root=folder, transform=test_transforms)
        #dataset = BODataset(folder)
        dataset = BOItemDataset(folder)
    return dataset

def balanced_sampler(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    class_weights = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        class_weights[i] = N/float(count[i])
    #print("class weights {}".format(weight_per_class))
    weights = [0] * len(images)
    for idx, val in enumerate(images):
        weights[idx] = class_weights[val[1]]
    return WeightedRandomSampler(weights, len(weights))

def balanced_dataloader(dataset, batch_size=16, num_workers=1):
    sampler = balanced_sampler(dataset.imgs, len(dataset.classes))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=False)

def standard_dataloader(dataset, batch_size=16, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

if __name__ == '__main__':

    dataset = prepare_data_from_folder('/home/markpp/datasets/bo/val_448', train=False)
    dataloader = balanced_dataloader(dataset)

    # cleanup output dir
    import os, shutil
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output")

    import numpy as np
    import cv2
    data = iter(dataloader)
    labels = []
    for batch in range(10):
        images,targets = next(data)
        for i, img_tar in enumerate(zip(images,targets)):
            img, tar = img_tar
            labels.append(tar.item())
            img = unnormalize(img)
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            if batch == 0:
                cv2.imwrite("output/b{}_i{}_c{}.png".format(batch,i,tar),img)
        print("# defects {}, ok {}".format(np.count_nonzero(np.array(labels) == 0),np.count_nonzero(np.array(labels) == 1)))
