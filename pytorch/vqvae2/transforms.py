import albumentations as A

def train_transforms(frame_size, crop_size, mean=0.0, std=1.0, norm=True):
    transforms = []
    transforms.append(A.SmallestMaxSize(max_size=frame_size, always_apply=True))
    transforms.append(A.CenterCrop(frame_size, frame_size, always_apply=True))
    transforms.append(A.Resize(crop_size, crop_size, always_apply=True))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))
    return A.Compose(transforms)

def aug_transforms(frame_size, crop_size, mean=0.0, std=1.0, norm=True):
    transforms = []
    transforms.append(A.SmallestMaxSize(max_size=frame_size+10, always_apply=True))
    transforms.append(A.Rotate(limit=5, interpolation=1, border_mode=4, always_apply=True))
    #transforms.append(A.RandomBrightness(limit=0.2, always_apply=True))
    transforms.append(A.RandomCrop(frame_size, frame_size, always_apply=True))
    transforms.append(A.Resize(crop_size, crop_size, always_apply=True))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))
    return A.Compose(transforms)

mean, std = 0.0, 1.0
#mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # normalize
#mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # imagenet normalization

def denormalize(img):
    if isinstance(mean, tuple):
        img[0] = img[0] * std[0] + mean[0]
        img[1] = img[1] * std[1] + mean[1]
        img[2] = img[2] * std[2] + mean[2]
    else:
        img[0] = img[0] * std + mean
    return img

def normalize(img):
    if isinstance(mean, tuple):
        img[0] = (img[0] - mean[0]) / std[0]
        img[1] = (img[1] - mean[1]) / std[1]
        img[2] = (img[2] - mean[2]) / std[2]
    else:
        img[0] = img[0] * std + mean
    return img

