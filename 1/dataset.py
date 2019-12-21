import glob
import os

from PIL import Image

import torch.utils.data as data
from torchvision import models, transforms


def make_datapath_list(phase='train'):
    rootpath = './data/hymenoptera_data/'
    target_path = os.path.join(rootpath + phase + '/**/*.jpg')
    print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label


if __name__ == '__main__':
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    train_dataset = HymenopteraDataset(
        file_list=train_list,
        transform=ImageTransform(size, mean, std),
        phase='train')

    val_dataset = HymenopteraDataset(
        file_list=val_list,
        transform=ImageTransform(size, mean, std),
        phase='val')

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])
