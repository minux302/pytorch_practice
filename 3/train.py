import random
import math
import time
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
from network import PSPNet


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train():
    rootpath = \
        "/nfs01/home/azuma/practice/pytorch_segmentation/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = \
        make_datapath_list(rootpath=rootpath)

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = VOCDataset(train_img_list,
                               train_anno_list,
                               phase="train",
                               transform=DataTransform(input_size=475,
                                                       color_mean=color_mean,
                                                       color_std=color_std))
    val_dataset = VOCDataset(val_img_list,
                             val_anno_list,
                             phase="val",
                             transform=DataTransform(input_size=475,
                                                     color_mean=color_mean,
                                                     color_std=color_std))

    batch_size = 8
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    net = PSPNet(n_classes=150)
    state_dict = torch.load("./weights/pspnet50_ADE20K.pth")
    net.load_state_dict(state_dict)

    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(in_channels=512,
                                                  out_channels=n_classes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)
    net.aux.classification = nn.Conv2d(in_channels=256,
                                       out_channels=n_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
    net.decode_feature.classification.apply(weights_init)
    net.aux.classification.apply(weights_init)

    print("weight load successed!.")


if __name__ == '__main__':
    train()
