import os.path as osp
from PIL import Image

import torch.utils.data as data


def make_datapath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_templete = osp.join(rootpath, 'SegmentationClass', '%s.png')

    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_templete % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_templete % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


if __name__ == '__main__':
    rootpath = './data/VOCdevkit/VOC2012/'
    train_img_list, train_anno_list, \
        val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

    print(train_img_list[0])
    print(train_anno_list[0])
