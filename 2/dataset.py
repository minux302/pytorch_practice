import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import torch
import torch.utils.data as data

from utils.data_augmentation import (Compose,
                                     ConvertFromInts,
                                     ToAbsoluteCoords,
                                     PhotometricDistort,
                                     Expand,
                                     RandomSampleCrop,
                                     RandomMirror,
                                     ToPercentCoords,
                                     Resize,
                                     SubtractMeans,
                                     ToTensor)


def make_datapath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath, 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath, 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class Anno_xml2list(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        ret = []
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            bndbox = []
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in (pts):
                cur_pixel = int(bbox.find(pt).text) - 1

                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)
            ret += [bndbox]

        return np.array(ret)


class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(),
                SubtractMeans(color_mean),
                ToTensor()
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean),
                ToTensor()
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list,
                 phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path,
                                        width, height)

        img, boxes, labels = self.transform(img,
                                            self.phase,
                                            anno_list[:, :4],
                                            anno_list[:, 4])
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)
    return imgs, targets


if __name__ == '__main__':
    rootpath = './data/VOCdevkit/VOC2012/'
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    input_size = 300  # 画像のinputサイズを300×300にする

    train_img_list, train_anno_list, val_img_list, val_anno_list =  \
        make_datapath_list(rootpath)
    train_dataset = VOCDataset(train_img_list,
                               train_anno_list,
                               phase="train",
                               transform=DataTransform(input_size,
                                                       color_mean),
                               transform_anno=Anno_xml2list(voc_classes))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                             transform=DataTransform(input_size, color_mean),
                             transform_anno=Anno_xml2list(voc_classes))

    # img, gt = val_dataset.__getitem__(1)
    # print(gt)
    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=od_collate_fn)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=od_collate_fn)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
    images, targets = next(batch_iterator)  # 1番目の要素を取り出す
    print(images.size())  # torch.Size([4, 3, 300, 300])
    print(len(targets))
    print(targets[1].size())  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数