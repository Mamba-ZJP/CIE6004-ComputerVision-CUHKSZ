import sys, os, pdb
from os import path as osp

import torch, torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2


class EG1800Dataset(Dataset):
    def __init__(self, cfg, is_train=True):
        super().__init__()
        # root = osp.join(*cfg.img_dir.split('/')[:-1])
        self.img_path = sorted(
            [osp.join(cfg.img_dir, file) for file in os.listdir(cfg.img_dir) if file.endswith('.png') or file.endswith('.jpg')]
        )
        self.ann_path = sorted(
            [osp.join(cfg.ann_dir, ann) for ann in os.listdir(cfg.ann_dir) if ann.endswith('.png') or ann.endswith('.jpg')]
        )
        self.to_tensor = T.ToTensor()
        # print(self.img_path[0])

    def __getitem__(self, idx):
        '''
        output:
            img_orgin, img, edge_gt, mask_gt
        '''
        img_orig = cv2.imread(self.img_path[idx])
        img_aug = img_orig.copy()
        mask_gt = cv2.imread(self.ann_path[idx], flags=cv2.IMREAD_UNCHANGED)

        assert len(mask_gt.shape) == 2, "the channel number is not one!"

        edge_gt = self.get_edge(mask_gt)

        img_orig, img_aug, mask_gt, edge_gt = self.to_tensor(img_orig), self.to_tensor(img_aug),\
                                                self.to_tensor(mask_gt), self.to_tensor(edge_gt)
        return img_orig, img_aug, mask_gt, edge_gt

    def __len__(self):
        return len(self.img_path)


    def get_edge(self, mask):
        H, W = mask.shape
        edge_gt = np.zeros((H, W), np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edge_gt = cv2.drawContours(edge_gt, contours, -1, 255, 4)

        return edge_gt


def get_train_loader(cfg):
    train_loader = DataLoader(
        dataset=EG1800Dataset(cfg, is_train=True),
        batch_size=cfg.batch_size,
        shuffle=True
    )
    return train_loader


def get_valid_loader(cfg):
    valid_loader = DataLoader(
        dataset=EG1800Dataset(cfg, is_train=False),
        batch_size=cfg.batch_size,
        shuffle=False
    )
    return valid_loader