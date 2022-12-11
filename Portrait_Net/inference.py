import os, sys, pdb
from os import path as osp
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import torchvision

from model.portraitnet import get_portraitnet_mobilenetv2
from dataset.datasets import get_valid_loader
from configs.config import cfg

'''
This file is for loading the model for the inference.
Inference's dataloader depends on the situation.
'''

class ModelInfer():
    def __init__(self, ckpt_pth, model) -> None:
        super().__init__()
        load = torch.load(ckpt_pth) # return a dict
        model.load_state_dict(load['state_dict'], strict=True)
        
        cfg.batch_size = 4
        self.device = torch.device(f'cuda:{cfg.gpus[0]}')

        self.model = model.to(self.device)
        self.infer_loader = get_valid_loader(cfg) # use valid_loader for now

    def forward(self, x):
        raise NotImplemented
    
    def to_cuda(self, batch):
        if isinstance(batch, list) or isinstance(batch, tuple):
            for i, _ in enumerate(batch):
                batch[i] = batch[i].to(self.device)
        else:
            batch = batch.to(self.device)
        return batch
    
    def to_ndarray(self, x, permute):
        '''
            convert Tensor to ndarray and transpose the dim for cv2 API
        '''
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            if permute:
                if len(x.shape) == 4:
                    x = np.transpose(x, axes=(0, 2, 3, 1))
                elif len(x.shape) == 3:
                    x = np.transpose(x, (1, 2, 0))
        return x

    def infer(self):
        for iter, batch in enumerate(self.infer_loader):
            if iter == 10:
                break
            img_orig, img_aug, _, _ = self.to_cuda(batch)
            # img_orig's colorSpace: bgr (don't need to change); img_aug's colorSpace: bgr don't need to change
            pred_mask, _ = self.model(img_aug)
            pred_mask = F.softmax(pred_mask, dim=1)[:, 1, ...]
            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0
            pred_mask = pred_mask.unsqueeze(dim=1).expand(-1,3,-1,-1) # broadcasting
            # pdb.set_trace()

            input_pred = torch.concat([img_aug, pred_mask], dim=0)
            grid = torchvision.utils.make_grid(input_pred, nrow=cfg.batch_size).mul(255).clamp(0,255)
            grid = self.to_ndarray(grid, True)
            cv2.imwrite(osp.join(cfg.output_dir, f'grid_{iter}.jpg'), grid)
            
            # cv2.cvtColor(img_orig, code=cv2.COLOR_BGR2RGB) # 按照dataset中，原图是按照cv2的bgr直接读进来的


if __name__ == '__main__':
    ckpt_pth = './output/best_dist.pth'
    model = get_portraitnet_mobilenetv2()
    inferer = ModelInfer(ckpt_pth, model)
    inferer.infer()