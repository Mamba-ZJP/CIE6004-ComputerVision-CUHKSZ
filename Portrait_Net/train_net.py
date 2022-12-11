import os, sys, pdb
from os import path as osp
import argparse

import torch
from torch import distributed as dist

from dataset.datasets import get_train_loader, get_valid_loader
from model.portraitnet import get_portraitnet_mobilenetv2
from train.trainer import Trainer
from train.optimize import get_optimizer
from train.lr_scheduler import get_lr_scheduler
from utils.loss import get_loss
from configs.config import cfg


def synchronize():
    if not dist.is_available():
        return 
    if not dist.is_initialized():
        return 
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def train(cfg, model):
    trainer = Trainer(model)

    train_loader = get_train_loader(cfg)
    valid_loader = get_valid_loader(cfg)
    optimizer = get_optimizer(cfg, net=model)
    loss_fn = get_loss(cfg)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    best_val_acc = 0.85
    for epoch in range(0, cfg.train.epoch):
        trainer.train(train_loader, loss_fn, optimizer, epoch)
        lr_scheduler.step()


        val_acc = trainer.val(valid_loader, loss_fn, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            save_pth = '{}_{}.pth'.format('best', 'dist' if cfg.distributed else '')
            if cfg.local_rank == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'best_val_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                }, osp.join(cfg.output_dir, save_pth))
            
        

if __name__ == '__main__':
    model = get_portraitnet_mobilenetv2()
    # print(os.environ['RANK']) # 相当于在每张卡上都运行了一遍
    # try distributed training firstly
    # pdb.set_trace()
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)  # 对当前进程启用所用的gpu
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://") # start the distributed backend
        synchronize()
    
    train(cfg, model)

