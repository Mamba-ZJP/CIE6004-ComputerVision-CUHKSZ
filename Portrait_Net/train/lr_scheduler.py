import torch
from torch import nn



def get_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg_scheduler.milestones, gamma=cfg_scheduler.gamma
        )
    elif cfg_scheduler.type == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg_scheduler.step_size, gamma=cfg_scheduler.gamma
        )
    return lr_scheduler
