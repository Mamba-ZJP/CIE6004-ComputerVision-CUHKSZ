import os, sys, pdb

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

'''
Two losses:
    mask loss
    edge loss
'''

def get_loss(cfg):
    loss_fn = Loss(cfg)
    return loss_fn

class Loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.loss_mask = nn.CrossEntropyLoss()  # sigmoid + BCEloss
        self.loss_edge = FocalLoss(gamma=2)

        self.t = cfg.temperature
        self.edge_ratio = cfg.edgeRatio

    def forward(self, pred_mask_orig, pred_edge_orig, pred_mask_aug, pred_edge_aug, mask_gt, edge_gt):
        '''
        Args
            
        Return
        '''
        # assert x_aug.shape[1:] == torch.Size([2, 224, 224]), "x.shape[1:] is not [2, 224, 224]"

        loss_orig = self.loss_mask(pred_mask_orig, mask_gt) + \
                                self.edge_ratio * self.loss_edge(pred_edge_orig, edge_gt) 
        loss_aug = self.loss_mask(pred_mask_aug, mask_gt) + \
                                self.edge_ratio * self.loss_edge(pred_edge_aug, edge_gt)                                 

        loss_mask_stable = loss_KL(student_outputs=pred_mask_aug, 
                                    teacher_outputs=pred_mask_orig, T=self.t)
        loss_edge_stable = loss_KL(student_outputs=pred_edge_aug, 
                                    teacher_outputs=pred_edge_orig, T=self.t)
        return loss_orig + loss_aug + loss_mask_stable + loss_edge_stable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C

        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()



def loss_KL(student_outputs, teacher_outputs, T):
    """
    Code referenced from: 
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), 
                             F.softmax(teacher_outputs/T, dim=1)) * T * T
    return KD_loss