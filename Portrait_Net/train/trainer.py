import os, sys, pdb
from os import path as osp

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel 
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np
from d2l import torch as d2l
import cv2
from mmseg.core.evaluation import metrics

from configs.config import cfg
from train.dist_utils import is_main_process, reduce_value

class Trainer(object):
    def __init__(self, model) -> None:
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        model = model.to(device)
        self.writer = SummaryWriter(log_dir=cfg.log_dir, flush_secs=1)

        if cfg.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model,
                device_ids=[cfg.local_rank],  # note this parameter
                output_device=cfg.local_rank
            )
        self.model = model
        self.local_rank = cfg.local_rank
        self.device = device
        self.train_metric = d2l.Accumulator(3) # train_loss, train_acc, val_loss, val_acc
        self.val_metric = d2l.Accumulator(3)
        self.mode = None
        
    def to_cuda(self, batch):
        # pdb.set_trace()
        for i, _ in enumerate(batch):
            batch[i] = batch[i].to(self.device)
        return batch


    def train(self, data_loader, loss_fn, optimizer, epoch):
        if not cfg.distributed or (cfg.distributed and is_main_process()):
            print(f'Epoch {epoch}: training')

        self.model.train()
        metric = d2l.Accumulator(3)

        for i, batch in enumerate(data_loader):
            img_orig, img_aug, mask_gt, edge_gt = self.to_cuda(batch)
            
            num = img_orig.shape[0]
            pred_mask_orig, pred_edge_orig = self.model(img_orig)
            pred_mask_aug, pred_edge_aug = self.model(img_aug)
            
            loss = loss_fn(
                pred_mask_orig, pred_edge_orig, 
                pred_mask_aug, pred_edge_aug, 
                mask_gt, edge_gt
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss and accuracy,  不停地累计
            acc = self.get_accuracy(pred_mask_aug, mask_gt)
            loss = reduce_value(loss, average=True) # 获得不同设备之间的loss的mean
            acc = reduce_value(acc, average=True)
            metric.add(loss, acc, 1)

            if not cfg.distributed or (cfg.distributed and is_main_process()):
                print("Epoch[{}][{}/{}]\t" \
                    "Loss {:.3f}\t"\
                    "Accuracy {:.3f}\t"\
                    "Batch {}\t"\
                    "lr {:.6f}\t"\
                    "Cuda {}".format(
                            epoch, i, len(data_loader), metric[0] / metric[2], metric[1] / metric[2], num,
                            optimizer.param_groups[0]['lr'],
                            torch.cuda.current_device()
                        )
                    )
    
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Accuracy/train", acc, epoch)
        self.writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
        torch.cuda.synchronize(torch.cuda.current_device())  # 如果都是在training的话比较有用，但可能是因为卡算的速度本来就一样；但是training可能会在validation结束之前


    def val(self, data_loader, loss_fn, epoch):
        if not cfg.distributed or (cfg.distributed and is_main_process()):
            print(f'Epoch {epoch}: validation')

        self.mode = 'validation'
        self.model.eval()
        metric = d2l.Accumulator(3)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # pdb.set_trace()
                img_orig, img_aug, mask_gt, edge_gt = self.to_cuda(batch)
                num = img_orig.shape[0]
                pred_mask_orig, pred_edge_orig = self.model(img_orig)
                pred_mask_aug, pred_edge_aug = self.model(img_aug)
                
                loss = loss_fn(
                    pred_mask_orig, pred_edge_orig, 
                    pred_mask_aug, pred_edge_aug, 
                    mask_gt, edge_gt
                )

                loss = reduce_value(loss, True)
                acc = self.get_accuracy(pred_mask_aug, mask_gt, *(img_orig, img_aug, epoch))
                acc = reduce_value(acc, True)
                metric.add(loss, acc, 1) # loss, acc 取mean, 那么按照batch去做平均即可

                if not cfg.distributed or (cfg.distributed and is_main_process()):
                    # print('Epoch {}: Validation'.format(epoch))
                    print("Epoch[{}][{}/{}]\t" \
                        "Loss {:.3f}\t"\
                        "Accuracy {:.3f}\t"\
                        "Cuda {}".format(
                                epoch, i, len(data_loader), metric[0] / metric[2], metric[1] / metric[2], 
                                torch.cuda.current_device()
                            )
                        )
            self.writer.add_scalar('Loss/validation', loss, epoch)
            self.writer.add_scalar('Accuracy/validation', acc, epoch)

        torch.cuda.synchronize(torch.cuda.current_device()) 
        return acc

    def get_accuracy(self, pred_mask, mask_gt, *args):
        '''
        Args:
            pred_mask: [B, C, H, W]
            mask_gt: [B, H, W]
        Return 
        '''
        # [b,c,h,w] => [b,h,w] 因为gt里面label 0代表背景, 1代表mask，所以这里取1通道
        pred_mask = F.softmax(pred_mask, dim=1)[:, 1, ...] 
        pred_mask[pred_mask > 0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0

        if len(args) != 0:
            num_example = 5
            img_orig, img_aug, epoch = args
            pred = pred_mask.unsqueeze(dim=1).expand(-1, 3, -1, -1)

            input_pred = torch.concat(
                [img_orig[:num_example], img_aug[:num_example], pred[:num_example]], dim=0
            )
            grid = torchvision.utils.make_grid(input_pred, nrow=num_example)\
                                                .mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # cv2.cvtColor(grid, cv2.COLOR_RGB2BGR) # 这里应该是img_orig是bgr, img_aug是rgb
            # pdb.set_trace()
            
            self.writer.add_images('input and predict', grid, global_step=epoch, dataformats='CHW')
            # self.writer.add_images('predict', pred_mask, global_step=epoch, dataformats='NCHW')

        return self.calc_IoU(pred_mask, mask_gt)

    def calc_IoU(self, pred, gt):
        '''Calculate meanIoU between pred and gt'''
        # pdb.set_trace()
        # sum1 = pred + gt
        # sum1[sum1 > 0] = 1
        # sum2 = pred + gt
        # sum2[sum2 < 2] = 0
        # sum2[sum2 >= 2] = 1
        # if torch.sum(sum1) == 0:
        #     iou = 1
        # else:
        #     iou = 1.0 * torch.sum(sum2) / torch.sum(sum1)

        ''' 
        Use API in MMSegmentation. 
        More details can be found https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/core/evaluation/metrics.py
        Careful about ignore_index
        '''
        to_ndarray = lambda x: x.cpu().detach().numpy()
        pred = to_ndarray(pred)
        gt = to_ndarray(gt)
        # If ignore background index, which ignore the corresponding index in pred and gt, makes wrong pred less -> less union -> higher iou
        iou_portrait = metrics.mean_iou(pred, gt, num_classes=2, ignore_index=-1)['IoU'][1] 
    
        return iou_portrait
    



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


