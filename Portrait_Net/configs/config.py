import os, sys, pdb

import yacs
from yacs.config import CfgNode as CN
import argparse

cfg = CN()


cfg.distributed = True
cfg.local_rank = 0
cfg.root = '/home/zjp/CV_lec/Portrait_Net'
cfg.log_dir = './log/'
cfg.img_dir = '/home/zjp/CV_lec/Portrait_Net/data/EG1800/Images'
cfg.ann_dir = '/home/zjp/CV_lec/Portrait_Net/data/EG1800/Labels'
cfg.output_dir = './output/'

cfg.gpus = list(range(4))
cfg.batch_size = 45

# train config 
cfg.train = CN()
cfg.train.epoch = 500
# cfg.train.optim = 'sgd'

cfg.train.optim = 'adam'
cfg.train.lr = 0.005
cfg.train.weight_decay = 0.0001
# cfg.train.scheduler = CN({'type': 'MultiStepLR', 'milestones': [50, 100, 150], 'gamma': 0.2})
cfg.train.scheduler = CN({'type': 'StepLR', 'step_size': 20, 'gamma': 0.8})


#? ====================================================================
#? for original PortraitNet setting
cfg.data_root = '/home/zjp/CV_lec/Portrait_Net/dataset'
cfg.file_root = '/home/zjp/CV_lec/Portrait_Net/dataset/select_data'
cfg.istrain = True
cfg.task = 'seg'
cfg.datasetlist = ['EG1800'] # 'support: [EG1800, supervisely_face_easy, ATR, MscocoBackground]'
# datasetlist: ['supervisely_face_easy'] # 'support: [EG1800, supervisely_face_easy, ATR, MscocoBackground]'

cfg.input_height = 224 # the height of input images
cfg.input_width = 224 # the width of input images

cfg.video = False # if exp_args.video=True, add prior channel for input images
cfg.prior_prob = 0.5 # the probability to set empty prior channel

cfg.addEdge = True # whether to add boundary auxiliary loss 
cfg.edgeRatio = 0.1 # the weight of boundary auxiliary loss
cfg.stability = True # whether to add consistency constraint loss
cfg.use_kl = True # whether to use KL loss in consistency constraint loss
cfg.temperature = 1 # temperature in consistency constraint loss
cfg.alpha = 2 # the weight of consistency constraint loss

# input normalization parameters
cfg.padding_color = 128
cfg.img_scale = 1
cfg.img_mean = [103.94, 116.78, 123.68] # BGR order, image mean
cfg.img_val = [0.017, 0.017, 0.017] # BGR order, image val

cfg.init = False # whether to use pretrain model to init portraitnet
cfg.resume = False # whether to continue training

cfg.useUpsample = False # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
cfg.useDeconvGroup = False # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d




parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", required=True, type=str)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

def parse_cfg(cfg, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.train.lr *= len(cfg.gpus)
    print('learning rate: {}, optimizer: {}'.format(cfg.train.lr, cfg.train.optim))
    return cfg


def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        cur_cfg = CN.load_cfg(f)
    
    cfg.merge_from_other_cfg(cur_cfg)

    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)

    return cfg

cfg = make_cfg(args)