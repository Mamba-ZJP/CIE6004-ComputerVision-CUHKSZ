import sys, os, pdb
import argparse

import torch
import torch.distributed as dist

'''
Use `CUDA_VISIBILE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4  script.py (--args1)`
So you can read local_rank from `os.environ['LOCAL_RANK']`
'''


def init_distributed_mode(args):
    # 'RANK' 'WORLD_SIZE' 'LOCAL_RANK' will automatically be keys of os.environ, if use `torchrun` or `python -m torch.distributed`
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # print(args)
        
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    # print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=args.world_size, rank=args.rank)
    
    if get_world_size() == 1:
        return
    
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value) # 不同process对loss进行求和，value就变成了所有gpu的总和
        if average:
            value /= world_size

        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    init_distributed_mode(args)
    print('Done!')