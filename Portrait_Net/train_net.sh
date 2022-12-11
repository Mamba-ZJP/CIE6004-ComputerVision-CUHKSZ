cfg_file=./configs/model_mobilenetv2_with_two_auxiliary_losses.yaml

# python -m torch.distributed.launch --nproc_per_node=2 train_net.py --cfg_file $cfg_file
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_net.py --cfg_file $cfg_file train.optim adam

# CUDA_VISIBLE_DEVICES=2 python train_net.py --cfg_file $cfg_file distributed False