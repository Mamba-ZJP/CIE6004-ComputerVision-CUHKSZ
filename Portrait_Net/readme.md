# Re-implementation of PortraitNet
This repo is basically my personal practice to **use tensorboard, distributed training (dist_utils), combination with other library (mmsegmentation)**.

I also figure out a nice way to construct your AI project:
```bash
config
data
log
output
lib
    model
    dataset
    train
        trainer
        optimizer
        lr_scheduler
        ...
    ...
train_net.py
...
```
**Also, the file [dist_utils](./train/dist_utils.py) can be directly used as an independent package.**

## Training
```bash
bash train_net.sh
```