import torch

_optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def get_optimizer(cfg, net, lr=None, weight_decay=None):
    params = []
    lr = cfg.train.lr if lr is None else lr
    

    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    # for key, val in net.named_parameters():
    #     if not val.requires_grad:
    #         continue
    #     params += [{'params': [val], 'lr': lr, 'weight_decay': weight_decay}]
    params = net.parameters()

    if 'adam' in cfg.train.optim:
        optimizer = _optimizers[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizers[cfg.train.optim](params, lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer