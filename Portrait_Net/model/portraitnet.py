import os, sys, pdb
import torch
from torch import nn
from torch.nn import functional as F
import typing
import numpy as np

def get_portraitnet_mobilenetv2():
    net = PortraitNet(backbone=MobileNetV2(), decoder=Decoder(), num_class=2)
    return net

def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w    


class PortraitNet(nn.Module):
    def __init__(self, backbone, decoder, num_class) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.mask_conv = nn.Conv2d(8, num_class, (1, 1), stride=1, padding=0) # binary classify
        self.edge_conv = nn.Conv2d(8, num_class, (1, 1), stride=1, padding=0) # binary classify
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                initial_weight = make_bilinear_weights(m.kernel_size[0], m.out_channels) # same as caffe
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), 'x is not a Tensor'
        assert len(x.shape) == 4, 'the dims of x is not 4'

        feats = self.backbone(x)
        out = self.decoder(feats)
        mask, edge = self.mask_conv(out), self.edge_conv(out)

        assert mask.shape[1] == edge.shape[1] == 2, "the channel dim of mask and edge is not 2"
        return mask, edge
        # return torch.cat([mask, edge], dim=1)


class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 1/2
            [6, 24, 2, 2], # 1/4
            [6, 32, 3, 2], # 1/8
            [6, 64, 4, 2], # 1/16
            [6, 96, 3, 1], # 1/16
            [6, 160, 3, 2], # 1/32
            [6, 320, 1, 1], # 1/32
        ]
        stages = []
        stages.append(conv_bn_relu6(3, 32, (3, 3), stride=2, padding=1)) # 1/2
        
        in_c = 32
        for expand_factor, out_c, repeat, s in interverted_residual_setting:
            layers = []
            for i in range(repeat):
                stride = s if i == 0 else 1  # only the first block in this stage
                layers.append(
                    InvertedResidual(in_c, out_c, stride=stride, expand_factor=expand_factor)
                )
                in_c = out_c
            stages.append(nn.Sequential(*layers))
        
        # self.features = nn.Sequential(*layers)
        # self.layers = layers  # 没有把这个layers放到sequential里导致没有放到cuda上，这个list类成员实际上没有进行注册: register_parameter()，可能换成继承了Module的类会自动注册成parameter
        self.stages = nn.Sequential(*stages)
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i != 0:
                out.append(x)
        return out  # setting is 7, but paper has only 5 parts


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # the deconv still uses the depthwise deconv
        decoder_channel_setting = [320, 96, 32, 24, 16, 8]
        self.down_sample_stage = [6, 4, 2, 1, 0]
        stages = []
        in_c = decoder_channel_setting[0]
        for ch in decoder_channel_setting[1:]:
            out_c = ch
            stages.append(nn.Sequential(
                TransitionModule(in_c, out_c),
                nn.ConvTranspose2d(out_c, out_c, (4, 4), groups=out_c, stride=2, padding=1, dilation=1),  # depthwise deconv
            ))
            in_c = out_c
        self.stages = nn.Sequential(*stages)
    
    def forward(self, X):
        
        for i, stage in enumerate(self.stages):  # module() returns all modules that derives from nn.Module: conv2d and so on
            if i != 0:
                out = stage(out + X[self.down_sample_stage[i]])
            else:
                out = stage(X[self.down_sample_stage[0]])
        return out


class TransitionModule(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.left_branch = nn.Sequential(
            nn.Conv2d(in_c, out_c, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.right_branch = nn.Sequential(
            conv_bn_relu6(in_c, in_c, (3, 3), stride=1, padding=1, groups=in_c),
            conv_bn_relu6(in_c, out_c, (1, 1), groups=1), # 在第一个深度可分离卷积里面完成通道数减少

            conv_bn_relu6(out_c, out_c, (3, 3), stride=1, padding=1, groups=out_c),
            nn.Conv2d(out_c, out_c, (1, 1), groups=1),
            nn.BatchNorm2d(out_c, 1e-5)
        )
    
    def forward(self, x):
        out = self.left_branch(x) + self.right_branch(x)
        out = F.relu6(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_factor) -> None:
        super().__init__()

        self.res_connect = stride == 1 and in_c == out_c
        hidden_c = in_c * expand_factor

        layers = []
        if expand_factor != 1:
            layers.append(conv_bn_relu6(in_c, hidden_c, (1, 1), stride=1, groups=1))
        layers.extend([
            conv_bn_relu6(hidden_c, hidden_c, (3, 3), stride=stride, padding=1, groups=hidden_c),
            nn.Conv2d(hidden_c, out_c, (1, 1), stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_c)
        ])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        if self.res_connect:
            return out + x
        return out
            


def conv_bn_relu6(in_c, out_c, kernel_s, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_s, 
            stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(num_features=out_c, eps=1e-5),
        nn.ReLU6(inplace=True)
    )


