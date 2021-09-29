import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d

class TitaModel(nn.Module):
    def __init__(self):
        super(TitaModel, self).__init__()

        # model (ResNet)
        

    def forward(self, x):

        return x

def conv_1(): # model start
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

def bottleneck_block(in_channels, mid_channels, out_channels, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
    ])
    return nn.Sequential(*layers)

