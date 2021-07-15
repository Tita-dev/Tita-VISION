import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.activation import ReLU

class TitaModel(nn.Module):
    def __init__(self):
        super(TitaModel, self).__init__()

        # model (alexnet)
        self.titamodel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4), # 227 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1), # 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), 
        )

        self.fclayer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x