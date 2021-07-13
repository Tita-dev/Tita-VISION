from typing_extensions import Concatenate
import torch
import os
import pandas as pd
import numpy as np
from torch.utils import *
import torch.nn as nn
import matplotlib.pyplot as plt

class MyImageDataset(torch.utils.data.dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        data_list = os.listdir(self.img_dir)
        data_list = [file for file in data_list if file.endswith(".jpg")]
        self.data_list = data_list
        

    def __len__(self):
        return len(self.data_list)


    def __add__(self, other):
        return Concatenate(self, other)

    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.img_dir, self.data_list[index]))

        ox = os.path.basename(os.path.join(self.img_dir)) # 인덱스에 맞는 파일명 가져오기
        if "O" in ox:
            label = "0"
        else : label = "1"

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        
        if img.dtype == np.uint8:
            img = img / 255.0
        
        data = tuple(img, label)
        
        data = self.to_tensor(data)
        
        return data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data
        

