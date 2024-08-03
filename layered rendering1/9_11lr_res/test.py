# 这是直接测试生成数据的nino3.4值的
import torch
from model_la import CNN
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt
import matplotlib
import os
import netCDF4 as nc
from netCDF4 import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import scipy.io as io
import math
import numpy as np
# print(torch.__version__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from numpy import iscomplexobj, cov, trace
from scipy.linalg import sqrtm

class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()

        self.linear = nn.Linear(100, 6912)
        self.linear_bn = nn.BatchNorm2d(256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 6, 4, 2, 1)

        self.relu = nn.ReLU()


    def forward(self, x):
        bs, _ = x.size()
        x = self.linear(x)
        x = x.reshape([400, -1, 3, 9])
        x = self.relu(self.linear_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

model_G = WGAN_G()
model_P = CNN()
model_P.load_state_dict(torch.load('H:/layerattention/7mon1best_model.pth', map_location=torch.device('cpu')))
file_path = 'F:/gan/res/che_shi/best_layered_rendering_2.pt'
loaded_data = torch.load(file_path)
tensor_data = torch.tensor(loaded_data)
x = tensor_data[0:30, :, :, :]

print("__________________________")
print(model_P(x))

