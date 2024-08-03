import torch
from model import CNN
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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
import scipy.io as io
import math
import numpy as np
print(torch.__version__)
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
for i in range(-20, 21):
    x = i / 10.0

    # 定义目标标量
    tar = x
    target = torch.ones(400) * tar
    # 初始化随机噪声
    noise = torch.randn(400, 100, requires_grad=True)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam([noise], lr=0.01)
    criterion = torch.nn.MSELoss()
    model_P.load_state_dict(torch.load('G:/month8/bestcmip_model.pth', map_location=torch.device('cpu')))
    model_G.load_state_dict(torch.load('G:/G_model_2560_fid_15.337425305337948.pth', map_location=torch.device('cpu')))

    # 迭代优化随机噪声
    for i in range(150):
        # 前向传播

        output = model_P(model_G(noise))

        # 计算损失
        loss = criterion(output, target.detach().float())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 输出损失值
        if i % 10 == 0:
            print('Iteration: {}, Loss: {:.4f}'.format(i, loss.item()))
            print('Iteration: {}, predict_mean: {:.4f}'.format(i, output.data.mean()))
    res = torch.tensor(model_G(noise)).float()
    print(res.size())
    # 输出微调后的随机噪声和模型预测值
    torch.save(res, f"F:/gan/che_shi/best_layered_rendering_{tar}.pt")



