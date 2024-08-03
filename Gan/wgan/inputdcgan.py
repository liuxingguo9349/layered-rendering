import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import scipy.io as io
import math
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc

sst_size = [6, 24, 72]
batch_size = 40
z_size = 100
use_gpu = torch.cuda.is_available()

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.linear = nn.Linear(100, 13824)
        self.linear_bn = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 6, 4, 2, 1)

        self.relu = nn.ReLU()

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        bs, _ = x.size()
        x = self.linear(x)
        x = x.reshape([batch_size, -1, 3, 9])
        x = self.relu(self.linear_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.conv1      = nn.Conv2d(6, 128, 4, 2, 1)
        self.conv2      = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn   = nn.BatchNorm2d(256)
        self.conv3      = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn   = nn.BatchNorm2d(512)
        self.linear     = nn.Linear(13824, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.weight_init()
        self.sigmoid = nn.Sigmoid()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = x.reshape([batch_size, -1])
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Read Data (NetCDF4)
inp1 = nc.Dataset('/content/drive/MyDrive/gan/dataset/input/CMIP5.input.36mon.1861_2001.nc','r')
inpv1 = np.zeros((2961, 12, 6, 24, 72),dtype=np.float32)
npsst = np.array(inp1.variables['sst1'])
nphc = np.array(inp1.variables['t300'])
npsst2 = np.expand_dims(npsst,axis=1)
nphc2 = np.expand_dims(nphc,axis=1)
for i in range(0,12):
    inpv1[:,i,0:3,:,:]=npsst2[:,0,i:i+3,:,:]
    inpv1[:,i,3:6,:,:]=nphc2[:,0,i:i+3,:,:]

inpv1 = inpv1.reshape(-1,6,24,72)



class MyDataSet(Data.Dataset):
    def __init__(self, inputs):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
    def __len__(self):
        return self.inputs.shape[0]
    def __getitem__(self, index):
        return self.inputs[index]



train_dataset = MyDataSet(inpv1)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                       pin_memory=False,
                                           )

generator = generator()
discriminator = discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)


loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)


if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

# train
num_epoch = 200
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(train_loader):
        gt_inpl = mini_batch

        z = torch.randn(batch_size, z_size)

        if use_gpu:
            gt_inpl = gt_inpl.to("cuda")
            z = z.to("cuda")

        pred_inpl = generator(z)
        g_optimizer.zero_grad()
        recons_loss = torch.abs(pred_inpl - gt_inpl).mean()
        b = discriminator(pred_inpl)
        g_loss = recons_loss * 0.05 + loss_fn(b, labels_one)
        g_loss.backward()
        g_optimizer.step()
        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(gt_inpl), labels_one)
        fake_loss = loss_fn(discriminator(pred_inpl.detach()), labels_zero)
        d_loss = (real_loss + fake_loss)
        d_loss.backward()
        d_optimizer.step()

        if i % 10 == 0:
            print(
                f"step:{len(train_loader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 40 == 0:
            x = pred_inpl.data
            torch.save(x, f"input_{len(train_loader) * epoch + i}.pt")