import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt
import matplotlib
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
print(torch.__version__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from numpy import iscomplexobj, cov, trace
from scipy.linalg import sqrtm

batch_size = 400

inpv1 = torch.load("F:/home/t_gan/inpv01.pt", map_location='cpu')


class MyDataSet(Data.Dataset):


    def __init__(self, inputs):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
    def __len__(self):
        return self.inputs.shape[0]


    def __getitem__(self, index):
        return self.inputs[index]


train_dataset = MyDataSet(inpv1[0:32000, :, :, :])  # 32000：6：24：72
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=False,
                                           )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()

        self.linear = nn.Linear(100, 6912)
        self.linear_bn = nn.BatchNorm2d(256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.cbam1 = CBAM(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        self.deconv3 = nn.ConvTranspose2d(64, 6, 4, 2, 1)
        self.cbam3 = CBAM(6, ratio=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs, _ = x.size()
        x = self.linear(x)
        x = x.reshape([batch_size, -1, 3, 9])
        x = self.relu(self.linear_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.cbam1(x)
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        # x = self.cbam2(x)
        x = torch.tanh(self.deconv3(x))
        # x = self.cbam3(x)

        return x


class WGAN_D(nn.Module):
    def __init__(self):
        super(WGAN_D, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(256, affine=True)

        self.linear1 = nn.Linear(6912, 1024)
        self.linear2 = nn.Linear(1024, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.linear_in = nn.InstanceNorm1d(1, affine=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2_in(self.conv2(x)))
        x = self.leaky_relu(self.conv3_in(self.conv3(x)))
        x = x.reshape([batch_size, -1])
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = x.reshape(x.size(0), 1, -1)
        x = self.linear_in(x)
        x = self.linear2(x)
        return x.view(-1, 1).squeeze(1)


lambda_gp = 10


def compute_gradient_penalty(D, real_samples, fake_samples):
    eps = torch.FloatTensor(real_samples.size(0), 1, 1, 1).uniform_(0, 1).to(device)
    X_inter = (eps * real_samples + ((1 - eps) * fake_samples)).requires_grad_(True)
    d_interpolates = D(X_inter)
    fake = torch.full((real_samples.shape[0],), 1, device=device,
                      requires_grad=False)
    # 求梯度
    gradients = autograd.grad(outputs=d_interpolates,
                              inputs=X_inter,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True
                              )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penaltys


z_dimension = 100

D = WGAN_D().to(device)
G = WGAN_G().to(device)

num_epochs = 120
d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))

for epoch in range(num_epochs):
    for i, mini_batch in enumerate(train_loader):
        real_img = mini_batch.to(device)

        for ii in range(5):
            d_optimizer.zero_grad()

            real_out = D(real_img)

            z = torch.randn(batch_size, z_dimension).to(device)
            fake_img = G(z).detach()

            fake_out = D(fake_img)

            gradient_penalty = compute_gradient_penalty(D, real_img.data, fake_img.data)

            d_loss = -torch.mean(real_out) + torch.mean(fake_out) + gradient_penalty
            d_loss.backward()
            d_optimizer.step()

        for ii in range(1):
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, z_dimension).to(device)
            fake_img = G(z)
            fake_out = D(fake_img)
            g_loss = -torch.mean(fake_out)
            g_loss.backward()
            g_optimizer.step()
        if i % 200 == 0:
            y = fake_img.data
            fid_cur = 0
            torch.save(y, f"/home/t_gan/res/data1/input_{len(train_loader) * epoch + i}.pt")
            fake_res = fake_img.detach()
            fake_res = np.array(fake_res.cpu())
            for k in range(40):
                for j in range(6):
                    act1 = fake_res[k, j, :, :]

            if (fid_cur / 240) < fid_min:
                fid_min = fid_cur / 240
                torch.save(copy.deepcopy(G.state_dict()),
                           f"/home/t_gan/res/model1/CBAG_model_{len(train_loader) * epoch + i}_fid:{fid_min}.pth")

    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} ''D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epochs,
                                                                                                  d_loss.data,
                                                                                                  g_loss.data,
                                                                                                  real_out.data.mean(),
                                                                                                  fake_out.data.mean()))
