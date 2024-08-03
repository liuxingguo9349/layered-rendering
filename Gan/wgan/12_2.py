import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt
import matplotlib
import os
from netCDF4 import Dataset
import netCDF4 as nc
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import scipy.io as io
import math
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

batch_size = 40

inp1 = nc.Dataset('/content/drive/dataset/input/CMIP5.input.36mon.1861_2001.nc','r')
inpv1 = torch.zeros([2961, 12, 6, 24, 72],dtype=torch.float32)
npsstbefore = np.array(inp1.variables['sst1'])
npsst = torch.Tensor(npsstbefore)
nphcbefore = np.array(inp1.variables['t300'])
nphc = torch.Tensor(nphcbefore)
npsst2 = torch.unsqueeze(npsst,1)
nphc2 = torch.unsqueeze(nphc,1)


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


train_dataset = MyDataSet(inpv1[0:32000,:,:,:])
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                       pin_memory=False,
                                           )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()

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


class WGAN_D(nn.Module):
    def __init__(self):
        super(WGAN_D, self).__init__()

        self.conv1 = nn.Conv2d(6, 128, 4, 2, 1)

        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(256, affine=True)

        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(512, affine=True)

        self.linear = nn.Linear(13824, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.weight_init()

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
        x = self.leaky_relu(self.conv2_in(self.conv2(x)))
        x = self.leaky_relu(self.conv3_in(self.conv3(x)))
        x = x.reshape([batch_size, -1])
        x = self.linear(x)

        return x.squeeze(1)

lambda_gp = 10

def compute_gradient_penalty(D, real_samples, fake_samples):

    eps = torch.FloatTensor(real_samples.size(0), 1, 1, 1).uniform_(0, 1).to(device)

    X_inter = (eps * real_samples + ((1 - eps) * fake_samples)).requires_grad_(True)
    d_interpolates = D(X_inter)
    fake = torch.full((real_samples.size(0),), 1, device=device)
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
outdir='./w_img'

num_epochs=30
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

os.makedirs(outdir, exist_ok=True)

for epoch in range(num_epochs):
    for i, mini_batch in enumerate(train_loader):
        real_img = mini_batch.to(device)

        for ii in range(5):
            d_optimizer.zero_grad()
            real_out = D(real_img)
            z = torch.randn(batch_size, z_dimension).to(device)
            fake_img = G(z)
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
                x = fake_img.data
                torch.save(x, f"/content/drive/MyDrive/wg/input_{len(train_loader) * epoch + i}.pt")


    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} ''D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epochs,
                                                                                                  d_loss.data,
                                                                                                  g_loss.data,
                                                                                                  real_out.data.mean(),
                                                                                                  fake_out.data.mean()))
    # 保存训练模型
torch.save(G.state_dict(), os.path.join(outdir, 'generator.pth'))
torch.save(D.state_dict(), os.path.join(outdir, 'discriminator.pth'))