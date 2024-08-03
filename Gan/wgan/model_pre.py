from calendar import EPOCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import sys
import math
import numpy as np
from numpy import iscomplexobj, cov, trace
from scipy.linalg import sqrtm



class CNN(nn.Module):
    def __init__(self, num_conv, num_hidd, layer_scale_init_value=5e-1, spatial_scale_init_value=5e-1):
        super(CNN, self).__init__()
        self.pad1 = nn.ZeroPad2d((3, 4, 1, 2))
        self.conv1 = nn.Conv2d(6, num_conv, kernel_size=(4, 8), stride=1)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.pad2 = nn.ZeroPad2d((1, 2, 0, 1))
        self.conv2 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)
        self.pad3 = nn.ZeroPad2d((1, 2, 0, 1))  # 需要补的情况和pad2一样
        self.conv3 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)
        self.linear = nn.Linear(num_conv * 6 * 18, num_hidd)
        self.output = nn.Linear(num_hidd, 1)
        self.ssconv1 = nn.Parameter(spatial_scale_init_value * torch.ones((24, 72)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None

        self.ssconv2 = nn.Parameter(spatial_scale_init_value * torch.ones((12, 36)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None

        self.ssconv3 = nn.Parameter(spatial_scale_init_value * torch.ones((6, 18)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None

        self.lsconv1 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.lsconv2 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.lsconv3 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gammalinear = nn.Parameter(layer_scale_init_value * torch.ones((num_hidd,)),
                                        requires_grad=True) if layer_scale_init_value > 0 else None

        self.apply(self._init_weights)

    def truncated_normal_(self, tensor, mean=0, std=0.02):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)

            nn.init.constant_(m.bias, 0)


    def forward(self, x):

        # 第一层
        x = self.pad1(x)
        x = self.conv1(x)

        if self.ssconv1 is not None:

            x = self.ssconv1 * x

            if self.lsconv1 is not None:
                x = x.permute(0, 2, 3, 1)
                x = self.lsconv1 * x
                x = x.permute(0, 3, 1, 2)

        x = self.tanh(x)
        x = self.maxpool(x)
        x = self.pad2(x)
        x = self.conv2(x)
        if self.ssconv2 is not None:

            x = self.ssconv2 * x

            if self.lsconv2 is not None:
                x = x.permute(0, 2, 3, 1)
                x = self.lsconv2 * x
                x = x.permute(0, 3, 1, 2)

        x = self.tanh(x)
        x = self.maxpool(x)
        x = self.pad3(x)
        x = self.conv3(x)

        if self.ssconv3 is not None:

            x = self.ssconv3 * x

            if self.lsconv3 is not None:
                x = x.permute(0, 2, 3, 1)
                x = self.lsconv3 * x
                x = x.permute(0, 3, 1, 2)

        x = self.tanh(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.gammalinear is not None:
            x = self.gammalinear * x
        x = self.tanh(x)
        x1 = x.detach().cpu().numpy()


        output = self.output(x)
        output = output.squeeze(-1)

        return output


class MyDataSet(Data.Dataset):

    def __init__(self, inputs, lable):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
        self.lable = lable

    def __len__(self):
        return self.inputs.shape[0]


    def __getitem__(self, index):
        return self.inputs[index], self.lable[index]


def train_one_epoch_noupdate_lr(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        print('step:', step)
        datacmip, labels = data
        datacmip, labels = datacmip.to(device), labels.to(device)

        pred = model(datacmip)

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        print("[train epoch {}], batch mean loss: {:.8f}, lr: {:.8f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        print('step:', step)
        datacmip, labels = data
        datacmip, labels = datacmip.to(device), labels.to(device)

        pred = model(datacmip)
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        print("[train epoch {}], batch mean loss: {:.8f}, lr: {:.8f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step()

    return accu_loss.item() / (step + 1)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor


    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



