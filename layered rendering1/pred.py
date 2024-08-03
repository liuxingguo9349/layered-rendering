import torch
import torch.nn as nn
import torch.utils.data as Data
import copy
import sys
import os
import math
import numpy as np
from netCDF4 import Dataset

from model import CNN, train_one_epoch, train_one_epoch_noupdate_lr, MyDataSet, create_lr_scheduler

devices = 'cudanum'
epochs = 300
batch_size = 400
learning_rate = 5e-3  # 原cnn学习率
num_conv = 30
num_hidd = 30
lead_mon = 3
target_mon = 8


tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)

inp1 = Dataset('/home/input/CMIP5.input.36mon.1861_2001.nc', 'r')
inp2 = Dataset('/home/input/CMIP5.label.12mon.1863_2003.nc', 'r')

inpv1 = np.zeros((2961, 6, 24, 72), dtype=np.float32)
inpv1[:, 0:3, :, :] = inp1.variables['sst1'][0:2961, ld_mn1:ld_mn2, :, :]
inpv1[:, 3:6, :, :] = inp1.variables['t300'][0:2961, ld_mn1:ld_mn2, :, :]
inpv2 = np.zeros((2961), dtype=np.float32)
inpv2[:] = inp2.variables['pr'][0:2961, tg_mn, 0, 0]
soda = Dataset('/home/input/SODA.input.36mon.1871_1970.nc', 'r')
lablesoda = Dataset('/home/input/SODA.label.12mon.1873_1972.nc', 'r')
testsoda = np.zeros((100, 6, 24, 72), dtype=np.float32)
testsoda[:, 0:3, :, :] = soda.variables['sst'][0:100, ld_mn1:ld_mn2, :, :]
testsoda[:, 3:6, :, :] = soda.variables['t300'][0:100, ld_mn1:ld_mn2, :, :]
testlablesoda = np.zeros((100), dtype=np.float32)
testlablesoda[:] = lablesoda.variables['pr'][0:100, tg_mn, 0, 0]


def main():
    device = torch.device(devices if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    train_dataset = MyDataSet(inpv1, inpv2)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               )

    model = CNN(num_conv=num_conv, num_hidd=num_hidd).to(device)


    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-10)

    many = epochs // 10
    grad = np.zeros((many, 6, 24, 72))
    bestcor = -1
    bestloss = float("inf")
    cor = np.zeros((epochs))

    # 打印训练的参数和梯度情况
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    for epoch in range(epochs):
        print('grad shape:', grad.shape)
        train_loss = train_one_epoch_noupdate_lr(model=model,
                                                 optimizer=optimizer,
                                                 data_loader=train_loader,
                                                 device=device,
                                                 epoch=epoch)

        # 预测
        predict_dataset = torch.from_numpy(testsoda)

        model.eval()
        print('output shape:')
        output = torch.squeeze(model(predict_dataset.to(device))).cpu()
        result = output.detach().numpy()
        print('result min and max:', result.min(), result.max())
        print('mean:', result.mean())

        obs = testlablesoda
        thiscor = np.round(np.corrcoef(obs[:], result[:])[0, 1], 5)
        cor[epoch] = thiscor
        print('epoch,cor:', epoch, cor[epoch])
        cor.astype('float32').tofile('/home/document/lmont/cmip/chlist/ENnumber/lmontcmipcor.gdat')

        if not math.isnan(thiscor):
            if thiscor >= bestcor:
                torch.save(copy.deepcopy(model.state_dict()),
                           '/home/document/lmont/cmip/chlist/ENnumber/bestcmip_model.pth')
                bestcor = thiscor
                print('best cor:', bestcor)
            else:
                print('this cor is not the best')
                print('best cor is:', bestcor)

            torch.save(model.state_dict(), '/home/document/lmont/cmip/chlist/ENnumber/last_model.pth')
        else:
            if train_loss < bestloss:
                torch.save(copy.deepcopy(model.state_dict()),
                           '/home/document/lmont/cmip/chlist/ENnumber/bestcmip_model.pth')
                bestloss = train_loss

if __name__ == "__main__":
    main()