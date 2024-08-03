import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scipy.io as io

load_torch = torch.load("F:/gan/layered_rendering.pt",map_location='cpu')
df=load_torch[0,0,:,:]
df=df.squeeze(0)
df = df.transpose(0,1)

height, width = df.shape

temperature = np.zeros((72, 24))

for i in range(height - 1):
    for j in range(width - 1):

        if df[i, j] < -1000:
            temperature[i, j] = 0
        else:
            temperature[i, j] = df[i, j]
X, Y = np.meshgrid(np.linspace(-55, 60, 24), np.linspace(0, 180 * 2, 72))

figure = plt.figure()
axis = figure.add_subplot(1, 1, 1)
color = axis.pcolormesh(Y, X, temperature, cmap='Spectral_r')
x = figure.colorbar(color)
x.set_label('Temperature')
plt.show()

