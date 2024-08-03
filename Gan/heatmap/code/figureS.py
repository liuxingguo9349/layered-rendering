#!/usr/bin/env python
# coding: utf-8

from netCDF4 import Dataset
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import numpy as np
import cv2
import seaborn as sns

deg = u'\xb0'
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

shape1 = ['gradsameleadmean','samelead','bm1gradmean.gdat','lead1']
shape2 = ['alltransgrad','singlemodel','bm1mon12gradmean.gdat','lead1mon12']
tmp = shape2


for l in range(10):
    allgrad = np.zeros((12,6,24,72))
    for t in range(12):
        target = str(t+1)
        name = 'lead'+str(l+1) + 'spring'
        result = 'bm'+ str(l+1) + 'mon'+ target + 'gradmean.gdat'

        ipth1 = 'F:/pytorchCNN/'+tmp[0]+'/'
        ipth2 = 'F:/pytorchCNN/'+tmp[1]+'/springseparate/'

        f = open(ipth1 + result, 'r')

        heat_each = np.fromfile(f, dtype=np.float32).reshape(6, 24, 72)
        allgrad[t] = heat_each

    spring = allgrad[3:8]

    heat_each1 = np.mean(spring,axis=(0,1))
    print(heat_each1.max())
    heat_each1 = heat_each1 - np.min(heat_each1)
    heat_each1 = heat_each1 / (np.max(heat_each1))
    ext_heatmap = np.append(heat_each1,heat_each1[:,0:4],axis=1)

    std_heatmap = np.std(ext_heatmap, axis=0)

    mean_heatmap = np.mean(ext_heatmap, axis=0)

    a, b = ext_heatmap.max(), ext_heatmap.min()
    a = 1 if a > 0.999 else a
    print(a, b)
    cax = plt.imshow(ext_heatmap, cmap='RdBu_r', clim=[b, a],
                     interpolation="bicubic", extent=[0, 380, 60, -55], zorder=1)

    plt.gca().invert_yaxis()

    map = Basemap(projection='cyl', llcrnrlat=-55, urcrnrlat=59, resolution='c',
                  llcrnrlon=20, urcrnrlon=380)
    map.drawcoastlines(linewidth=0.2)
    map.drawparallels(np.arange(-90., 90., 30.), labels=[1, 0, 0, 0], fontsize=6.5,
                      color='grey', linewidth=0.2)
    map.drawmeridians(np.arange(0., 380., 60.), labels=[0, 0, 0, 1], fontsize=6.5,
                      color='grey', linewidth=0.2)
    map.fillcontinents(color='silver', zorder=2)
    space = '                                              '
    plt.title('Lead '+str(l+1)+' month '+ 'spring forecast'+ space + '[El Niño Heatmap]', fontsize=8, y=0.962, x=0.5)
    cax1 = plt.axes([0.08, 0.28, 0.72, 0.013])
    cbar = plt.colorbar(cax=cax1, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6.5, direction='out', length=2, width=0.4, color='black')
    plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
    plt.savefig(ipth2  + name +'.png', dpi=500,bbox_inches='tight')
    print(str(l+1),str(t+1))
    plt.close()


