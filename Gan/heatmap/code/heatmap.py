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
import pandas as pd
import torch
import scipy.io as io

deg = u'\xb0'
CH_list = ['C30H30', 'C30H50', 'C50H30', 'C50H50']

shape1 = ['gradsameleadmean', 'samelead', 'bm1gradmean.gdat', 'lead1']
shape2 = ['alltransgrad', 'singlemodel', 'bm1mon12gradmean.gdat', 'lead1mon12']
tmp = shape2
load_torch = torch.load("F:\output.pt",map_location='cpu')


for l in range(12):
    for i in range(6):
        df = load_torch[l, i, :, :]
        heat_each1 = df
        heat_each1 = heat_each1.numpy()
        heat_each1 = heat_each1 - np.min(heat_each1)

        ext_heatmap = heat_each1 / np.max(heat_each1)

        a, b = ext_heatmap.max(), ext_heatmap.min()
        a = 1 if a > 0.999 else a
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

        plt.title('[El Niño Heatmap]',
                  fontsize=8, y=0.962, x=0.5)
        cax1 = plt.axes([0.08, 0.28, 0.72, 0.013])
        cbar = plt.colorbar(cax=cax1, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6.5, direction='out', length=2, width=0.4, color='black')
        plt.subplots_adjust(bottom=0.10, top=0.9, left=0.08, right=0.8)
        plt.savefig("F:/ksh4800/"+str(l) +'..'+str(i)+ '.png', dpi=500, bbox_inches='tight')

        plt.close()