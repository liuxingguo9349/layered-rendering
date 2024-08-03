from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

m = Basemap(lon_0 = 0 , lat_0 = 0)
m.drawcoastlines(linewidth=0.3)
m.drawstates(color='b')
m.drawcountries()

m.drawparallels(np.arange(-90., 91., 10.),
labels=[1,0,0,0], fontsize=10,color='none')
m.drawmeridians(np.arange(-180., 181., 40.),
labels=[0,0,0,1], fontsize=10,color='none')

lons = np.linspace(70,140,71)
lats = np.linspace(0,60,61)
lon, lat = np.meshgrid(lons, lats)
x, y = m(lon, lat)

plt.show()