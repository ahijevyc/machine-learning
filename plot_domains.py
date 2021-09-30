#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import *
import pickle

fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011042700/diags_d01_2011-04-27_12_00_00.nc', 'r')
lats = fh.variables['XLAT'][0,:]
lons = fh.variables['XLONG'][0,:]
ll_lat3, ll_lon3, ur_lat3, ur_lon3 = lats[0,0], lons[0,0], lats[-1,-1], lons[-1,-1]
fh.close()

fh = Dataset('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/2011042700/diags_d01_2011-04-27_12_00_00.nc', 'r')
lats1 = fh.variables['XLAT'][0,:]
lons1 = fh.variables['XLONG'][0,:]
ll_lat1, ll_lon1, ur_lat1, ur_lon1 = lats1[0,0], lons1[0,0], lats1[-1,-1], lons1[-1,-1]
fh.close()

dpi = 200
fig = plt.figure(dpi=dpi)
m = Basemap(projection='lcc', resolution='i', llcrnrlon=ll_lon3, llcrnrlat=ll_lat3, urcrnrlon=ur_lon3, urcrnrlat=ur_lat3, \
                lat_1=32, lat_2=46, lon_0=-101, area_thresh=10000)

m.drawcoastlines(linewidth=0.1)
m.drawstates(linewidth=0.1)
m.drawcountries(linewidth=0.1)

lw = 0.25
# PLOT 1KM DOMAIN
#x1, y1 = m(ll_lon3, ll_lat3)
#x2, y2 = m(ur_lon3, ur_lat3)

#m.plot([x1, x1], [y1, y2], color='k', linewidth=2)
#m.plot([x1, x2], [y2, y2], color='k', linewidth=2)
#m.plot([x2, x2], [y1, y2], color='k', linewidth=2)
#m.plot([x1, x2], [y1, y1], color='k', linewidth=2)

# PLOT 1KM DOMAIN
x1, y1 = m(ll_lon1, ll_lat1)
x2, y2 = m(ur_lon1, ur_lat1)

m.plot([x1, x1], [y1, y2], color='k', linewidth=lw)
m.plot([x1, x2], [y2, y2], color='k', linewidth=lw)
m.plot([x2, x2], [y1, y2], color='k', linewidth=lw)
m.plot([x1, x2], [y1, y1], color='k', linewidth=lw)

# PLOT VERIFICATION MASK
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
grid81 = awips.makegrid(93, 65, returnxy=True)
xorig, yorig = m(grid81[0], grid81[1])
x = (xorig[1:,1:] + xorig[:-1,:-1])/2.0
y = (yorig[1:,1:] + yorig[:-1,:-1])/2.0

#mask = pickle.load(open('../2013RT/maskgt105.pk', 'rb')).reshape((65,93))
mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb')).reshape((65,93))
m.pcolormesh(x, y, np.ma.masked_less(mask[1:,1:], 1.0), alpha=0.25, edgecolor='None', cmap=plt.get_cmap('Greys_r'), linewidth=0.05)
#m.pcolormesh(x, y, np.ma.masked_less(mask[1:,1:], 1.0), alpha=0.6, edgecolor='None', color='0.7', linewidth=0.05)

ax = plt.gca()
for l in ['left', 'right', 'top', 'bottom']: ax.spines[l].set_linewidth(0.4)

plt.savefig('domains.png', bbox_inches='tight')
