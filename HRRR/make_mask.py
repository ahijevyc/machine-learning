#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
#import matplotlib.nxutils as nx
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import spatial
import pickle

# AWIPS 80-KM GRID POINTS
awips_proj = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95)
lats, lons, gridptsx, gridptsy = awips_proj.makegrid(93, 65, returnxy=True)
test = np.array((gridptsx.flatten(), gridptsy.flatten())).T

# READ IN US LAT/LON FILE
fh = open('/glade/u/home/sobash/2013RT/uspoints', 'r')
pts = fh.readlines()[0].split(',0')
pts2 = [a.split(',') for a in pts]
pts3 = np.array(pts2[:-1], dtype=float)
print(test.shape, pts3.shape)

# CONVERT US LAT LONS TO MAP PROJ COORDS 
temp = awips_proj(pts3[:,0], pts3[:,1])

# DETERMINE IF GRID POINTS ARE WITHIN US LAT LON BOUNDARY
usaPath = Path(np.array(temp).T)
mask = usaPath.contains_points(test)
#mask = nx.points_inside_poly(test, np.array(temp).T)KE FIGURE TO CHECK US MASK
print(mask.shape)

#mask = np.zeros((65,93))
mask = mask.reshape(lats.shape)
add_these_pts = [ (30,72), (30,73), (29,73), (27,73), (26,73), (34,75), (36,78), (19,68), (18,62), (40,79) ]
for p in add_these_pts:
    mask[p[0],p[1]] = 1
    print('adding %f %f'%(lats[p[0],p[1]], lons[p[0],p[1]]))
this_mask = np.ma.masked_array( mask, mask=np.logical_not(mask) ).reshape(lats.shape)

plt.figure(figsize=(12,6))
ax = plt.gca()
awips_proj.imshow(this_mask, interpolation='nearest')
awips_proj.drawcountries()
awips_proj.drawstates()
awips_proj.drawcoastlines()
plt.savefig('mask.png', bbox_inches='tight', dpi=120)

pickle.dump(mask.flatten(), open('usamask_mod.pk', 'wb'))
