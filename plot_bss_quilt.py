#!/usr/bin/env python

# PLOT FSS DATA COMPUTED USING compute_sspf_fss.py

from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import sys
import matplotlib.ticker as mtick

mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16.0

bss_nn = np.flipud(np.array([[0.07,0.15,0.19,0.26,0.33],
           [ 0.10,0.17,0.22,0.28,0.28],
           [ 0.14,0.19,0.23,0.25,0.19],
           [ 0.20,0.22,0.24,0.22,0.11],
           [ 0.24,0.23,0.23,0.18,0.09],
           [ 0.20,0.21,0.22,0.22,np.nan],
           [ 0.12,0.17,0.23,0.23,np.nan ]]))
bss_uh = np.flipud(np.array([[-0.29,-0.08,-0.01,0.02,0.04],
[-0.12,0.01,0.07,0.06,0.06],
[-0.09,0.02,0.09,0.06,-0.04],
[-0.04,0.07,0.11,0.08,-0.06],
[0.04,0.11,0.14,0.06,-0.1],
[0.08,0.11,0.12,0.07,np.nan],
[0.1,0.09,0.13,0.04,np.nan]]))
bss_diff = bss_nn - bss_uh
vals = bss_uh

roca_nn = np.flipud(np.array([[0.93,0.96,0.97,0.97,0.97],
[0.91,0.89,0.88,0.87,0.87],
[0.90,0.87,0.83,0.81,0.77],
[0.90,0.86,0.81,0.78,0.71],
[0.88,0.84,0.79,0.75,0.71],
[0.85,0.81,0.78,0.77,np.nan],
[0.78,0.77,0.79,0.79,np.nan]]))

roca_uh = np.flipud(np.array([[0.75,0.80,0.82,0.82,0.83],
[0.70,0.73,0.74,0.71,0.68],
[0.68,0.71,0.72,0.70,0.64],
[0.72,0.73,0.73,0.72,0.64],
[0.75,0.75,0.75,0.72,0.66],
[0.75,0.74,0.74,0.75,np.nan],
[0.74,0.73,0.74,0.77,np.nan]]))	
roca_diff = roca_nn - roca_uh

#vals = bss_diff
vals = roca_nn

# MAKE COLORMAP
levs = np.arange(-0.4,0.41,0.05)
levs = np.arange(0.6,1.01,0.05)

#test = readNCLcm('MPL_Greys')[::-1][25:] + readNCLcm('MPL_Reds')[10:-20]
#test = readNCLcm('MPL_RdGy')[::-1][5:-5]
#test = readNCLcm('MPL_RdGy')[::-1][10:-10]

#cmap = colors.ListedColormap(test)
cmap = plt.get_cmap('RdGy_r')
cmap = plt.get_cmap('Reds')
#cmap = truncate_colormap(cmap, 0.2, 1.0)

norm = colors.BoundaryNorm(levs, cmap.N, clip=True)

# MAKE PLOT
fig, ax = plt.subplots(1, figsize=(8,4))
#ax2 = ax.twiny()
#ax.imshow(fss, interpolation='nearest', cmap=cmap, norm=norm)

t = ax.pcolormesh(vals, cmap=cmap, norm=norm, edgecolors='k', lw=0.25)
for j, x in enumerate(vals.astype('|S5')):
    for i, y in enumerate(x):
        val = round(float(y), 2)
        ax.text(i+0.5, j+0.5, val, horizontalalignment='center', verticalalignment='center', size=14, color='k')
        
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelsize=12)
#ax2.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelsize=11) 

# tick labels at bottom
ax.set_xticks(np.arange(5)+0.5)
ax.set_xticklabels([0,10,20,30,40])
#ax.set_xticklabels(thresh)

ax.set_yticks(np.arange(7)+0.5)
ax.set_yticklabels([4000,3000,2000,1000,500,100,0])
#ax.xaxis.tick_top()

# tick labels at top
#ax.set_xticklabels([0,10,20,30,40])
#ax.set_xticks([0,10,20,30,40])

ax.set_ylabel(r'MUCAPE ($J kg^{-1}$)', fontsize=11)
ax.set_xlabel(r'SHR06 ($m s^{-1}$)', fontsize=11)

cbar = plt.colorbar(t, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Brier skill score', rotation=270, labelpad=15, fontsize=11)
plt.savefig('bss_nn.pdf', bbox_inches='tight')

