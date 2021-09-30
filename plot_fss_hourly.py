#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import sys
import matplotlib.ticker as mtick
from scipy.interpolate import InterpolatedUnivariateSpline
from sspf_helper import readNCLcm, truncate_colormap

mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16.0

fig = plt.figure(figsize=(7,4))
#gs = gridspec.GridSpec(2,1,height_ratios=[5,1])
numrows, numcols = 1, 1
numpanels = numrows*numcols
gs = gridspec.GridSpec(numrows,numcols)
gs.update(hspace=0.07)

if numpanels > 1: fontsize=6
else: fontsize=10

ylim=(0,1.0)
yticks=np.arange(0,0.21,0.1)
ylabel='Fractions Skill Score'

for n,ax in enumerate(gs):
        ax1 = plt.subplot(ax)

        ax1.grid(color='0.7', linewidth=0.25) #gridlines 
        
        # adjust axes labels/ticks/etc
        #ax1.tick_params(bottom='on', axis='both', direction='out', labelsize=9)
        ax1.tick_params(bottom='on', axis='both', width=0.5, direction='out', labelsize=fontsize-2, labelbottom='off')
        fig.suptitle('', fontsize=fontsize+2)
        if n > (numpanels)-numcols-1:
            ax1.tick_params(labelbottom='on')
            ax1.set_xlabel('Forecast Hour', fontsize=fontsize-1, labelpad=4)
        if n%numcols < 1:
            ax1.set_ylabel(ylabel, fontsize=fontsize-1)
        for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(0.5)
        ax1.set_ylim((0,1.0))
        #ax1.set_xlim((0,132))
        #ax1.set_xticks(range(0,133,6))
        ax1.set_xlim((0,36))
        ax1.set_xticks(range(0,37,6))
       
        #verif_file = 'scores_MPAS_UP_HELI_MAX_obsall_hourly.nc' 
        verif_file = 'scores_NCAR2019_RANDOM-FOREST120km_obsall_hourly.nc' 
        fh = Dataset(verif_file, 'r')
        fss  = fh.variables['fss'][:]

        print fss.shape

        idxmax = np.argmax(fss, axis=1)
        print idxmax[5,:]       

        fssmax = np.amax(fss, axis=1)

        xrange = range(1,37)
        ax1.plot(xrange, fssmax[0,:], marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=2.5)
        ax1.plot(xrange, fssmax[5,:], marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=2.5)
        ax1.plot(xrange, fssmax[14,:], marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=2.5)

        thresh_list = fh.variables['threshList'][:]
        max_thresh = [ thresh_list[j] for j in idxmax[5,:] ]
        #for i in range(1,133,3): ax1.text(i, 0.01, int(max_thresh[i]), ha='center', fontsize=4)
        for i in range(1,37,3): ax1.text(i, 0.01, int(max_thresh[i]), ha='center', fontsize=4)
 
        ax1.axhline(y=0.5,xmin=0,xmax=132,color='k',linewidth=0.5) 
        #for i in range(0,121,24): ax1.axvspan(i,i+12,ymin=0,ymax=1,facecolor='gray',alpha=0.25)
        for i in range(0,37,24): ax1.axvspan(i,i+12,ymin=0,ymax=1,facecolor='gray',alpha=0.25)

        ax1.text(0, 1.0, verif_file, va='bottom', ha='left', fontsize=4)
 
plt.savefig('fss.pdf')
