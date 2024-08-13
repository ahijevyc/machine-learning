#!/usr/bin/env python

import datetime
import pickle, sys
import numpy as np
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.ndimage.filters import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import *
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap,BoundaryNorm
from netCDF4 import Dataset

def readcm(name):
    '''Read colormap from file formatted as 0-1 RGB CSV'''
    rgb = []
    fh = open(name, 'r')
    for line in fh.read().splitlines(): rgb.append(list(map(float,line.split()))) #added list for python3
    return rgb

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in list(fh.read().splitlines()):
        if appending: rgb.append(list(map(float,line.split())))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def plot_reflectivity():
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    fig, axes, m = pickle.load(open('data/rt2015_ch_CONUS.pk', 'rb'))
    lons, lats = awips.makegrid(93, 65, returnxy=False)
  
    levels = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    refl_colors = readcm('/glade/u/home/sobash/RT2015_gpx/cmap_rad.rgb')[1:14]
    cmap = colors.ListedColormap(refl_colors)
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
 
    fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2012121900/diags_d01_2012-12-20_09_00_00.nc', 'r')
    lats = fh.variables['XLAT'][0,:]
    lons = fh.variables['XLONG'][0,:]
    cref = fh.variables['REFL_COM'][0,:]
    wspd = fh.variables['WSPD10MAX'][0,:]
    uh = fh.variables['UP_HELI_MAX'][0,:]
    fh.close()

    x, y = m(lons, lats)
    plt.contourf(x, y, uh, levels=levels, cmap=cmap, norm=norm)

    plt.savefig('cref.png', dpi=300)

plot_reflectivity()
