#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset, MFDataset
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
from mpl_toolkits.basemap import *
from scipy import spatial
import os
import pandas as pd

def readcm(name):
    '''Read colormap from file formatted as 0-1 RGB CSV'''
    rgb = []
    fh = open(name, 'r')
    for line in fh.read().splitlines(): rgb.append(map(float,line.split()))
    return rgb

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    fh = open('/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps/%s.rgb'%name, 'r')
    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def computeshr01(row):
    return np.sqrt(row['USHR1-potential_mean']**2 + row['VSHR1-potential_mean']**2)

def computeshr06(row):
    return np.sqrt(row['USHR6-potential_mean']**2 + row['VSHR6-potential_mean']**2)

def computeSTP(row):
    lclterm = ((2000.0-row['MLLCL-potential_mean'])/1000.0)
    lclterm = np.where(row['MLLCL-potential_mean']<1000, 1.0, lclterm)
    lclterm = np.where(row['MLLCL-potential_mean']>2000, 0.0, lclterm)

    shrterm = (row['shr06']/20.0)
    shrterm = np.where(row['shr06'] > 30, 1.5, shrterm)
    shrterm = np.where(row['shr06'] < 12.5, 0.0, shrterm)

    stp = (row['SBCAPE-potential_mean']/1500.0) * lclterm * (row['SRH01-potential_mean']/150.0) * shrterm
    return stp

def read_csv_files(r):
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        if r == '1km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_1km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        #elif r == '3km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        elif r == '3km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print 'Reading %s files'%(len(all_files))
    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    #df['shr01'] = df.apply(computeshr01, axis=1)
    #df['shr06'] = df.apply(computeshr06, axis=1)
    #df['stp'] = df.apply(computeSTP, axis=1)   
    #df['ratio'] = df['RVORT1_MAX_max'] / df['RVORT5_MAX_max']
 
    return df, len(all_files)

m = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)


sdate = dt.datetime(2011,4,1,0,0,0)
edate = dt.datetime(2011,5,1,0,0,0)
dateinc = dt.timedelta(days=1)

### PLOT 2D HISTOGRAM OF STORM NUMBERS
levels = np.arange(5,71,5)
#levels = np.arange(10,43,2)
cmap = ListedColormap(readcm('/glade/u/home/sobash/RT2015_gpx/cmap_rad.rgb')[1:14])
#cmap = ListedColormap(readNCLcm('prcp_1')[:16])
norm = BoundaryNorm(levels, cmap.N)

df, numfcsts = read_csv_files('3km')

# filter objects
df = df[df['UP_HELI_MAX01_max'] > 14.362]
#df = df[df['UP_HELI_MAX01_max'] > 100.03]
df = df.reset_index()

#fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011061700/diags_d01_2011-06-18_02_00_00.nc')
fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST/2011061700/diags_d01_2011-06-18_02_00_00.nc')
lats = fh.variables['XLAT'][0,:]
lons = fh.variables['XLONG'][0,:]
fh.close()

x_storm, y_storm = m(df['Centroid_Lon'].values, df['Centroid_Lat'].values)
x_grid, y_grid = m(lons, lats)

tree = spatial.KDTree(zip(x_grid.ravel(),y_grid.ravel()))
distances, closest = tree.query(zip(x_storm.ravel(),y_storm.ravel()))
nearest_y, nearest_x = np.unravel_index(closest, x_grid.shape)

print df.count()
for i in range(len(df)):
    thisdate = dt.datetime.strptime(df.iloc[i]['Run_Date'], '%Y-%m-%d %H:%M:%S')
    fhr = df.iloc[i]['Forecast_Hour']
    yyyymmddhh = thisdate.strftime('%Y%m%d%H')
    if yyyymmddhh != "2011042700": continue
 
    wrfvalidstr = (thisdate + dt.timedelta(hours=fhr)).strftime('%Y-%m-%d_%H_%M_%S')
    wrffile = "/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/%s/diags_d01_%s.nc"%(yyyymmddhh, wrfvalidstr)

    x_min, x_max = nearest_x[i]-48, nearest_x[i]+48
    y_min, y_max = nearest_y[i]-48, nearest_y[i]+48

    if x_min < 0 or x_max > lats.shape[1]: continue
    if y_min < 0 or y_max > lats.shape[0]: continue

    fh = Dataset(wrffile)
    cref = fh.variables['REFL_COM'][0,y_min:y_max+1, x_min:x_max+1]
    uh01 = fh.variables['UP_HELI_MAX01'][0,y_min:y_max+1, x_min:x_max+1]
    #wind = fh.variables['WSPD10MAX'][0,y_min:y_max+1, x_min:x_max+1]
    fh.close()
    
    print i, yyyymmddhh, fhr, nearest_x[i], nearest_y[i], cref.max()

    plt.clf()
    #cs1 = plt.contourf(x_grid[y_min:y_max+1,x_min:x_max+1], y_grid[y_min:y_max+1,x_min:x_max+1], wind, levels=levels, cmap=cmap, norm=norm, extend='max')
    cs1 = plt.contourf(x_grid[y_min:y_max+1,x_min:x_max+1], y_grid[y_min:y_max+1,x_min:x_max+1], cref, levels=levels, cmap=cmap, norm=norm, extend='max')
    cs1 = plt.contourf(x_grid[y_min:y_max+1,x_min:x_max+1], y_grid[y_min:y_max+1,x_min:x_max+1], uh01, levels=[10,100], alpha=0.5, colors='black', extend='max')
    plt.savefig('object_cref_%d.png'%i)
