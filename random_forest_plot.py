#!/usr/bin/env python

from netCDF4 import Dataset, MFDataset
import numpy as np
from datetime import *
import time, os, sys
import matplotlib.pyplot as plt
from get_osr_gridded_new import *
from mpl_toolkits.basemap import *
import matplotlib.colors as colors
import scipy.ndimage as ndimage
import cPickle as pickle
import multiprocessing
from sspf_helper import readSSRsparsemax

def readSPCsparse(fname, type):
    grid = np.zeros((65,93), dtype=np.uint8)

    fh = Dataset(fname, 'r')
    if type+'prob' not in fh.variables:
        fh.close()
        return grid

    prob = fh.variables[type+'prob'][:]
    px = fh.variables[type+'x_pixel'][:]
    py = fh.variables[type+'y_pixel'][:]
    pc = fh.variables[type+'pixel_count'][:]

    # RECONSTRUCT GRID
    for i in range(px.size):
        grid[py[i],px[i]:px[i]+pc[i]] = prob[i]

    fh.close()
    return grid

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    import os
    rgb, appending = [], False
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def getlatlon():
    wrffile = "/glade/scratch/sobash/RT2013/grid_3km.nc"
    wrffile = "/glade/u/home/sobash/rt2015_grid.nc"
    f = Dataset(wrffile, 'r')
    lats = f.variables['XLAT'][0,:]
    lons = f.variables['XLONG'][0,:]
    ny, nx = lats.shape
    f.close()
    return (lats, lons)

def get_ssrs_osrs(thisdate):
    yyyymmdd  = thisdate.strftime("%Y%m%d%H")
    ssrfile = "/glade/work/sobash/SSR/ssr_sparse_grid81_%s_max/ssr_sparse_grid81_%s_%s_%s.nc"%(model,model,field,yyyymmdd)
    print ssrfile

    ssr_ensemble = readSSRsparsemax(ssrfile)
    ssr = np.amax(ssr_ensemble[:,win1:win2,:], axis=1)
    
    obs_start, obs_end = thisdate+timedelta(hours=win1-1) - gmt2cst, thisdate+timedelta(hours=win2-1) - gmt2cst
    osr = np.amax(get_osr_gridded(obs_start, obs_end, 93, 65, report_types=['wind', 'hailone', 'torn'], inc=24), axis=0)
    #osr = np.amax(get_osr_gridded(obs_start, obs_end, 93, 65, report_types=['torn'], inc=24), axis=0)
    return (ssr, osr)

###################################
##### SET PARAMETERS HERE ######
print 'Started at '+ time.ctime(time.time())

### DATES 
try:
    loopsdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
    loopedate = datetime.strptime(sys.argv[2], '%Y%m%d%H')
except:
    loopsdate = datetime(2016,6,2,0)
    loopedate = datetime(2016,6,2,0)
loopinc   = timedelta(hours=24)
gmt2cst   = timedelta(hours=6)
model     = sys.argv[3]
field     = sys.argv[4]
win1, win2 = 13, 37
###################################

### PLOT MAP ###
lats3, lons3 = getlatlon()
#m = Basemap(projection='lcc', llcrnrlon=lons[0,0], llcrnrlat=lats[0,0], urcrnrlon=lons[-1,-1], urcrnrlat=lats[-1,-1], lat_1=32.0, lat_2=46.0, lon_0=-101, resolution='i', area_thresh=10000.)
fig, axes, m  = pickle.load(open('/glade/u/home/sobash/RT2015_gpx/rt2015_ch_CONUS.pk', 'r'))
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)

x3, y3 = m(lons3, lats3)

#fig, axes = plt.subplots(1, 1, figsize=(12,12*0.6))

# PLOT MAP BACKGROUND FOR EACH SUBPLOT
#axes = axes.ravel()
#for ax in axes:
#m.drawcoastlines(linewidth=0.5, ax=axes)
#m.drawstates(linewidth=0.1, ax=axes)
#m.drawcountries(ax=axes)
for i in axes.spines.itervalues(): i.set_linewidth(0.5)

cmap = [[1,1,1]] + readNCLcm('perc2_9lev')

#a = m.imshow(field, cmap=cmap, norm=norm, interpolation='nearest')
grid81 = awips.makegrid(93, 65, returnxy=True)
xorig, yorig = m(grid81[0], grid81[1])
x = (xorig[1:,1:] + xorig[:-1,:-1])/2.0
y = (yorig[1:,1:] + yorig[:-1,:-1])/2.0

process = True
plot    = True



if process:
  ### READ IN SSRS AND OSRS IN PARALLEL ###
  nprocs         =  int(sys.argv[5])
  daysdiff       =  loopedate - loopsdate
  range_of_dates = []
  for i in range(0,daysdiff.days+1):
      thisdate = loopsdate + timedelta(days=i)
      file = "/glade/work/sobash/SSR/ssr_sparse_grid81_%s_max/ssr_sparse_grid81_%s_%s_%s.nc"%(model,model,field,thisdate.strftime('%Y%m%d%H'))
      if os.path.exists(file): range_of_dates.append(thisdate)
  chunksize      =  int(math.ceil(len(range_of_dates) / float(nprocs)))
  print len(range_of_dates)
  pool = multiprocessing.Pool(processes=nprocs)
  results = pool.map(get_ssrs_osrs, range_of_dates, chunksize)

  ssrs = np.array( [row[0] for row in results ] )
  osrs = np.array( [row[1] for row in results ] )

  mask  = pickle.load(open('/glade/u/home/sobash/2013RT/maskgt105.pk', 'r'))

  if field == 'UP_HELI_MAX01' and model == 'NSC1km': desired_threshold = 100.03
  if field == 'UP_HELI_MAX03' and model == 'NSC1km': desired_threshold = 566.2
  if field == 'UP_HELI_MAX' and model == 'NSC1km': desired_threshold = 861.0
  if field == 'UP_HELI_MAX' and model == 'NSC3km-12sec': desired_threshold = 170.95
  if field == 'UP_HELI_MAX01' and model == 'NSC3km-12sec': desired_threshold = 14.362
  if field == 'UP_HELI_MAX03' and model == 'NSC3km-12sec': desired_threshold = 105.0
 
  desired_threshold = 12.7

  ssrs = (ssrs>desired_threshold).sum(axis=0).mean(axis=0)
  osrs = osrs.sum(axis=0)
  print ssrs.shape, osrs.shape

if plot:
  #ssrs, osrs = pickle.load(open('ssrs_osrs_%s'%model, 'r'))

  #fh = MFDataset('/glade/scratch/sobash/diags_d02*.nc')
  #uh = fh.variables['UP_HELI_MAX'][:]
  #print uh.shape
  #fh.close()

  fontdict = dict(fontsize=11)
  bbox = dict(facecolor='#EFEFEF', linewidth=0.5, pad=10)
  
  mask = np.logical_not(mask.reshape(65,93))
  ssrs[mask] = 0.0
  osrs[mask] = 0.0

  #levels = [1] + range(2,20,2)
  #levels = np.arange(-90,91,10)
  #levels = np.arange(-10,11,1)
  #levels = np.arange(0,2.1,0.1)
  levels = np.arange(0.1,1.1,0.1)
  #levels = np.arange(1,50,5)
  #levels = np.arange(-5,6)

  test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
  #test = readNCLcm('perc2_9lev')[1::]
  cmap = colors.ListedColormap(test)
  cmap = plt.get_cmap('RdGy_r')
  norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

  ssrs_smooth = ndimage.filters.gaussian_filter(ssrs.astype(np.float), sigma=0)
  osrs_smooth = ndimage.filters.gaussian_filter(osrs, sigma=0.75)

  diff = ssrs_smooth - osrs_smooth
  ratio = np.nan_to_num(ssrs_smooth/osrs_smooth)
  normdiff = np.nan_to_num(diff/ssrs_smooth)
  ratio = np.where(osrs == 0, 0.0, ratio)

  #a = m.pcolormesh(x, y, np.ma.masked_less(ssrs[uh_thresh,1:,1:], 1), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)
  a = m.pcolormesh(x, y, np.ma.masked_less(ssrs_smooth[1:,1:], 1), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)
  #a = m.pcolormesh(x, y, np.ma.masked_less(osrs_smooth[1:,1:], 1), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)
  #a = m.pcolormesh(x, y, np.ma.masked_equal(diff[1:,1:], 0), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)
  
  rmse = np.sqrt((1/float(ssrs_smooth.size))*np.sum((ssrs_smooth - osrs_smooth)**2))
  print 'ssr total', ssrs[1:,1:].sum()
  print 'osr total', osrs.sum()
  print 'rmse', rmse

  #levels = [10,25,50,75,100,125,150,175,200,250,300,400,500]
  #cmap = colors.ListedColormap(readNCLcm('prcp_1')[1:15])
  #norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
  #a = m.contourf(x3, y3, np.amax(uh[13:37,:], axis=0), levels=levels, cmap=cmap, norm=norm)
  
  #a = m.pcolormesh(x, y, np.ma.masked_less(ssrs_smooth[1:,1:], 0.1), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)

  # plot OSRs
  osrs = osrs.astype(np.int)
  m.scatter(xorig[osrs>0], yorig[osrs>0], s=40, marker='x', color='k', lw=1)

  # ADD COLORBAR
  cax = fig.add_axes([0.02,0.1,0.02,0.3])
  cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
  cb.outline.set_linewidth(0.5)
  cb.ax.tick_params(labelsize=10)
  #cb.set_label('SSR - OSR')

  #plt.subplots_adjust(wspace=0.01, hspace=0.03)
  plt.savefig('test.png', bbox_inches='tight')
  #plt.savefig('test.png')
