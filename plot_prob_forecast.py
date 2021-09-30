#!/usr/bin/env python

import datetime
import pickle, sys
import numpy as np
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.ndimage.filters import uniform_filter, gaussian_filter
import matplotlib as mpl
mpl.use('Agg')
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

def plot_2d_hist(predx, predy):
    cmap = plt.get_cmap('Greys')
    #norm = colors.BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
    norm = colors.BoundaryNorm(np.logspace(0,6,num=10), ncolors=cmap.N, clip=True)

    histox, bins = np.histogram(predx, bins=np.arange(0,1.1,0.1))
    histoy, bins = np.histogram(predy, bins=np.arange(0,1.1,0.1))

    plt.rcParams.update({'font.size': 14})

    # 2D histogram figure
    fig = plt.figure(figsize=(9,9))
    h = plt.hist2d(predx, predy, bins=np.arange(0,1.01,0.025), cmin=1, cmap=cmap, norm=norm)
    plt.style.use('seaborn-white')
    plt.plot([0,1], [0,1], color='k')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid()
    plt.xlabel('SSPF')
    plt.ylabel('NN')
    plt.savefig('hist2d.png')

    # 2D histogram figure
    fig = plt.figure(figsize=(8,8))
    plt.style.use('seaborn-white')
    n_sspf, bins, patches = plt.hist(predx, bins=np.arange(0,1.01,0.01), color='#1f77b4', alpha=0.5, histtype='stepfilled', edgecolor='none', log=True)
    n_ml, bins, patches = plt.hist(predy, bins=np.arange(0,1.01,0.01), color='#ff7f0e', alpha=0.5, histtype='stepfilled', edgecolor='none', log=True)
    plt.grid()
    plt.xlim((0,1))
    plt.ylim((1,1e8))
    plt.xlabel('Probability')
    plt.ylabel('Number of grid points')
    plt.savefig('hist_sspf.png')

    #print((predx<0.0001).sum(), (predy<0.0001).sum())
    #print(n_sspf, n_ml, n_ml/n_sspf)

    #print(histox, histoy)
    #print(np.histogram(predx-predy, bins=np.arange(-1,1.1,0.1)))

def readSevereClimo(fname, day_of_year, hr):
    from scipy.interpolate import RectBivariateSpline
    data = np.load(fname)
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    grid81 = awips.makegrid(93, 65, returnxy=True)
    x, y = awips(data['lons'], data['lats'])

    #spline = RectBivariateSpline(x[0,:], y[:,0], data['severe'][day_of_year-1,hr,:].T, kx=3, ky=3)
    #interp_data = spline.ev(grid81[2].ravel(), grid81[3].ravel())
    return np.reshape(interp_data, (65,93))

def make_gridded_forecast(predictions, labels, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)

    # just grid predictions for this class
    predictions = predictions.reshape((num_dates, num_fhr, -1))
    labels      = labels.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_predictions[i,j,thismask] = predictions[i,j,:]
            gridded_labels[i,j,thismask]      = labels[i,j,:]      
        #print(dt, gridded_predictions[i,:].max())

    # return only predictions for US points
    return (gridded_predictions.reshape((num_dates, num_fhr, 65, 93)), gridded_labels.reshape((num_dates, num_fhr, 65, 93)))

def grid_data(field):
    # convert 1d array into 4d array with shape (num_dates, num_fhr, 65, 93)
    gridded_field = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    field = field.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_field[i,j,thismask] = field[i,j,:]

    return gridded_field.reshape((num_dates, num_fhr, 65, 93))

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

def plot_forecast(data2d, fname='forecast.png'):
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb'))
    lons, lats = awips.makegrid(93, 65, returnxy=False)

    x, y = m(lons, lats)
    
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    test = readNCLcm('MPL_Greys')[35::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
    cmap = ListedColormap(test)
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)        

    labels_flatten = np.amax(labels_gridded[fmask,:], axis=0).flatten()
    x, y = x.flatten(), y.flatten()
    for i,b in enumerate(data2d.flatten()):
        color = cmap(norm([b])[0])

        if labels_flatten[i]: axes.scatter(x[i], y[i], color='black', marker='o', s=13**2, lw=1, facecolors='None', edgecolors='0.6')
        if not np.isnan(b) and not np.isinf(b) and thismask[i] and b>0.05:
        #if not np.isnan(b) and not np.isinf(b) and thismask[i] and b>5:
             
            #val = int(round(b))
            val = int(round(b*100))
            #if val > 99: val = 99
            #if val < -99: val = -99

            a = axes.text(x[i], y[i], val, fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')
            #a = axes.text(x[i], y[i], val, fontsize=10, ha='center', va='center', family='monospace', color='k', fontweight='bold')

    plt.savefig(fname, dpi=150)

def plot_reflectivity():
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb'))
    lons, lats = awips.makegrid(93, 65, returnxy=False)
  
    levels = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    refl_colors = readcm('/glade/u/home/sobash/RT2015_gpx/cmap_rad.rgb')[1:14]
    cmap = colors.ListedColormap(refl_colors)
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
 
    fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2012121900/diags_d01_2012-12-20_10_00_00.nc', 'r')
    lats = fh.variables['XLAT'][0,:]
    lons = fh.variables['XLONG'][0,:]
    cref = fh.variables['REFL_COM'][0,:]
    fh.close()

    x, y = m(lons, lats)
    plt.contourf(x, y, cref, levels=levels, cmap=cmap, norm=norm)

    plt.savefig('cref.png')

classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}    
i = 0
print(classes[i])
numclasses = 6
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
smooth_sigma = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
thismask = mask.flatten()

fcst_file  = 'predictions_nn_120km_2hr_noupperair'
climo_file = 'climo_severe_120km_2hr.npz'

print('reading data') 
predictions_all_nn, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open(fcst_file, 'rb'))

unique_forecasts, unique_fhr = np.unique(dates_all), np.unique(fhr_all)
num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)
print(unique_forecasts)

print('making date arrays')
dates_dt   = np.array([ datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in unique_forecasts ])
months_all  = np.array([ d.month for d in dates_dt ])
doy_unique  = np.array([ d.timetuple().tm_yday for d in dates_dt ])

dates_dt   = np.repeat(dates_dt, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))
months_all = np.repeat(months_all, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))
doy_all    = np.repeat(doy_unique, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))

print('reading climo')
data = np.load(climo_file)
climo = data['severe'][:]
climo_all = []
for doy in doy_unique:
    arr3 = np.append( climo[doy,:,:,:], climo[doy+1,:12,:,:], axis=0 )
    climo_all.append(arr3)
climo_all = np.array(climo_all)

predictions_all = predictions_all_nn

print('Verifying %d forecast points'%predictions_all.shape[0])

#predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>60), labels_all[:,i], dates_all, fhr_all)
#predictions_gridded_uh_smoothed        = smooth_gridded_forecast(predictions_gridded_uh)

optimal_uh = True
if optimal_uh:
    predictions_gridded_uh, labels_gridded = make_gridded_forecast(uh120_all, labels_all[:,i], dates_all, fhr_all)
    
    optimal_uh_warmseason, num_rpts_warm = pickle.load(open('optimal_uh_warmseason', 'rb'))
    optimal_uh_coolseason, num_rpts_cool = pickle.load(open('optimal_uh_coolseason', 'rb'))

    months_all = months_all.reshape((num_dates, num_fhr, -1))
    months_all = months_all[:,0,0]

    uh_binary = []
    for k,m in enumerate(months_all):
        if m in [4,5,6,7]: this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_warmseason )
        else:              this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_coolseason )
        this_uh = this_uh.reshape((num_fhr,-1))[:,thismask]
        uh_binary.append(this_uh)

    uh_binary = np.array(uh_binary).flatten()

### compute BSS for updraft helicity forecasts
print('computing BSS for UH forecasts')
predictions_gridded_uh, labels_gridded  = make_gridded_forecast((uh_binary).astype(np.int32), labels_all[:,i], dates_all, fhr_all)
predictions_gridded_uh_smoothed         = smooth_gridded_forecast(predictions_gridded_uh)

#apply lower threshold to predictions
predictions_gridded_uh_smoothed = np.where(predictions_gridded_uh_smoothed<0.001, 0.0, predictions_gridded_uh_smoothed)

### compute BSS for ML predictions
print('computing BSS for ML forecasts')
predictions_gridded, labels_gridded   = make_gridded_forecast(predictions_all[:,i], labels_all[:,i], dates_all, fhr_all)

#apply lower threshold to predictions
predictions_gridded = np.where(predictions_gridded<0.001, 0.0, predictions_gridded)
predictions_all     = np.where(predictions_all<0.001, 0.0, predictions_all)

### plot stuff here   
#preduh = predictions_gridded_uh_smoothed[8,:].reshape((num_dates,num_fhr,-1))[:,:,thismask]
#plot_2d_hist(preduh.flatten(), predictions_all[:,i])
#sys.exit()

### compute average difference of ML-SSPF forecast
#average_diff = (predictions_gridded - predictions_gridded_uh).mean(axis=(0))

gridded_uh_values = grid_data(uh120_all)
fmask = np.where( (unique_forecasts == '2012-12-19 00:00:00') )[0][0]
plot_forecast( np.amax( gridded_uh_values[fmask,:], axis=0 ), 'gridded_uh.png' )

# plot forecasts for given date
fmask = np.where( (unique_forecasts == '2012-12-19 00:00:00') )[0][0]
#fmask = np.where( (unique_forecasts == '2012-07-01 00:00:00') )[0][0]
plot_forecast( np.amax( predictions_gridded_uh_smoothed[8,fmask,:], axis=0 ) , 'predictions_uh.png' )
plot_forecast( np.amax( predictions_gridded[fmask,:], axis=0 ), 'predictions_ml.png' )

#plot_reflectivity()
