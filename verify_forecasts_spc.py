#!/usr/bin/env python

import datetime
import pickle, sys
import numpy as np
from sklearn import metrics
from scipy.ndimage.filters import uniform_filter, gaussian_filter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import *
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap,BoundaryNorm
from netCDF4 import Dataset

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

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

def apply_optimal_UH():
    # compute binary grid where UH exceeds spatially and temporally varying UH optimal threshold
    predictions_gridded_uh, labels_gridded = make_gridded_forecast(uh120_all, labels_all[:,hazard_idx], dates_all, fhr_all)

    optimal_uh_warmseason, num_rpts_warm = pickle.load(open('./trained_models_paper/optimal_uh_warmseason', 'rb'))
    optimal_uh_coolseason, num_rpts_cool = pickle.load(open('./trained_models_paper/optimal_uh_coolseason', 'rb'))

    this_months_all = months_all.reshape((num_dates, num_fhr, -1))
    this_months_all = this_months_all[:,0,0]

    uh_binary = []
    for k,m in enumerate(this_months_all):
        if m in [4,5,6,7]: this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_warmseason )
        else:              this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_coolseason )
        this_uh = this_uh.reshape((num_fhr,-1))[:,thismask]
        uh_binary.append(this_uh)

    uh_binary = np.array(uh_binary).flatten()

def print_scores(obs, fcst):
    obs, fcst = obs.astype(np.int).flatten(), fcst.astype(np.int).flatten()

    cm = metrics.confusion_matrix(obs, fcst)
    hits = cm[1,1]
    false_alarms = cm[0,1]
    misses = cm[1,0]
    correct_neg = cm[0,0]
    hits_random = (hits + misses)*(hits + false_alarms) / float(hits + misses + false_alarms + correct_neg)

    ets = (hits-hits_random)/float(hits + false_alarms + misses - hits_random)
    hss = 2*(hits*correct_neg - false_alarms*misses) / ( ( hits + misses ) * ( misses + correct_neg) + (hits + false_alarms) * ( false_alarms + correct_neg ) )
    bias = (hits+false_alarms)/float(hits+misses)
    pod = hits/float(hits+misses)
    far = false_alarms/float(hits+false_alarms)
    pofd = false_alarms/float(correct_neg + false_alarms)

    print (bias, pod, pofd, far, ets, hss)

##########################
### SET VARIABLES HERE ###
classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
hazard_idx = 0
numclasses = 6
compute_optimal_uh = True
print(classes[hazard_idx])

smooth_sigma = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
smooth_sigma = [2.0]

sfhr, efhr = 14, 34

fcst_file  = './trained_models_paper/predictions_nn_120km_2hr_all'
#fcst_file = 'predictions_nn_120km_2hr_NSC1km_all'
#fcst_file  = 'predictions_rf_40km_2hr_rt2020_test'
#fcst_file2  = 'predictions_rf_120km_2hr_rt2020'
#fcst_file  = 'predictions_nn_120km_2hr_uhonly_all'
#fcst_file  = 'predictions_nn_120km_2hr_envonly_all'
#fcst_file  = 'predictions_nn_120km_2hr_basicplus_all'

climo_file = 'climo_severe_120km_2hr_torn.npz'
##########################

#############################
### READ AND PROCESS DATA ###
print('reading data')

mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
thismask = mask.flatten()

# not converting to float32 due to small changes in computations?
predictions_all_nn, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open(fcst_file, 'rb'))
#predictions_all_rf, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open(fcst_file2, 'rb'))

# read in UH01 forecasts in separate file
#uh120_all = pickle.load(open('predictions_nn_120km_2hr_uh01', 'rb'))

unique_forecasts, unique_fhr = np.unique(dates_all), np.unique(fhr_all)
num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)

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

#predictions_all = (predictions_all_nn + predictions_all_rf) / 2.0
predictions_all = predictions_all_nn

##############################w
print('Verifying %d forecast points'%predictions_all.shape[0])

# compute binary grid where UH exceeds spatially and temporally varying UH optimal threshold
if compute_optimal_uh: apply_optimal_UH()

### convert lists to grids to enable smoothing, and then smooth UH forecasts
print('computing BSS for UH forecasts')
predictions_gridded_uh, labels_gridded   = make_gridded_forecast((uh120_all>20).astype(np.int32), labels_all[:,hazard_idx], dates_all, fhr_all)
#predictions_gridded_uh, labels_gridded  = make_gridded_forecast((uh_binary).astype(np.int32), labels_all[:,hazard_idx], dates_all, fhr_all)
predictions_gridded_uh_smoothed          = smooth_gridded_forecast(predictions_gridded_uh)
predictions_gridded, labels_gridded      = make_gridded_forecast(predictions_all[:,hazard_idx], labels_all[:,hazard_idx], dates_all, fhr_all)

### filter predictions by forecast hour
num_fhr = (efhr - sfhr) + 1
climo_all                            = climo_all[:,sfhr:efhr+1,:]
predictions_gridded, labels_gridded  = predictions_gridded[:,sfhr:efhr+1,:], labels_gridded[:,sfhr:efhr+1,:]
predictions_gridded_uh               = predictions_gridded_uh[:,sfhr:efhr+1,:]
predictions_gridded_uh_smoothed      = predictions_gridded_uh_smoothed[:,:,sfhr:efhr+1,:]

### apply lower threshold to predictions
predictions_gridded_uh_smoothed = np.where(predictions_gridded_uh_smoothed<0.001, 0.0, predictions_gridded_uh_smoothed)

# read in SPC forecasts for verification
all_spc = []
available_outlooks = []
for n,d in enumerate(dates_dt[:,0,0,0]):
    outlook_day, issue_time, type = 1, '0600', 'CAT'
    grid = np.zeros((65,93), dtype=np.uint8)
    
    fname = '/glade/p/mmm/parc/sobash/spc/%d/spc_sparse_fcst_day%d_%s%s.nc'%(d.year, outlook_day, d.strftime('%Y%m%d'), issue_time)
    print(fname)
    if os.path.exists(fname): fh = Dataset(fname, 'r')
    else: print('missing'); continue

    available_outlooks.append(n)

    if type+'prob' not in fh.variables:
        all_spc.append(grid)
        continue

    prob = fh.variables[type+'prob'][:]
    px = fh.variables[type+'x_pixel'][:]
    py = fh.variables[type+'y_pixel'][:]
    pc = fh.variables[type+'pixel_count'][:]
    
    # RECONSTRUCT GRID
    for i in range(px.size): grid[py[i],px[i]:px[i]+pc[i]] = prob[i]
    
    # spc added mrgl/enh at 15z 22 oct 2014 
    if d <= datetime.datetime(2014,10,22,0,0,0): all_spc.append( (grid >= 1) )
    else: all_spc.append( (grid >= 2) )

all_spc = np.array(all_spc)
print(all_spc.shape, all_spc.max(), all_spc.min())
print(len(available_outlooks))

predictions_gridded_max24 = np.amax(predictions_gridded, axis=1)
predictions_gridded_uh_smoothed_max24 = np.amax(predictions_gridded_uh_smoothed, axis=1)
labels_gridded_max24 = np.amax(labels_gridded, axis=1)

print_scores(labels_gridded_max24[available_outlooks,:], all_spc)
for p in np.arange(0.05,0.5,0.05):
    print_scores(labels_gridded_max24[available_outlooks,:], predictions_gridded_max24[available_outlooks,:] >= p)

