#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
import numpy as np
import datetime as dt
import sys, os, pickle, time
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from mpl_toolkits.basemap import *
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf
from scipy import spatial
from netCDF4 import Dataset
import datetime as dt

from ml_functions import read_csv_files, normalize_multivariate_data, log, get_features

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

def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    bs = brier_score_keras(obs, preds)
    ratio = (bs / climo)
    return climo

def auc(obs, preds):
    auc = tf.metrics.auc(obs, preds)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

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

def plot_forecast(predictions, prefix="", fhr=36):
    test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    cmap = ListedColormap(test)
    #cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    #print(predictions)

    #awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)

    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/ch_pk_files/rt2015_ch_CONUS.pk', 'r'))
    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/dav_pk_files/rt2015_ch_CONUS.pk', 'rb'))
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb')) 

    lats, lons = predictions['lat'].values, predictions['lon'].values
    x, y = m(lons, lats)

    # do something convoluted here to only plot each point once
    probmax = {}
    for i,p in enumerate(predictions['predict_proba'].values):
        thiskey = '%f%f'%(lats[i],lons[i])
        if thiskey in probmax:
            if p > probmax[thiskey]:
                probmax[thiskey] = p
        else:
           probmax[thiskey] = p

    # need to do this before calling text
    #m.set_axes_limits(ax=axes)

    for i,p in enumerate(predictions['predict_proba'].values):
        thiskey = '%f%f'%(lats[i],lons[i])
        thisvalue = probmax[thiskey]

        color = cmap(norm([thisvalue])[0])
        probmax[thiskey] = -999
        if x[i] < m.xmax and x[i] > m.xmin and y[i] < m.ymax and y[i] > m.ymin and thisvalue > 0.05:
        #if thisvalue >= 0.15:
            a = axes.text(x[i], y[i], int(round(thisvalue*100)), fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')
           # a = axes.text(x[i], y[i], int(round(thisvalue*100)), fontsize=12, ha='center', va='center', family='monospace', color=color, fontweight='bold')
    #a = m.scatter(x, y, s=50, c=predictions['predict_proba'].values, lw=0.5, edgecolors='k', cmap=cmap, norm=norm)

    ax = plt.gca()
    cdate = sdate + dt.timedelta(hours=fhr)
    sdatestr = (cdate - dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    edatestr = (cdate + dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    #plt.text(0,1.01,'Probability of severe within 120-km of a point valid %s - %s'%(sdatestr, edatestr), fontsize=14, transform=ax.transAxes)
    plt.text(0,1.01,'Max 4-h, 120-km all-severe NNPF over all forecast hours for WRF init %s'%sdate.strftime('%Y%m%d%H'), fontsize=14, transform=ax.transAxes)

    # ADD COLORBAR
    #cax = fig.add_axes([0.02,0.1,0.02,0.3])
    #cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    #cb.outline.set_linewidth(0.5)
    #cb.ax.tick_params(labelsize=10)

    # plot reflectivity
    plot_cref = False
    if plot_cref:
        initstr = sdate.strftime('%Y%m%d00')
        wrfcdate = cdate.strftime('%Y-%m-%d_%H_%M_%S')
        fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/%s/diags_d01_%s.nc'%(initstr,wrfcdate), 'r')
        lats = fh.variables['XLAT'][0,:]
        lons = fh.variables['XLONG'][0,:]
        cref = fh.variables['REFL_COM'][0,:]
        fh.close()

        x, y = m(lons, lats)
        plt.contourf(x, y, cref, levels=[35,1000], colors='k', alpha=0.5)

    plt.savefig('forecast%s.png'%prefix, dpi=150)

def make_labels():
    #labels   = ((df['hail_rptdist'+twin] < d) & (df['hail_rptdist'+twin] > 0)) |  \
    labels   = ((df['hailone_rptdist'+twin] < d) & (df['hailone_rptdist'+twin] > 0)) |  \
               ((df['wind_rptdist'+twin] < d) & (df['wind_rptdist'+twin] > 0)) | \
               ((df['torn_rptdist'+twin] < d) & (df['torn_rptdist'+twin] > 0))

    labels_wind     =  ((df['wind_rptdist'+twin] < d) & (df['wind_rptdist'+twin] > 0))
    labels_hailone  =  ((df['hailone_rptdist'+twin] < d) & (df['hailone_rptdist'+twin] > 0))
    labels_torn     =  ((df['torn_rptdist'+twin] < d) & (df['torn_rptdist'+twin] > 0))
    labels_sighail  =  ((df['sighail_rptdist'+twin] < d) & (df['sighail_rptdist'+twin] > 0))
    labels_sigwind  =  ((df['sigwind_rptdist'+twin] < d) & (df['sigwind_rptdist'+twin] > 0))

    # labels for multi-class neural network
    if multiclass: labels = np.array([ labels, labels_wind, labels_hailone, labels_torn, labels_sighail, labels_sigwind ]).T
    else: labels = np.array([ labels ]).T

    return labels

def compute_optimal_uh():
    predictions_gridded_uh, labels_gridded = make_gridded_forecast(uh120_all, labels_all[:,0], dates_all, fhr_all)

    optimal_uh_warmseason, num_rpts_warm = pickle.load(open('/glade/work/sobash/NSC_objects/optimal_uh_warmseason', 'rb'))
    optimal_uh_coolseason, num_rpts_cool = pickle.load(open('/glade/work/sobash/NSC_objects/optimal_uh_coolseason', 'rb'))

    #months_all = months_all.reshape((num_dates, num_fhr, -1))
    #months_all = months_all[:,0,0]
    #for k,m in enumerate(months_all):

    m = sdate.month  
    print(num_fhr) 
    if m in [4,5,6,7]: this_uh = ( predictions_gridded_uh >= optimal_uh_warmseason[:num_fhr,:] )
    else:              this_uh = ( predictions_gridded_uh >= optimal_uh_coolseason[:num_fhr,:] )
        
    this_uh = this_uh.reshape((num_fhr,-1))[:,thismask]

    uh_binary = np.array(this_uh).flatten()
    return uh_binary

def output_csv(fname):
    # output probabilities for one forecast

    # output 80-km grid locations
    #awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    #lons, lats = awips.makegrid(93, 65)
    #np.savetxt('grid.out', np.array([lons.flatten(), lats.flatten(), thismask]).T, fmt='%.3f,%.3f,%.0d', header='lon,lat,mask')

    idxarray = np.tile(np.arange(0,93*65)[np.newaxis,:], (num_fhr,1)).flatten()
    #fhrarray = np.tile(np.arange(1,num_fhr+1)[:,np.newaxis], (1,93*65)).flatten()
    fhrarray = np.tile(unique_fhr[:,np.newaxis], (1,93*65)).flatten()
    usmask   = np.tile(thismask[np.newaxis,:], (num_fhr,1)).flatten()

    all_probs = 100*predictions_all_gridded.reshape((7,-1)) #should become (7,num_fhr*93*65)
    all_probs = np.where(all_probs<1, 0, all_probs)

    # want to only include areas where ANY prob is non-zero and within US mask area (smoothed UH likely has probs outside of US, maybe ML too)
    probmask =  ( np.any(all_probs, axis=0) & usmask )

    np.savetxt(fname, np.array([idxarray[probmask], fhrarray[probmask], \
                                all_probs[0,probmask], all_probs[1,probmask], all_probs[2,probmask], \
                                all_probs[3,probmask], all_probs[4,probmask], all_probs[5,probmask], all_probs[6,probmask]]).T, \
                            delimiter=',', fmt='%.0d', comments='', header='idx,fhr,psvr,pwind,phail,ptorn,psighail,psigwind,puh')

    #probarray =100*predictions_gridded[fmask,:,:].flatten()
    #probmask = (probarray >= 1)
    #np.savetxt('test.out', np.array([idxarray[probmask], fhrarray[probmask], probarray[probmask]]).T, delimiter=',', fmt='%.0d', header='idx,fhr,prob')

### NEURAL NETWORK PARAMETERS ###

nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 10, \
              'report_window_space':[ 40,120 ], 'report_window_time':[ 2 ] }

rf_params = { 'ntrees': 100, 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 10 }
#rf_params = { 'ntrees': 200, 'max_depth': 20, 'min_samples_split': 20 }

#years         =  [2011,2012,2013,2014,2015,2016] #k-fold cross validation for these years
#years         =  [ int(sys.argv[3]) ]
model         =  sys.argv[2]
mem           =  1

plot          =  True

multiclass    =  True
thin_data     =  True
thin_fraction =  0.5
smooth_probs  =  False
smooth_sigma  =  1
simple_features = True
dataset = 'HRRR'
scaling_dataset = 'HRRR'
scaling_dataset = 'HRRRX'
subset = 'all'
use_nschrrr_features = False
expname = 'epoch30'
expname = 'hrrrv4-epoch30'

#mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask  = pickle.load(open('/glade/work/sobash/NSC_objects/HRRR/usamask_mod.pk', 'rb'))
thismask = mask.flatten()

trained_models_dir = '/glade/work/sobash/NSC_objects'
trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models_paper'
trained_models_dir = '/glade/work/sobash/NSC_objects/HRRR/trained_models'

#sdate   = dt.datetime(2020,3,2,0,0,0)
#edate   = dt.datetime(2020,3,2,0,0,0)
sdate = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H')
edate = sdate
dateinc = dt.timedelta(days=1)

##################################

if multiclass: numclasses = 6
else: numclasses = 1
twin = "_%dhr"%nn_params['report_window_time'][0]

features = get_features(subset, use_nschrrr_features)

log('Number of features %d'%len(features))
log(nn_params)
log(rf_params)

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'
df, numfcsts = read_csv_files(sdate, edate, dataset)

# restrict hrrr values to min/max range
#with open('hrrrv3_minmax') as fh:
#    for line in fh.read().splitlines()[1:]:
#        r = line.split(',')
#        if r[0] in ['fhr', 'xind', 'yind', 'lat', 'lon', 'year', 'month', 'hour', 'dayofyear']: continue        
#        field, minv, maxv = r[0], float(r[1]), float(r[2])
#        df.loc[df[field] > maxv, field] = maxv
#        df.loc[df[field] < minv, field] = minv
#        #print(field, df[field].min(), df[field].max())

mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask = mask.reshape((65,93))

# what forecast points to use
forecast_mask = ( (df['fhr'] >= 1) )
these_points = df[forecast_mask]
year = sdate.year

#if year > 2016: year = 2016 #use NN without 2016 for any date past 2016
#year = 2017
year = 2020
year = 2021
#year = 2016

classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
hazard_type = 0

fhr_all  = df[forecast_mask]['fhr'].values
dates_all = df[forecast_mask]['Date'].values
unique_forecasts, unique_fhr = np.unique(dates_all), np.unique(fhr_all)
smooth_sigma, num_dates, num_fhr = [2.0], len(unique_forecasts), len(unique_fhr)  

for d in nn_params['report_window_space']:
    labels_all = make_labels()

    if model == 'nn':
        scaling_values = pickle.load(open('/glade/work/sobash/NSC_objects/HRRR/scaling_values_all_%s.pk'%scaling_dataset, 'rb'))
        norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)
        
        this_in_data = norm_in_data[forecast_mask,:]
        
        dense_model = None
        model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f_%s.h5'%(trained_models_dir,year,d,twin,\
                                                                      nn_params['num_neurons'][0],nn_params['dropout'],expname)
        dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
        print('Using', model_fname)
       
        predictions_all = dense_model.predict(this_in_data)

        these_points['predict_proba'] = predictions_all[:,hazard_type]    

    if model == 'rf':         
        this_in_data = these_points[features].values

        model_fname = '%s/rf_gridded_%s_%dkm%s_n%d_d%d_m%d_l%d.pk'%(trained_models_dir,year,d,twin,rf_params['ntrees'],rf_params['max_depth'],rf_params['min_samples_split'],rf_params['min_samples_leaf'])
        rf = pickle.load(open(model_fname, 'rb'))

        predictions_all = rf.predict_proba(this_in_data)
        predictions_all = np.array(predictions_all)[:,:,1].T #needs to be in shape (examples,classes)

        these_points['predict_proba'] = predictions_all[:,hazard_type]

    if d == 40 and twin == '_2hr': uh120_all  = df[forecast_mask]['UP_HELI_MAX-N1T5'].values
    if d == 80 and twin == '_2hr': uh120_all  = df[forecast_mask]['UP_HELI_MAX80-N1T5'].values
    if d == 120 and twin == '_2hr': uh120_all = df[forecast_mask]['UP_HELI_MAX120-N1T5'].values

    #uh_binary = compute_optimal_uh()
    #predictions_gridded_uh, labels_gridded  = make_gridded_forecast((uh_binary).astype(np.int32), labels_all[:,0], dates_all, fhr_all)

    # convert predictions into grid and add UH probs
    predictions_all_gridded = []
    for i in range(6):
        predictions_gridded, labels_gridded = make_gridded_forecast(predictions_all[:,i], labels_all[:,i], dates_all, fhr_all)
        predictions_all_gridded.append(predictions_gridded)
    
    predictions_gridded_uh, labels_gridded  = make_gridded_forecast((uh120_all>75).astype(np.int32), labels_all[:,0], dates_all, fhr_all)
    predictions_gridded_uh_smoothed = smooth_gridded_forecast(predictions_gridded_uh) 
    predictions_all_gridded.append(predictions_gridded_uh_smoothed[0,:])
    predictions_all_gridded = np.array(predictions_all_gridded)

    log('Outputting CSV')
    fname = 'probs_%s_%s_%dkm.out'%(model,sdate.strftime('%Y%m%d%H'),d)
    output_csv(fname)
    
    log('Outputting grib')
    import write_grib as wg
    #ofile = "./grib/hrrr_ml_%dkm_2hr_%s.grb"%(d,sdate.strftime('%Y%m%d%H'))
    #wg.write_grib(predictions_gridded[0,:,:], sdate, 0, ofile) #write all forecast hours into one file

    # write grib for each hour for each scale, combine hazard probs into one file
    for i,f in enumerate(unique_fhr):
        ofile = "./grib/hrrr_ml_%dkm_2hr_%sf%03d.grb"%(d,sdate.strftime('%Y%m%d%H'), f)
        wg.write_grib(predictions_all_gridded[:6,0,i-1,:], sdate, f, ofile) #output first 6 elements (ignore UH)

log('Finished')
