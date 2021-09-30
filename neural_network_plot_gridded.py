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

def log(msg):
    print( time.ctime(time.time()), msg ) 

def read_csv_files():
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        if simple_features: csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_%s_d01_%s-0000.csv'%(dataset,yyyymmdd)
        else: csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_%s_d01_%s-0000.csv'%(dataset,yyyymmdd)
        csv_file = '/glade/work/sobash/NSC_objects/grid_data_NCARSTORM_d01_%s-0000.csv'%(yyyymmdd)
        print(csv_file)

        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    log('Reading %s forecasts'%(len(all_files)))

    #df = pd.concat((pd.read_csv(f, compression='gzip', dtype=type_dict) for f in all_files))
    df = pd.concat((pd.read_csv(f, dtype=type_dict) for f in all_files))

    #if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)   
    #if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    df['datetime']  = pd.to_datetime(df['Date'])
    #df['Run_Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    df['year']      = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['hour']      = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear

    return df, len(all_files)

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

def normalize_multivariate_data(data, scaling_values=None):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    log('%s, %s'%(data.shape, data.dtype))
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols, index=features)
        #for i in range(data.shape[-1]): scaling_values.loc[i, ["mean", "std"]] = [data[:, i].mean(), data[:, i].std()]
        for i in range(data.shape[-1]): scaling_values.loc[features[i], ["mean", "std"]] = [data[:, i].mean(), data[:, i].std()]

    for i in range(data.shape[-1]):
        #normed_data[:, i] = (data[:, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
        normed_data[:, i] = (data[:, i] - scaling_values.loc[features[i], "mean"]) / scaling_values.loc[features[i], "std"]

    return normed_data, scaling_values

def make_grid(df, predictions, labels):
    """ return 2d grid of probability or binary values """
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))

    unique_forecasts = df['datetime'].unique()
    unique_fhr = df['fhr'].unique()
    num_dates, num_fhr, num_classes = len(unique_forecasts), len(unique_fhr), predictions.shape[1]

    gridded_predictions = np.zeros((num_dates,num_fhr,65*93,num_classes), dtype='f')
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93,num_classes), dtype='f')

    for i, datetime in enumerate(unique_forecasts):
        for j, fhr in enumerate(unique_fhr):
            thismask = (df['datetime'] == datetime) & (df['fhr'] == fhr)
            gridded_predictions[i,j,mask,:] = predictions[thismask,:]
            gridded_labels[i,j,mask,:]      = labels[thismask,:]
        print(datetime, gridded_predictions[i,:].max())

    if smooth_probs:
        predictions = gridded_predictions.reshape((num_dates,num_fhr,65,93,num_classes))
        predictions = gaussian_filter(predictions, sigma=[0,0,smooth_sigma,smooth_sigma,0]).reshape((num_dates,num_fhr,-1,num_classes))

    # return only predictions for US points
    return predictions[:,:,mask,:].reshape((-1,num_classes))

def make_gridded_forecast(predictions, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    unique_forecasts, unique_fhr = np.unique(dates), np.unique(fhr)
    num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)

    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype='f')

    thismask = mask.flatten()

    # just grid predictions for this class
    predictions = predictions.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_predictions[i,j,thismask] = predictions[i,j,:]
        #print(dt, gridded_predictions[i,:].max())

    # return only predictions for US points
    return (gridded_predictions.reshape((num_dates, num_fhr, 65, 93)))

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

def train_random_forest():
    # set up random forest classifier
    rf = RandomForestClassifier(n_estimators=rf_params['ntrees'], max_depth=rf_params['max_depth'], min_samples_split=rf_params['min_samples_split'], \
                                oob_score=True, random_state=10, n_jobs=36)
    in_data = df[features].values

    # trained with unnormalized data
    rf.fit(in_data[train_indices], labels[train_indices])
    
    return rf

def init_neural_network():
    session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                               gpu_options=K.tf.GPUOptions(allow_growth=True),
                                               log_device_placement=False))
    K.set_session(session)

def train_neural_network():
    dense_model = None
    
    # Input layer
    dense_in = Input(shape=norm_in_data.shape[1:])

    # Hidden layers
    for n in range(0,nn_params['num_layers']):
        # First hidden layer
        dense = Dense(nn_params['num_neurons'][n], kernel_regularizer=l2())(dense_in)
        dense = Activation("relu")(dense)
        dense = Dropout(nn_params['dropout'])(dense)
        dense = BatchNormalization()(dense)

    # Output layer
    dense = Dense(numclasses)(dense)
    dense = Activation("sigmoid")(dense)

    # Creates a model object that links input layer and output
    dense_model = Model(dense_in, dense)

    # Optimizer object
    opt_dense = SGD(lr=nn_params['lr'], momentum=0.99, decay=1e-4, nesterov=True)

    # Compile model with optimizer and loss function
    if multiclass: dense_model.compile(opt_dense, loss="binary_crossentropy", metrics=[brier_score_keras, brier_skill_score_keras, auc])
    else: dense_model.compile(opt_dense, loss="mse", metrics=[brier_score_keras, brier_skill_score_keras, auc])

    # Train model
    dense_hist = dense_model.fit(norm_in_data[train_indices], labels[train_indices],
                             batch_size=1024, epochs=nn_params['num_epochs'], verbose=1,
                             validation_data=(norm_in_data[test_indices], labels[test_indices]))

    return (dense_hist, dense_model)

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

def print_scores(fcst, obs, rptclass):
    # print scores for this set of forecasts
    # histogram of probability values
    print(np.histogram(fcst))

    # reliability curves
    true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=10)
    for i in range(true_prob.size): print(true_prob[i], fcst_prob[i])

    # BSS
    bss_val = bss(obs, fcst)
    print(bss_val)

    # ROC auc
    auc = metrics.roc_auc_score(obs, fcst)
    print(auc)

    # output statistics
    if output_stats:
        model_name = model_fname.split('/')[-1]
        fh = open('%s_validation_fhr13-36_%s'%(model,rptclass), 'a')
        rel_string = [ '%.3f, %.3f'%(t,f) for t, f in zip(true_prob, fcst_prob) ]
        rel_string = ', '.join(rel_string)
        print(rel_string)
        fh.write('%s %s, %.3f, %.3f, %s\n'%(smooth_probs, model_name, bss_val, auc, rel_string))
        fh.close() 

### NEURAL NETWORK PARAMETERS ###

nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 10, \
              'report_window_space':[ int(sys.argv[1]) ], 'report_window_time':[ int(sys.argv[2]) ] }

rf_params = { 'ntrees': 100, 'max_depth': 30, 'min_samples_split': 2 }

years         =  [2011,2012,2013,2014,2015,2016] #k-fold cross validation for these years
#years         =  [ int(sys.argv[3]) ]
model         =  'nn'

plot          =  True

multiclass    =  True
output_stats  =  True
thin_data     =  True
thin_fraction =  0.5
smooth_probs  =  False
smooth_sigma  =  1
simple_features = True
dataset = 'NSC3km-12sec'
dataset = 'RT2020'
scaling_dataset = 'NSC3km-12sec'

trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models'
trained_models_dir = '/glade/work/sobash/NSC_objects'
trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models_paper'

sdate   = dt.datetime(2020,3,9,0,0,0)
edate   = dt.datetime(2020,3,9,0,0,0)
dateinc = dt.timedelta(days=1)

##################################

if multiclass: numclasses = 6
else: numclasses = 1
twin = "_%dhr"%nn_params['report_window_time'][0]

# complex features
basic_features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','PREC_ACC_NC','CAPESHEAR', 'STP', 'LR75']
large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']
neighbor_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC', 'PREC_ACC_NC']
neighbor_features = [ f+'-%s'%n for f in neighbor_fields for n in ['E1', 'S1', 'N1', 'W1', 'TP1', 'TM1', 'TM2', 'TP2', 'SE1', 'NE1', 'NW1', 'NE1'] ]
features = basic_features + large_scale_features + neighbor_features

# simple features
if simple_features:
    simple_max_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC']
    simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']
    simple_max_features = [ f+'-N%dT%d'%(x,t) for f in simple_max_fields for x in [3,5] for t in [1,3,5] ]
    simple_mean_features = [ f+'-N%dT%d'%(x,t) for f in simple_mean_fields for x in [3,5] for t in [1,3,5] ]
    features = basic_features + large_scale_features + simple_max_features + simple_mean_features
    print(features)

log('Number of features %d'%len(features))
log(nn_params)
log(rf_params)

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'
df, numfcsts = read_csv_files()
            
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask = mask.reshape((65,93))

if plot:
    # what forecast points to use
    forecast_mask = ( (df['fhr'] >= 13) )
    these_points = df[forecast_mask]
    year = sdate.year
    if year > 2016: year = 2016 #use NN without 2016 for any date past 2016

    classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
    plot_type = 0

    if model == 'nn':

        scaling_values = pickle.load(open('scaling_values_all_%s.pk'%scaling_dataset, 'rb'))
        norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), scaling_values=scaling_values)
        
        this_in_data = norm_in_data[forecast_mask,:]
        
        dense_model = None
        model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f.h5'%(trained_models_dir,year,nn_params['report_window_space'][0],twin,\
                                                                      nn_params['num_neurons'][0],nn_params['dropout'])
        dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
        
        predictions = dense_model.predict(this_in_data)

        these_points['predict_proba'] = predictions[:,plot_type]    

    if model == 'rf':
         
        this_in_data = these_points[features].values

        model_fname = '%s/rf_gridded_%s_%dkm%s_n%d_d%d_m%d.pk'%(trained_models_dir,year,nn_params['report_window_space'][0],twin,rf_params['ntrees'],rf_params['max_depth'],rf_params['min_samples_split'])
        rf = pickle.load(open(model_fname, 'rb'))

        predictions = rf.predict_proba(this_in_data)
        predictions = np.array(predictions)[:,:,1].T #needs to be in shape (examples,classes)

        these_points['predict_proba'] = predictions[:,plot_type]

    plot_forecast(these_points)
    sys.exit()

    smooth_sigma = [1.75]
    predictions_all = []
    for i in range(12,37):
        print(i)
        forecast_mask = ( these_points['fhr'] == i)
        plot_forecast(these_points[forecast_mask], prefix='_fhr%02d'%i, fhr=i)

        # make gridded UH forecast
        make_uh_grid = True
        if make_uh_grid:
            tp = these_points[forecast_mask]
            uh120_all = tp['UP_HELI_MAX120-N1T5'].values
            predictions_gridded_uh = make_gridded_forecast((uh120_all>50).astype(np.int32), tp['Date'], tp['fhr'])
            
            # make smoothed gridded UH forecast
            predictions_gridded_uh_smoothed = smooth_gridded_forecast(predictions_gridded_uh)
            
            predictions_all.append(predictions_gridded_uh_smoothed)

            # extract only CONUS points and only other masked points
            uh120_smoothed = predictions_gridded_uh_smoothed[:,:,:,mask].reshape((len(smooth_sigma),-1))

            tp['predict_proba'] = uh120_smoothed[0,:]
            plot_forecast(tp, prefix='_fhr%02d'%i, fhr=i)

    # plot max UH probs
    predictions_all = np.amax(predictions_all, axis=0)
    predictions_all = predictions_all[:,:,:,mask].reshape((len(smooth_sigma),-1))
    tp['predict_proba'] = predictions_all[0,:]
    plot_forecast(tp, prefix='_uhall')

log('Finished')
