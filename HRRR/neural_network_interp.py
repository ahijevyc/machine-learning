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
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

def log(msg):
    print( time.ctime(time.time()), msg ) 

def read_csv_files():
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        yyyymmddhh = tdate.strftime('%Y%m%d%H')
        if simple_features: csv_file = '/glade/work/sobash/NSC_objects/HRRR/grid_data/grid_data_HRRR_d01_%s-0000.csv'%(yyyymmddhh)
        else: csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_NSC3km-12sec_d01_%s-0000.csv'%(yyyymmdd)
        #csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_NCARSTORM_d01_%s-0000.csv.gz'%(yyyymmdd)
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

def compute_bss(obs, preds):
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

def make_labels():
    d = nn_params['report_window_space'][0]
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

def print_scores(fcst, obs):
    # BSS
    bss_val = compute_bss(obs, fcst)

    # ROC auc
    auc = metrics.roc_auc_score(obs, fcst)

    return (bss_val, auc)

### NEURAL NETWORK PARAMETERS ###

nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 10, \
              'report_window_space':[ int(sys.argv[1]) ], 'report_window_time':[ int(sys.argv[2]) ] }

rf_params = { 'ntrees': 100, 'max_depth': 20, 'min_samples_split': 2 }

years         =  [2011,2012,2013,2014,2015,2016] #k-fold cross validation for these years
#years         =  [ int(sys.argv[3]) ]
model         =  'nn'

train         =  False 
predict       =  True 
plot          =  False

multiclass    =  True
output_stats  =  True
thin_data     =  True
thin_fraction =  0.5
smooth_probs  =  False
smooth_sigma  =  1
simple_features = True 

trained_models_dir = '/glade/work/sobash/NSC_objects/HRRR/trained_models'
#trained_models_dir = '/glade/work/sobash/NSC_objects'
#trained_models_dir = '/glade/scratch/sobash/trained_models'

sdate   = dt.datetime(2020,4,1,0,0,0)
edate   = dt.datetime(2020,4,30,0,0,0)
dateinc = dt.timedelta(days=1)

##################################

if multiclass: numclasses = 6
else: numclasses = 1
twin = "_%dhr"%nn_params['report_window_time'][0]

### what features to use?
# complex features
explicit_features = [ 'UP_HELI_MAX', 'UP_HELI_MAX03', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC' ]
env_features      = [ 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'MLCINH', 'SBLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75' ]
static_features   = [ 'fhr', 'dayofyear', 'lat', 'lon' ]

large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']
basic_features    = static_features + explicit_features + env_features

# these are not being used anymore
#neighbor_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC', 'PREC_ACC_NC']
#neighbor_features = [ f+'-%s'%n for f in neighbor_fields for n in ['E1', 'S1', 'N1', 'W1', 'TP1', 'TM1', 'TM2', 'TP2', 'SE1', 'NE1', 'NW1', 'NE1'] ]
#features = basic_features + large_scale_features + neighbor_features

# simple features
if simple_features:
    simple_max_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC']
    simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SBLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']
    simple_max_features = [ f+'-N%dT%d'%(x,t) for f in simple_max_fields for x in [3,5] for t in [1,3,5] ]
    simple_mean_features = [ f+'-N%dT%d'%(x,t) for f in simple_mean_fields for x in [3,5] for t in [1,3,5] ]
    features = basic_features + large_scale_features + simple_max_features + simple_mean_features
    print(features)

# subsets for 2020 ML paper...
# UH only
#features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX-N3T1', 'UP_HELI_MAX-N3T3', 'UP_HELI_MAX-N3T5', 'UP_HELI_MAX-N5T1', 'UP_HELI_MAX-N5T3', 'UP_HELI_MAX-N5T5'] 
# basic features only
#features = basic_features
# basic + largescale only
#features = basic_features + large_scale_features
# environmental features only
#features = static_features + env_features + large_scale_features + simple_mean_features
# no upper air features (this also removed the explicit features accidentally...
#features = static_features + env_features + simple_mean_features + simple_max_features

features = static_features + explicit_features + env_features + large_scale_features

log('Number of features %d'%len(features))
log(nn_params)
log(rf_params)

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'
df, numfcsts = read_csv_files()

if predict:
    log('Predicting Begin')
    predictions_all, labels_all = np.empty((0,numclasses)), np.empty((0,numclasses))

    scaling_values = pickle.load(open('scaling_values_all_HRRR.pk', 'rb'))
    norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), scaling_values=scaling_values)

    # neural network uses normalized data
    dense_model = None
    model_fname = '%s/neural_network_2020_%dkm%s_nn%d_drop%.1f.h5'%(trained_models_dir,nn_params['report_window_space'][0],twin,nn_params['num_neurons'][0],nn_params['dropout'])
    dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
    
    labels = make_labels() 
    
    forecast_hour_mask = (df['fhr'] < 12)
    norm_in_data = norm_in_data[forecast_hour_mask,:]
    labels = labels[forecast_hour_mask,:]
    print(norm_in_data.shape, labels.shape)

    # create correlation plot and dendrogram
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #corr = spearmanr(norm_in_data).correlation
    #corr_linkage = hierarchy.ward(corr)
    #dendro = hierarchy.dendrogram(corr_linkage, labels=features, ax=ax1,
    #                          leaf_rotation=90)
    #dendro_idx = np.arange(0, len(dendro['ivl']))

    #ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    #ax2.set_xticks(dendro_idx)
    #ax2.set_yticks(dendro_idx)
    #ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    #ax2.set_yticklabels(dendro['ivl'])
    #fig.tight_layout()
    #plt.savefig('test.png')

    #from collections import defaultdict
    #cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    #cluster_id_to_feature_ids = defaultdict(list)
    #for idx, cluster_id in enumerate(cluster_ids):
    #    cluster_id_to_feature_ids[cluster_id].append(idx)
    #selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    #print(cluster_id_to_feature_ids, selected_features)
    
    K.set_image_data_format('channels_first')

    # try activation maximization
    #layer_index = utils.find_layer_idx(dense_model, 'dense_2')

    #for z in range(len(dense_model.layers)):
    #    print(dense_model.layers[z])

    #dense_model.layers[layer_index].activation = activations.linear
    #dense_model = utils.apply_modifications(dense_model, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc})  
  
    #print(dense_model.input)
    #dense_model.summary()
 
    #visualization = visualize_activation(dense_model, layer_index, filter_indices=0, lp_norm_weight=None, tv_weight=None)
    #sys.exit()

    # COMPUTE original predictions (before shuffling)
    print('computing original predictions before shuffling')
    hazard_idx = 0 
    original_predictions = dense_model.predict(norm_in_data)
    bss, auc = print_scores(original_predictions[:,hazard_idx], labels[:,hazard_idx])
    print(bss, auc)  

    ### COMPUTE permutation importance
    field_to_shuffle = features
    
    # implement multi-pass permutation importance (maybe this can be parallelized?)
    print('computing predictions with shuffled data')
    importance_list = []
    for i in range(len(field_to_shuffle)):
        bss_all, auc_all = [], []
        shuffled_norm_in_data = norm_in_data.copy()
        for f in field_to_shuffle:
            idx = features.index(f)
            if idx in importance_list: continue
            
            # shuffle only this column
            this_norm_in_data = np.copy(norm_in_data)
            #this_norm_in_data[:,idx] = 0.0
            np.random.shuffle(this_norm_in_data[:,idx])

            # preserve this shuffled column to use later
            shuffled_norm_in_data[:,idx] = this_norm_in_data[:,idx].copy()

            # make predictions with shuffled data
            these_predictions = dense_model.predict(this_norm_in_data)
 
            # compute scores
            bss2, auc2 = print_scores(these_predictions[:,hazard_idx], labels[:,hazard_idx])
            bss_all.append(bss2)
            auc_all.append(auc2)

            log('%s %d %.3f %.3f'%(f, idx, bss2-bss, auc2-auc))

        # determine which feature is most important
        bss_all, auc_all = np.array(bss_all), np.array(auc_all)
        bss_loss_idx = np.argmax(bss - bss_all)
        print(features[bss_loss_idx])

        # replace original data with shuffled column for most important feature
        norm_in_data[:,bss_loss_idx] = shuffled_norm_in_data[:,bss_loss_idx].copy()
        importance_list.append(bss_loss_idx)
        print(importance_list)

    #### COMPUTE partial dependence here
    # partial dependence for W and prob of any severe weather
    #idx = features.index('SHR06')
    #idx2 = features.index('MUCAPE')

    #for i in range(0,51,5):
    #    for j in range(0,5001,500):
    #        this_val = (i - scaling_values.loc['SHR06', "mean"]) / scaling_values.loc['SHR06', "std"]   
    #        this_val2 = (j - scaling_values.loc['MUCAPE', "mean"]) / scaling_values.loc['MUCAPE', "std"]   

    #        norm_in_data[:,idx] = this_val
    #        norm_in_data[:,idx2] = this_val2

    #        predictions = dense_model.predict(norm_in_data)
    #        print(i, predictions[:,0].mean())

log('Finished')
