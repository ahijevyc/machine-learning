#!/usr/bin/env python

import numpy as np
import datetime as dt
import sys, os, pickle, time, math
import pandas as pd
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from sklearn import metrics

from scipy import spatial
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
    
from keras.models import Model, save_model, load_model
import tensorflow as tf
import keras.backend as K

from ml_functions import read_csv_files, normalize_multivariate_data, log, get_features

def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    bs = brier_score_keras(obs, preds)
    ratio = (bs / climo)
    return climo

def auc(obs, preds):
    auc = tf.metrics.auc(obs, preds)[1]
    #K.get_session().run(tf.local_variables_initializer())
    return auc

def compute_bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

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

def permutation_importance(idx):
    from keras.models import load_model
    import tensorflow as tf

    dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })

    # make copy of data to use for shuffling in place
    this_norm_in_data = np.copy(norm_in_data)
    #this_norm_in_data[:,idx] = 0.0

    # shuffle in place
    # idx < 0 means do not shuffle
    if idx >= 0 and idx not in importance_list:
        np.random.shuffle(this_norm_in_data[:,idx])

    # make predictions with shuffled data
    these_predictions = dense_model.predict(this_norm_in_data)

    # compute scores
    this_bss, this_auc = print_scores(these_predictions[:,hazard_idx], labels[:,hazard_idx])

    log((idx, fields_to_shuffle[idx], this_bss, this_auc))

    return {'idx': idx, 'bss': this_bss, 'auc': this_auc, 'data': this_norm_in_data[:,idx]}

### NEURAL NETWORK PARAMETERS ###

nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 10, \
              'report_window_space':[ int(sys.argv[1]) ], 'report_window_time':[ int(sys.argv[2]) ] }
rf_params = { 'ntrees': 100, 'max_depth': 20, 'min_samples_split': 2 }

years         =  [2011,2012,2013,2014,2015,2016] #k-fold cross validation for these years
#years         =  [ int(sys.argv[3]) ]
model         =  'nn'
multiclass    =  True
dataset       = 'NSC3km-12sec'
trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models_paper'

sdate   = dt.datetime(2011,4,1,0,0,0)
edate   = dt.datetime(2011,4,30,0,0,0)
dateinc = dt.timedelta(days=1)

##################################

twin = "_%dhr"%nn_params['report_window_time'][0]
features = get_features('basiclarge')

log('Number of features %d'%len(features))
log(nn_params)
log(rf_params)

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'
df, numfcsts = read_csv_files(sdate, edate, dataset)

log('Predicting Begin')
scaling_values = pickle.load(open('scaling_values_all_NSC3km-12sec.pk', 'rb'))
norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)

# neural network uses normalized data
model_fname = '%s/neural_network_2011_%dkm%s_nn%d_drop%.1f_basicplus.h5'%(trained_models_dir,nn_params['report_window_space'][0],twin,nn_params['num_neurons'][0],nn_params['dropout'])
    
labels = make_labels() 

# filter forecast data to only compute importance for selected points    
forecast_hour_mask = (df['fhr'] >= 12)
norm_in_data = norm_in_data[forecast_hour_mask,:]
labels = labels[forecast_hour_mask,:]
print(norm_in_data.shape, labels.shape)

### COMPUTE permutation importance
fields_to_shuffle = features

log('running upscaling in parallel')
hazard_idx = 0 
nprocs     = 6
nfeatures  = len(fields_to_shuffle)
chunksize  = int(math.ceil(nfeatures / float(nprocs)))

# implement multi-pass permutation importance
log('begin multi-pass permutation importance')
importance_list = []
for i in range(len(fields_to_shuffle)):
    bss_all, auc_all, idx_all = [], [], []
    shuffled_norm_in_data = norm_in_data.copy() 

    # shuffle each column in parallel and return the AUC/BSS
    pool      = multiprocessing.Pool(processes=nprocs)
    original_predictions = pool.map(permutation_importance, [-1], 1)
    data = pool.map(permutation_importance, range(nfeatures), chunksize)
    pool.close()
    
    # put values and shuffled columns into numpy arrays
    for d in data:
        bss_all.append(d['bss'])
        auc_all.append(d['auc'])
        idx_all.append(d['idx'])
        shuffled_norm_in_data[:,d['idx']] = d['data']
    bss_all, auc_all = np.array(bss_all), np.array(auc_all)

    # determine which feature is most important
    bss_loss_idx = np.argmax(original_predictions[0]['bss'] - bss_all)
    bss_loss_idx = idx_all[bss_loss_idx]

    # replace original data with shuffled column for most important feature
    norm_in_data[:,bss_loss_idx] = shuffled_norm_in_data[:,bss_loss_idx].copy()
    importance_list.append(bss_loss_idx)
    
    print(fields_to_shuffle[bss_loss_idx])

log('Finished') 


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
    
    #K.set_image_data_format('channels_first')

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
