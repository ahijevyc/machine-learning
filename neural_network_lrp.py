#!/usr/bin/env python

import numpy as np
import datetime as dt
import sys, os, pickle, time
from keras.models import Model, save_model, load_model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf
import pandas as pd
import innvestigate
import innvestigate.utils as iutils

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
    K.get_session().run(tf.local_variables_initializer())
    return auc

def log(msg):
    print( time.ctime(time.time()), msg ) 

### NEURAL NETWORK PARAMETERS ###
nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 30, \
              'report_window_space':[ int(sys.argv[1]) ], 'report_window_time':[ int(sys.argv[2]) ] }

dataset = 'RT2020'
scaling_dataset = 'NSC3km-12sec'
scaling_file = '/glade/work/sobash/NSC_objects/scaling_values_all_%s.pk'%scaling_dataset

trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models_paper'

sdate   = dt.datetime(2020,5,1,0,0,0)
edate   = dt.datetime(2020,5,10,0,0,0)
dateinc = dt.timedelta(days=1)

features = get_features('basic')

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'

df, numfcsts = read_csv_files(sdate, edate, dataset)
print(numfcsts)

scaling_values = pickle.load(open(scaling_file, 'rb'))
norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)

dense_model = None
model_fname = '%s/neural_network_2016_120km_2hr_nn%d_drop%.1f_basic.h5'%(trained_models_dir,nn_params['num_neurons'][0],nn_params['dropout'])
dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })

print(norm_in_data.shape)

analyzer = innvestigate.create_analyzer('lrp.alpha_2_beta_1', dense_model, neuron_selection_mode='index')
a = analyzer.analyze(norm_in_data, 0)

a /= np.max(np.abs(a))

a = a.reshape((36,1298,-1))
a = np.mean(a[24,:,:], axis=0)
print(a.shape)

for i,f in enumerate(features):    
    print(f, a[i])

log('Finished')
