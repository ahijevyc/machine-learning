#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
import numpy as np
import datetime as dt
import sys, os, pickle
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from mpl_toolkits.basemap import *
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf

from keras.utils import plot_model

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

### NEURAL NETWORK PARAMETERS ###

nn_params = { 'num_layers': 1, 'num_neurons': [ int(sys.argv[1]) ], 'dropout': float(sys.argv[2]), 'lr': 0.001, 'num_epochs': 10, \
              'report_window_space':[ 120 ], 'report_window_time':[ 2 ] }
year = 2011
d=120
twin = "_%dhr"%nn_params['report_window_time'][0]

##################################

print('Training random forest classifier')
features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','PREC_ACC_NC','CAPESHEAR', 'STP', 'LR75']
large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']
neighbor_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC', 'PREC_ACC_NC']
neighbor_features = [ f+'-%s'%n for f in neighbor_fields for n in ['E1', 'S1', 'N1', 'W1', 'TP1', 'TM1', 'TM2', 'TP2'] ]

features = features + large_scale_features + neighbor_features
print('Number of features', len(features))
print(features)

dense_model = None
model_fname = 'neural_network_%s_%dkm%s_nn%d_drop%.1f.h5'%(year,d,twin,nn_params['num_neurons'][0],nn_params['dropout'])
dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
