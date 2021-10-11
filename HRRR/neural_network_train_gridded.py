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
from keras import Sequential
import keras.backend as K
import tensorflow as tf
from scipy import spatial

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

def plot_forecast(predictions, prefix="", fhr=36):
    test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    cmap = ListedColormap(test)
    #cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    print(predictions)

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

    for i,p in enumerate(predictions['predict_proba'].values):
        thiskey = '%f%f'%(lats[i],lons[i])
        thisvalue = probmax[thiskey]

        color = cmap(norm([thisvalue])[0])
        probmax[thiskey] = -999
        if thisvalue >= 0.15:
            a = plt.text(x[i], y[i], int(round(thisvalue*100)), fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')
    #a = m.scatter(x, y, s=50, c=predictions['predict_proba'].values, lw=0.5, edgecolors='k', cmap=cmap, norm=norm)

    ax = plt.gca()
    cdate = sdate + dt.timedelta(hours=fhr)
    sdatestr = (cdate - dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    edatestr = (cdate + dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    plt.text(0,1.01,'Probability of tornado within 75-mi of a point valid %s - %s'%(sdatestr, edatestr), fontsize=14, transform=ax.transAxes)

    # ADD COLORBAR
    #cax = fig.add_axes([0.02,0.1,0.02,0.3])
    #cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    #cb.outline.set_linewidth(0.5)
    #cb.ax.tick_params(labelsize=10)

    plt.savefig('forecast%s.png'%prefix)

def train_random_forest():
    # set up random forest classifier
    rf = RandomForestClassifier(n_estimators=rf_params['ntrees'], max_depth=rf_params['max_depth'], min_samples_split=rf_params['min_samples_split'], \
                                min_samples_leaf=rf_params['min_samples_leaf'], oob_score=True, random_state=10, n_jobs=36)
    in_data = df[features].values

    # trained with unnormalized data
    rf.fit(in_data[train_indices], labels[train_indices])
    
    return rf

def init_neural_network():
    #K.tf doesnt work with newer keras?
    #session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                               gpu_options=tf.GPUOptions(allow_growth=True),
                                               log_device_placement=False))
    K.set_session(session)

def init_weights(model):
    log('Initializing weights from previously trained model')
    old_model = None
    if not os.path.exists(old_model_fname): sys.exit()
    old_model = load_model(old_model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })

    old_weights = old_model.get_weights()

    model.set_weights(old_weights)

    return model

def train_neural_network_transfer():
    previous_model = None
    if not os.path.exists(old_model_fname): sys.exit()
    previous_model = load_model(old_model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
    
    # build new model to use for transfer learning
    inputs = Input(shape=(148,), name='input')
    layer  = Dense(1024, kernel_regularizer=l2(),name="nscdense")(inputs)
    layer  = Activation("relu", name="nscact")(layer)
    layer  = Dropout(0.2, name="nscdrop")(layer)
    layer  = BatchNormalization(name="nscbn")(layer)

    layer  = Dense(1024, kernel_regularizer=l2(), name="hrrrdense")(layer)
    layer  = Activation("relu", name="hrrract")(layer)
    layer  = Dropout(0.2, name="hrrrdrop")(layer)
    layer  = BatchNormalization(name="hrrrbn")(layer)
    outputs  = Dense(6, activation="sigmoid", name="hrrrout")(layer)

    dense_model = Model(inputs, outputs)

    # use weights in four layers from previous model
    for n in [1,2,3,4]:
        weights = previous_model.layers[n].get_weights()
        print(np.array(weights).shape)
        dense_model.layers[n].set_weights( weights )
        dense_model.layers[n].trainable = False

    dense_model.summary()

    # Optimizer object
    opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)

    # Compile model with optimizer and loss function
    if multiclass: dense_model.compile(opt_dense, loss="binary_crossentropy", metrics=[brier_score_keras, brier_skill_score_keras, auc])
    else: dense_model.compile(opt_dense, loss="mse", metrics=[brier_score_keras, brier_skill_score_keras, auc])

    dense_hist = dense_model.fit(norm_in_data[train_indices], labels[train_indices],
                                 batch_size=1024, epochs=nn_params['num_epochs'], verbose=1)

    return (dense_hist, dense_model)

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

    # initialize weights here?
    if reuse_weights: dense_model = init_weights(dense_model)

    # Train model
    dense_hist = dense_model.fit(norm_in_data[train_indices], labels[train_indices],
                             batch_size=1024, epochs=nn_params['num_epochs'], verbose=1)
                             #validation_data=(norm_in_data[test_indices], labels[test_indices]))

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

nn_params = { 'num_layers': 1, 'num_neurons': [ 1024 ], 'dropout': 0.1, 'lr': 0.001, 'num_epochs': 30, \
              'report_window_space':[ int(sys.argv[1]) ], 'report_window_time':[ int(sys.argv[2]) ] }

rf_params = { 'ntrees': 100, 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 10 }

years         =  [2011,2012,2013,2014,2015,2016] #k-fold cross validation for these years
#years  = [2017]
years         =  [ int(sys.argv[3]) ]
model         =  'nn'

train         =  False
predict       =  True
plot          =  False

multiclass    =  True
output_stats  =  False 
thin_data     =  True
thin_fraction =  0.9999
smooth_probs  =  False
smooth_sigma  =  1
simple_features = True

dataset = 'NSC1km'
dataset = 'NSC3km-12sec'
dataset = 'HRRR'
#dataset = 'HRRRX' #train with HRRRX data to use with HRRRv4

scaling_dataset = 'NSC3km-12sec'
scaling_dataset = 'HRRR'
#scaling_dataset = 'HRRRX' # HRRRX training data scaling

subset = 'all'
expname = 'hrrrv4-epoch30'
#expname = 'hrrrv4-epoch30-model5'

expname = 'forhrrr'
use_nschrrr_features = True # use features that are in BOTH nsc and hrrr

trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models'
#trained_models_dir = '/glade/work/sobash/NSC_objects/trained_models_paper'
#trained_models_dir = '/glade/work/sobash/NSC_objects/HRRR/trained_models'

reuse_weights = False
old_model_fname = '/glade/work/sobash/NSC_objects/trained_models/neural_network_2016_40km_2hr_nn1024_drop0.1_forhrrr.h5'

if train:
    sdate   = dt.datetime(2019,10,1,0,0,0)
    edate   = dt.datetime(2020,12,2,0,0,0)
    #edate   = dt.datetime(2020,7,15,0,0,0)
elif predict:
    sdate   = dt.datetime(2020,3,1,0,0,0) #for verification
    edate   = dt.datetime(2020,6,30,0,0,0) #for verification
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

#for column in df:
#    if df[column].dtype in ['float32', 'float64', 'int32', 'int64']: print("%s,%f,%f,%s"%(column, df[column].min(), df[column].max(), df[column].dtype))

if train:
    log('Training Begin')
    init_neural_network()

    # normalize data if training a neural network, output scaling values
    if model == 'nn':
        if os.path.exists('scaling_values_all_%s.pk'%scaling_dataset):
            scaling_values = pickle.load(open('scaling_values_all_%s.pk'%scaling_dataset, 'rb'))
            norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)
        else:
            norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=None)
            pickle.dump(scaling_values, open('scaling_values_all_%s.pk'%(scaling_dataset), 'wb'))

    for d in nn_params['report_window_space']:
        labels = make_labels()
            
        # train on random subset of examples (to speed up processing)
        if thin_data and model == 'rf':
            df, df_test, labels, labels_test = train_test_split(df, labels, train_size=thin_fraction, random_state=10)
        elif thin_data and model == 'nn':
            df, df_test, norm_in_data, norm_in_data_test, labels, labels_test = train_test_split(df, norm_in_data, labels, train_size=thin_fraction, random_state=10)

        for year in years:
            # train on examples not occurring in this year
            train_indices = np.where(df['year'] != year)[0]
            test_indices  = np.where(df['year'] == year)[0]
            if train_indices.size < 1: continue
            #if train_indices.size < 1 or test_indices.size < 1: continue #test_indices only used for validation when training NN

            log('training with %d examples -- leaving out %d'%(len(train_indices), year))

            # train model!
            if model == 'nn':
                dense_hist, dense_model = train_neural_network()
                #dense_hist, dense_model = train_neural_network_transfer()

                log('Writing model') 
                model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f_%s.h5'%(trained_models_dir,year,d,twin,nn_params['num_neurons'][0],nn_params['dropout'],expname)
                dense_model.save(model_fname)

            if model == 'rf':
                rf = train_random_forest()
               
                log('Writing model')
                model_fname = '%s/rf_gridded_%s_%dkm%s_n%d_d%d_m%d_l%d.pk'%(trained_models_dir,year,d,twin,rf_params['ntrees'],rf_params['max_depth'],\
                                                                            rf_params['min_samples_split'],rf_params['min_samples_leaf']) 
                pickle.dump(rf, open(model_fname, 'wb'))


if predict:
    log('Predicting Begin')
    #predictions_all, labels_all, fhr_all, cape_all, shear_all, date_all = np.empty((0,numclasses)), np.empty((0,numclasses)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))
    #uh_all, uh80_all, uh120_all = np.empty((0,)), np.empty((0,)), np.empty((0,))

    # if predicting, use stored scaling values for NN
    if model == 'nn':
        #norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), scaling_values=None)
        #pickle.dump(scaling_values, open('scaling_values_all_%s.pk'%(scaling_dataset), 'wb'))
        scaling_values = pickle.load(open('scaling_values_all_%s.pk'%scaling_dataset, 'rb'))
        norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)

    for d in nn_params['report_window_space']:
        labels = make_labels()

        for year in years:
            # which forecasts to verify?            
            forecast_hours_to_verify = range(1,49)
            forecast_mask = ( (df['fhr'].isin(forecast_hours_to_verify)) & (df['year'] == year) )
            if forecast_mask.values.sum() < 1: continue
           
            if year == 2020: model_year = 2016 #use 2016 model that left out 2016 for 2020 predictions
            #else: model_year = year
            #model_year = 2021
 
            log('Making predictions for %d forecasts in %d'%(forecast_mask.values.sum(), year))
            if model == 'nn':
                # neural network uses normalized data
                this_in_data = norm_in_data[forecast_mask,:] 

                dense_model = None
                model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f_%s.h5'%\
                               (trained_models_dir,model_year,d,twin,nn_params['num_neurons'][0],nn_params['dropout'],expname)
                #model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f_all.h5'%\
                #               (trained_models_dir,model_year,d,twin,nn_params['num_neurons'][0],nn_params['dropout'])
                print('Using', model_fname)
                if not os.path.exists(model_fname): print('%s not found'%model_fname); continue
                dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
           
                predictions = dense_model.predict(this_in_data)

            if model == 'rf':
                # random forest uses unnormalized data
                this_in_data = df[features].values
                this_in_data = this_in_data[forecast_mask,:]

                model_fname = '%s/rf_gridded_%s_%dkm%s_n%d_d%d_m%d_l%d.pk'%(trained_models_dir,model_year,d,twin,rf_params['ntrees'],rf_params['max_depth'],\
                                                                            rf_params['min_samples_split'],rf_params['min_samples_leaf']) 
                print(model_fname)
                if not os.path.exists(model_fname): continue
                rf = pickle.load(open(model_fname, 'rb'))

                predictions = rf.predict_proba(this_in_data)
                if multiclass: predictions = np.array(predictions)[:,:,1].T #needs to be in shape (examples,classes)
                else: predictions = np.array([predictions])[:,:,1].T #needs to be in shape (examples,classes)

            #print('putting predictions back on grid and smoothing')
            #predictions = make_grid(df[forecast_mask], predictions, labels[forecast_mask,:]
 
            log('Appending predictions')
            #predictions_all = np.append(predictions_all, predictions, axis=0)
            #labels_all      = np.append(labels_all, labels[forecast_mask,:], axis=0)
            #fhr_all         = np.append(fhr_all, df[forecast_mask]['fhr'].values, axis=0)
            #cape_all        = np.append(cape_all, df[forecast_mask]['MUCAPE'].values, axis=0)
            #shear_all       = np.append(shear_all, df[forecast_mask]['SHR06'].values, axis=0)
            #uh_all          = np.append(uh_all, df[forecast_mask]['UP_HELI_MAX'].values, axis=0)
            #date_all        = np.append(date_all, df[forecast_mask]['Date'].values, axis=0)
            #if d == 40 and twin == '_2hr': uh120_all        = np.append(uh120_all, df[forecast_mask]['UP_HELI_MAX-N1T5'].values, axis=0)
            #if d == 80 and twin == '_2hr': uh120_all        = np.append(uh120_all, df[forecast_mask]['UP_HELI_MAX80-N1T5'].values, axis=0)
            #if d == 120 and twin == '_2hr': uh120_all        = np.append(uh120_all, df[forecast_mask]['UP_HELI_MAX120-N1T5'].values, axis=0)
            
            # create pandas dataframe here for output....
            df_out = df[forecast_mask][['fhr', 'MUCAPE', 'SHR06', 'UP_HELI_MAX', 'Date']]
            if d == 40 and twin == '_2hr': df_out["maxuhwin"] = df[forecast_mask]['UP_HELI_MAX-N1T5']
            elif d == 80 and twin == '_2hr': df_out["maxuhwin"] = df[forecast_mask]['UP_HELI_MAX80-N1T5']
            elif d == 120 and twin == '_2hr': df_out["maxuhwin"] = df[forecast_mask]['UP_HELI_MAX120-N1T5']
            for i in range(numclasses):
                df_out["predict%d"%i] = predictions[:,i]
                df_out["label%d"%i] = labels[forecast_mask,i]
            df_out.to_pickle('predictions_%s_%dkm%s_%s'%(model,d,twin,expname))

        #log('Verifying %d forecast points'%predictions_all.shape[0])
        #classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
        #for i in range(numclasses):
        #    print_scores(predictions_all[:,i], labels_all[:,i], classes[i])
        
        #pickle.dump([predictions_all, labels_all.astype(np.bool), fhr_all.astype(np.int8), cape_all.astype(np.int16), shear_all.astype(np.int16), \
        #             uh_all.astype(np.float32), uh120_all.astype(np.float32), date_all], \
        #             open('predictions_%s_%dkm%s_%s'%(model,nn_params['report_window_space'][0],twin,expname), 'wb'))

if plot:
    # what forecast points to use
    forecast_mask = ( (df['fhr'] > 12) )
    these_points = df[forecast_mask]
    year = sdate.year

    classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
    plot_type = 0

    if model == 'nn':

        scaling_values = pickle.load(open('scaling_values_all_%s.pk'%scaling_dataset, 'rb'))
        norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=scaling_values)

        for f in features: print(f)
        
        this_in_data = norm_in_data[forecast_mask,:]
        
        dense_model = None
        model_fname = '%s/neural_network_%s_%dkm%s_nn%d_drop%.1f.h5'%(trained_models_dir,year,nn_params['report_window_space'][0],twin,nn_params['num_neurons'][0],nn_params['dropout'])
        dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
        
        predictions = dense_model.predict(this_in_data)

        these_points['predict_proba'] = predictions[:,plot_type]    

    if model == 'rf':
         
        this_in_data = these_points[features].values
        model_fname = '%s/rf_gridded_%s_%dkm%s_n%d_d%d_m%d_l%d.pk'%(trained_models_dir,year,nn_params['report_window_space'][0],twin,rf_params['ntrees'],rf_params['max_depth'],\
                                                                            rf_params['min_samples_split'],rf_params['min_samples_leaf'])
        rf = pickle.load(open(model_fname, 'rb'))

        predictions = rf.predict_proba(this_in_data)
        predictions = np.array(predictions)[:,:,1].T #needs to be in shape (examples,classes)

        these_points['predict_proba'] = predictions[:,plot_type]

    plot_forecast(these_points)

    #for i in range(13,37):
    #    print(i)
    #    forecast_mask = ( these_points['fhr'] == i)
    #    plot_forecast(these_points[forecast_mask], prefix='_fhr%d'%i, fhr=i)

log('Finished')
