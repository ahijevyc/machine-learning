#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm

import numpy as np
import datetime as dt
import os, pickle
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
import pandas as pd
from mpl_toolkits.basemap import *
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from keras.models import Model, model_from_json, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf

def computeSTP(row):
    lclterm = ((2000.0-row['MLLCL-potential_mean'])/1000.0)
    lclterm = np.where(row['MLLCL-potential_mean']<1000, 1.0, lclterm)
    lclterm = np.where(row['MLLCL-potential_mean']>2000, 0.0, lclterm)

    shrterm = (row['shr06']/20.0)
    shrterm = np.where(row['shr06'] > 30, 1.5, shrterm)
    shrterm = np.where(row['shr06'] < 12.5, 0.0, shrterm)

    stp = (row['SBCAPE-potential_mean']/1500.0) * lclterm * (row['SRH01-potential_mean']/150.0) * shrterm
    return stp

def read_csv_files():
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_NCARSTORM_d01_%s-0000.csv.gz'%(yyyymmdd)

        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print('Reading %s files'%(len(all_files)))

    df = pd.concat((pd.read_csv(f, compression='gzip') for f in all_files))

    #if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)   

    #if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    #if model == 'NCAR':
    df['datetime']  = pd.to_datetime(df['Date'])
    #df['Run_Date']  = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    df['year']     = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['hour']     = df['datetime'].dt.hour
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
 
def normalize_multivariate_data(data, scaling_values=None):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    print(data.shape, data.dtype)
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
        for i in range(data.shape[-1]): scaling_values.loc[i, ["mean", "std"]] = [data[:, i].mean(), data[:, i].std()]

    for i in range(data.shape[-1]):
        normed_data[:, i] = (data[:, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]

    return normed_data, scaling_values

def plot_forecast(predictions, prefix=""):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)
    cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    print(predictions)

    #awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)

    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/ch_pk_files/rt2015_ch_CONUS.pk', 'r'))
    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/dav_pk_files/rt2015_ch_CONUS.pk', 'rb'))
    fig, axes, m = pickle.load(open('data/rt2015_ch_CONUS.pk', 'rb')) 

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
        if thisvalue >= 0.05:
            a = plt.text(x[i], y[i], int(round(thisvalue*100)), fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')
    #a = m.scatter(x, y, s=50, c=predictions['predict_proba'].values, lw=0.5, edgecolors='k', cmap=cmap, norm=norm)

    # ADD COLORBAR
    #cax = fig.add_axes([0.02,0.1,0.02,0.3])
    #cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    #cb.outline.set_linewidth(0.5)
    #cb.ax.tick_params(labelsize=10)

    plt.savefig('forecast%s.png'%prefix)

sdate = dt.datetime(2012,6,29,0,0,0)
edate = dt.datetime(2012,6,29,0,0,0)
dateinc = dt.timedelta(days=1)
df, numfcsts = read_csv_files()

print('Training random forest classifier')

features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75']
large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']
neighbor_features = [ f+'-%s1'%n for f in large_scale_features for n in ['E','S','N','W'] ]
neighbor_time_features = [ f+'-%s'%n for f in ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01'] for n in ['TP1', 'TM1'] ]
features = features + large_scale_features + neighbor_features + neighbor_time_features

# normalize data we want to use
scaling_values = pickle.load(open('scaling_values.pk', 'rb'))

norm_in_data, scaling_values = normalize_multivariate_data(df[features].values, scaling_values=scaling_values)

# load combined architecture and weights
dense_model = load_model('neural_network.h5', custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })

predictions_proba = dense_model.predict(norm_in_data)
print(predictions_proba.max())
print(predictions_proba) 

#labels: all, wind, hailone, torn
df['predict_proba'] = predictions_proba[:,1]
forecast_mask = (df['fhr'] > 12)
plot_forecast(df[forecast_mask])
