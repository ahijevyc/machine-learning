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
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf
from netCDF4 import Dataset

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

def read_conv_files():
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d%H')
        csv_file = '/glade/work/sobash/NSC/gridded_windows_conv_%s.nc'%(yyyymmdd)
        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    
    print('Reading %s files'%(len(all_files)))

    in_data, label_data, years = [], [], []
    for f in all_files:
        fh = Dataset(f, 'r')
        #print(f)
        year = f.split('_')[3][0:4]
        total_windows = fh.dimensions['windows'].size * fh.dimensions['fhrs'].size
        years.append(np.ones((total_windows,))*int(year))
        in_data.append(np.stack([fh.variables[v][:].reshape(-1,5,5) for v in features], axis=-1))
        label_data.append(np.stack([fh.variables[v][:] for v in ['reportdisttorn', 'reportdisthailone', 'reportdistwind']], axis=-1))
        fh.close()

    all_in_data = np.vstack(in_data)
    label_data = np.array(label_data)
    del in_data[:]
    return years, all_in_data, label_data

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
    climo = np.mean((obs - K.mean(obs)) ** 2)
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
    print(data.shape, data.dtype)
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
    print (scaling_values)
    for i in range(data.shape[-1]):
        print (i)
        scaling_values.loc[i, ["mean", "std"]] = [data[:, :, :, i].mean(), data[:, :, :, i].std()]
        normed_data[:, :, :, i] = (data[:, :, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
    return normed_data, scaling_values

def plot_forecast(predictions, prefix=""):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)
    cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

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

model = 'NSC'
sdate = dt.datetime(2011,1,1,0,0,0)
edate = dt.datetime(2015,12,31,0,0,0)
dateinc = dt.timedelta(days=1)

print('Training random forest classifier')
features = ['fhr', 'doy', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC',\
             'CAPESHEAR', 'STP', 'LR75', 'U850','U700','U500','V850','V700','V500','T850','T700','T500','TD850','TD700','TD500'] 

years, in_data, label_data = read_conv_files()
label_data = label_data.reshape((-1,3))
years = np.array(years).flatten()
print(years.shape, in_data.shape, label_data.shape)

#for d in [40,80,120,160,200,240]:
for d in [120]:
    labels = (label_data[:,0] <= d*1000.0) | (label_data[:,1] <= d*1000.0) | (label_data[:,2] <= d*1000.0)

    # pick out training and testing samples from given years
    train_indices = np.where(years != 2012)[0]
    test_indices = np.where(years == 2012)[0]    
    print(train_indices, test_indices)

    # normalize data we want to use
    norm_in_data, scaling_values = normalize_multivariate_data(in_data)

    print ('done normalizing')

    # Input data in shape (y, x, variable)
    session = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True,
                                                   gpu_options=K.tf.GPUOptions(allow_growth=True),
                                                   log_device_placement=False))
    K.set_session(session)

    # Input data in shape (y, x, variable)
    l2_a= 0.01
    conv_net_in = Input(shape=norm_in_data.shape[1:])

    # First 2D convolution Layer
    conv_net = Conv2D(256, (3, 3), padding="valid", kernel_regularizer=l2(l2_a))(conv_net_in)
    conv_net = Activation("relu")(conv_net)
    conv_net = Dropout(0.1)(conv_net)
    conv_net = MaxPooling2D()(conv_net)
    conv_net = BatchNormalization()(conv_net)

    # Second set of convolution and pooling layers
    conv_net = Conv2D(512, (1, 1), padding="valid", kernel_regularizer=l2(l2_a))(conv_net)
    conv_net = Activation("relu")(conv_net)
    conv_net = Dropout(0.1)(conv_net)
    conv_net = BatchNormalization()(conv_net)
    #conv_net = MaxPooling2D()(conv_net)

    # Third set of convolution and pooling layers
    #conv_net = Conv2D(512, (1, 1), padding="valid", kernel_regularizer=l2(l2_a))(conv_net)
    #conv_net = Activation("relu")(conv_net)
    #conv_net = Dropout(0.1)(conv_net)
    #conv_net = BatchNormalization()(conv_net)
    #conv_net = MaxPooling2D()(conv_net)

    # Flatten the last convolutional layer into a long feature vector
    conv_net = Flatten()(conv_net)

    # Dense output layer, equivalent to a logistic regression on the last layer
    conv_net = Dense(1)(conv_net)
    conv_net = Activation("sigmoid")(conv_net)
    conv_model = Model(conv_net_in, conv_net)

    # Use the Adam optimizer with default parameters
    opt = Adam()
    #opt = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)
    conv_model.compile(opt, "mse", metrics=[brier_skill_score_keras, brier_score_keras, auc])

    conv_model.summary()

    # Train model
    num_epochs = 20
    conv_hist = conv_model.fit(norm_in_data[train_indices], labels[train_indices], 
                                 batch_size=1024, epochs=num_epochs, verbose=2,
                                 validation_data=(norm_in_data[test_indices], labels[test_indices]))

    for i in range(num_epochs):
        print(i, 'bss', 1 - (conv_hist.history["val_brier_score_keras"][i]/conv_hist.history["val_brier_skill_score_keras"][i]))
    
    # save neural network and scaling weights for normalization
    #dense_model.save('neural_network.h5')
    #pickle.dump(scaling_values, open('scaling_values.pk', 'wb'))

    print('predicting using neural network')
    predictions_proba = dense_model.predict(norm_in_data[test_indices])
    
    # print histogram and scores for test
    print(np.histogram(predictions_proba))
    true_prob, fcst_prob = calibration_curve(labels[test_indices], predictions_proba, n_bins=10)
    for i in range(true_prob.size): print(true_prob[i], fcst_prob[i])
    print(metrics.roc_auc_score(labels[test_indices], predictions_proba))
    print(bss(labels[test_indices], predictions_proba))

    # plot predictions for each grid point on map
    test_data = df.iloc[test_indices,:]
    test_data['predict_proba'] = predictions_proba
    #test_features = norm_in_data.sort_values(by=['predict_proba'])

    forecast_date = '2012-04-14 00:00:00'
    #forecast_date = '2013-05-20 00:00:00'
    #forecast_date = '2014-05-22 00:00:00'
    forecast_mask = (test_data['datetime'] == forecast_date) & (test_data['fhr'] > 12)
    plot_forecast(test_data[forecast_mask])

    #for i in range(13,37):
    #    forecast_mask = (test_features['Date'] == forecast_date) & (test_features['fhr'] == i)
    #    plot_forecast(test_features[forecast_mask], prefix=str(i))

