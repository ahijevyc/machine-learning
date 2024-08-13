#!/usr/bin/env python

import numpy as np
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
import os
import cPickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mpl_toolkits.basemap import *

def computeshr01(row):
    if model == 'NSC': return np.sqrt(row['USHR1-potential_mean']**2 + row['VSHR1-potential_mean']**2)
    if model == 'NCAR': return np.sqrt(row['UBSHR1-potential_mean']**2 + row['VBSHR1-potential_mean']**2)

def computeshr06(row):
    if model == 'NSC': return np.sqrt(row['USHR6-potential_mean']**2 + row['VSHR6-potential_mean']**2)
    if model == 'NCAR': return np.sqrt(row['UBSHR6-potential_mean']**2 + row['VBSHR6-potential_mean']**2)

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
        if model == 'NSC': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv_preprocessed/track_step_NCARSTORM_d01_%s-0000_13_time2_filtered.csv'%(yyyymmdd)
        if model == 'NCAR': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncar_2016_csv_preprocessed/track_step_ncar_3km_%s_time2.csv'%(yyyymmdd)

        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print 'Reading %s files'%(len(all_files))

    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    df['shr01'] = df.apply(computeshr01, axis=1)
    df['shr06'] = df.apply(computeshr06, axis=1)

    if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)   

    if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    if model == 'NCAR':
        df['datetime']  = pd.to_datetime(df['Date'])
        df['Run_Date']  = pd.to_datetime(df['Date']) - pd.to_timedelta(df['Forecast_Hour'])
    df['year']     = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    if model == 'NCAR': df = df[df['Forecast_Hour']>12]
    #print df['datetime']
 
    return df, len(all_files)

def print_scores(labels, predictions, probs=np.array([])):
    cm = confusion_matrix(labels, predictions)
    #print cm
    hits = cm[1,1]
    false_alarms = cm[0,1]
    misses = cm[1,0]
    correct_neg = cm[0,0]
    hits_random = (hits + misses)*(hits + false_alarms) / float(hits + misses + false_alarms + correct_neg)

    ets = (hits-hits_random)/float(hits + false_alarms + misses - hits_random)
    bias = (hits+false_alarms)/float(hits+misses)
    pod = hits/float(hits+misses)
    far = false_alarms/float(hits+false_alarms)
    pofd = false_alarms/float(correct_neg + false_alarms)

    if probs.size > 0:
      bs_sums = ((probs[:,1]-labels)**2).sum()
      bs = (1/float(labels.size))*bs_sums
    else:
      bs = -999.0

    print 'BIAS=%0.3f, POD=%0.3f, FAR=%0.3f, POFD=%0.3f, ETS=%0.3f, BS=%0.3f'%(bias,pod,far,pofd,ets,bs) 

def plot_forecast(storms, predictions):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)
    cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    fig, axes, m  = pickle.load(open('/glade/u/home/sobash/RT2015_gpx/rt2015_ch_CONUS.pk', 'r'))

    lats, lons = storms['Centroid_Lat'].values, storms['Centroid_Lon'].values
    x, y = m(lons, lats)
    a = m.scatter(x, y, s=40, c=predictions[:,1], lw=0.5, edgecolors='k', cmap=cmap, norm=norm)
   
    # ADD COLORBAR
    cax = fig.add_axes([0.02,0.1,0.02,0.3])
    cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=10)
 
    plt.savefig('forecast.png')

model = 'NSC'
sdate = dt.datetime(2012,1,1,0,0,0)
edate = dt.datetime(2012,12,30,0,0,0)
dateinc = dt.timedelta(days=1)
df, numfcsts = read_csv_files()

rf = pickle.load(open('rf_severe_120km_NSC_test2018.pk', 'rb')) #all NSC storms (without UH01, that isnt stored in 2019 NCAR ensemble grib files)

features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01',\
              'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
               'UP_HELI_MAX_mean', \
               'UP_HELI_MIN_mean', \
               #'UP_HELI_MAX01_mean',\
               'UP_HELI_MAX03_mean',\
               'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean',\
               'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
               'PSFC-potential_mean', 'orientation']

d = 120.0
labels   = ((df['hail_report_closest_distance'] < d*1000.0) & (df['hail_report_closest_distance'] > 0)) |  \
               ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0)) | \
               ((df['torn_report_closest_distance'] < d*1000.0) & (df['torn_report_closest_distance'] > 0))

# compute random forest "proximity" for subset of storms
terminals = rf.apply(df[features])
nTrees = terminals.shape[1]
a = terminals[:,0]
proxMat = 1*np.equal.outer(a,a)

for i in range(1, nTrees):
    print i
    a = terminals[:,i]
    proxMat += 1*np.equal.outer(a,a)

proxMat = proxMat/float(nTrees) #normalization

outlyingness = (proxMat**2).sum(axis=0) - 1.0
outlyingness = 1/outlyingness

print proxMat.shape, proxMat
print outlyingness.max()

from sklearn.manifold import MDS
mds = MDS(n_components=2)
X_r = mds.fit_transform(proxMat)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
colors=((1,0,0),(0,1,0))
for label,color in zip( np.unique(labels),colors):
    position = (labels==label)
    ax.scatter(X_r[position,0],X_r[position,1],label="target= {0}".format(label),color=color)

ax.set_xlabel("X[0]")
ax.set_ylabel("X[1]")
ax.legend(loc="best")
ax.set_title("MDS")
plt.savefig('mds.png')

