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
    print df['datetime']
 
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
sdate = dt.datetime(2010,10,1,0,0,0)
edate = dt.datetime(2017,12,31,0,0,0)
dateinc = dt.timedelta(days=1)
df, numfcsts = read_csv_files()
test_year = 2015

print 'Training random forest classifier'

usecols = 'Step_ID,Track_ID,Ensemble_Name,Ensemble_Member,Run_Date,Valid_Date,Forecast_Hour,Valid_Hour_UTC,Duration,Centroid_Lon,Centroid_Lat,Centroid_X,Centroid_Y,Storm_Motion_U,Storm_Motion_V,UP_HELI_MAX_mean,UP_HELI_MAX_max,UP_HELI_MAX_min,GRPL_MAX_mean,GRPL_MAX_max,GRPL_MAX_min,WSPD10MAX_mean,WSPD10MAX_max,WSPD10MAX_min,W_UP_MAX_mean,W_UP_MAX_max,W_UP_MAX_min,W_DN_MAX_mean,W_DN_MAX_max,W_DN_MAX_min,RVORT1_MAX_mean,RVORT1_MAX_max,RVORT1_MAX_min,RVORT5_MAX_mean,RVORT5_MAX_max,RVORT5_MAX_min,UP_HELI_MAX03_mean,UP_HELI_MAX03_max,UP_HELI_MAX03_min,UP_HELI_MAX01_mean,UP_HELI_MAX01_max,UP_HELI_MAX01_min,UP_HELI_MIN_mean,UP_HELI_MIN_max,UP_HELI_MIN_min,REFL_COM_mean,REFL_COM_max,REFL_COM_min,REFL_1KM_AGL_mean,REFL_1KM_AGL_max,REFL_1KM_AGL_min,REFD_MAX_mean,REFD_MAX_max,REFD_MAX_min,PSFC_mean,PSFC_max,PSFC_min,T2_mean,T2_max,T2_min,Q2_mean,Q2_max,Q2_min,TD2_mean,TD2_max,TD2_min,U10_mean,U10_max,U10_min,V10_mean,V10_max,V10_min,SBLCL-potential_mean,SBLCL-potential_max,SBLCL-potential_min,MLLCL-potential_mean,MLLCL-potential_max,MLLCL-potential_min,SBCAPE-potential_mean,SBCAPE-potential_max,SBCAPE-potential_min,MLCAPE-potential_mean,MLCAPE-potential_max,MLCAPE-potential_min,MUCAPE-potential_mean,MUCAPE-potential_max,MUCAPE-potential_min,SBCINH-potential_mean,SBCINH-potential_max,SBCINH-potential_min,MLCINH-potential_mean,MLCINH-potential_max,MLCINH-potential_min,USHR1-potential_mean,USHR1-potential_max,USHR1-potential_min,VSHR1-potential_mean,VSHR1-potential_max,VSHR1-potential_min,USHR6-potential_mean,USHR6-potential_max,USHR6-potential_min,VSHR6-potential_mean,VSHR6-potential_max,VSHR6-potential_min,U_BUNK-potential_mean,U_BUNK-potential_max,U_BUNK-potential_min,V_BUNK-potential_mean,V_BUNK-potential_max,V_BUNK-potential_min,SRH03-potential_mean,SRH03-potential_max,SRH03-potential_min,SRH01-potential_mean,SRH01-potential_max,SRH01-potential_min,PSFC-potential_mean,PSFC-potential_max,PSFC-potential_min,T2-potential_mean,T2-potential_max,T2-potential_min,Q2-potential_mean,Q2-potential_max,Q2-potential_min,TD2-potential_mean,TD2-potential_max,TD2-potential_min,U10-potential_mean,U10-potential_max,U10-potential_min,V10-potential_mean,V10-potential_max,V10-potential_min,area,eccentricity,major_axis_length,minor_axis_length,orientation'

if model == 'NSC':
  features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat','Centroid_Lon','shr06','shr01',\
              'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
               'UP_HELI_MAX_mean', \
               'UP_HELI_MIN_mean', \
               'UP_HELI_MAX01_mean',\
               'UP_HELI_MAX03_mean',\
               'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean',\
               'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
               'PSFC-potential_mean', 'orientation', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation']

  # datetime features only
  #features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon']

  # storm features
  #features = ['area', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation']

  # environmental features only
  #features = ['shr06','shr01',\
  #            'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
  #            'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
  #             'PSFC-potential_mean']

  # surrogate fields only
  #features = ['UP_HELI_MAX_mean', 'UP_HELI_MIN_mean', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean',\
  #             'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean']

  # datetime+env features only
  #features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon',\
  #            'shr06','shr01',\
  #            'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
  #            'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
  #             'PSFC-potential_mean']

  # datetime + surrogate only
  #features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon',\
  #            'UP_HELI_MAX_mean', 'UP_HELI_MIN_mean', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean',\
  #            'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean']

  # surrogate+env features only
  #features = ['shr06','shr01',\
  #            'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
  #            'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
  #             'PSFC-potential_mean', 'UP_HELI_MAX_mean', 'UP_HELI_MIN_mean', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean',\
  #            'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean']
  
# use these features when training NSC to be used on NCAR ensemble forecasts
  #features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01','MUCAPE-potential_mean','SBCAPE-potential_mean','UP_HELI_MIN_min','UP_HELI_MAX_mean','W_UP_MAX_max','WSPD10MAX_mean',\
  #          'SBCINH-potential_mean','SRH03-potential_mean','SBLCL-potential_mean',\
  #          'UP_HELI_MAX_min', 'W_UP_MAX_min', 'W_DN_MAX_min', 'WSPD10MAX_min','orientation']

if model == 'NCAR':
  features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01','MUCAPE-potential_mean','CAPE_SFC-potential_mean','UP_HELI_MIN_min','UP_HELI_MAX_mean','W_UP_MAX_max','WSPD10MAX_mean',\
            'CIN_SFC-potential_mean','SRH3-potential_mean', 'LCL_HEIGHT-potential_mean',\
            'UP_HELI_MAX_min', 'W_UP_MAX_min', 'W_DN_MAX_min', 'WSPD10MAX_min','orientation']

#for d in [40,80,120,160,200,240]:
for d in [120]:
    labels   = ((df['hail_report_closest_distance'] < d*1000.0) & (df['hail_report_closest_distance'] > 0)) |  \
               ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0)) | \
               ((df['torn_report_closest_distance'] < d*1000.0) & (df['torn_report_closest_distance'] > 0))
    #labels   = ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0))

    # pick out training and testing samples from given years
    train_mask = (df['year'] != test_year)
    test_mask = (df['year'] == test_year)
    #train_mask = (df['year'] == 2010) & (df['year'] <= 2014)
    #test_mask = (df['month'] == 4) | (df['month'] == 5) | (df['month'] == 6) | (df['month'] == 7)
    #train_mask = (df['month'] == 1) | (df['month'] == 2) | (df['month'] == 3) | (df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12)

    # extract training and testing features and labels
    train_features, test_features = df[train_mask], df[test_mask]
    train_labels, test_labels = labels[train_mask], labels[test_mask]
    print 'train_features shape', train_features.shape
    print 'test_features shape', test_features.shape

    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=10)

    # set up random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=70, min_samples_split=2, oob_score=True, random_state=10, n_jobs=6)

    rf.fit(train_features[features], train_labels)

    #pickle.dump(rf, open('rf_severe_%dkm_%s_test%d_noUH01.pk'%(d,model,test_year), 'wb'))

    ### feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(features)):
        print("feature %s (%f)" % (features[f], importances[indices[f]]))

    #sanity_check = {'Valid_Hour_UTC': range(0,24), 'dayofyear': range(120,144), 'Centroid_Lat':np.arange(25,49), 'Centroid_Lon':[-90]*24 }
    #test_features = pd.DataFrame(data=sanity_check)

    ### predictions using test dataset
    print 'Predicting'
    predictions = rf.predict(test_features[features])
    predictions_proba = rf.predict_proba(test_features[features])

    # skill of RF
    print_scores(test_labels, predictions, probs=predictions_proba)
    
    # print histogram and scores for test
    print np.histogram(predictions_proba[:,1])
    print calibration_curve(test_labels, predictions_proba[:,1], n_bins=10)
    print metrics.roc_auc_score(test_labels, predictions_proba[:,1])
    
    # skill of UH25 > 50
    uh_max_values = (test_features['UP_HELI_MAX_max'] > 50).values
    print_scores(test_labels, uh_max_values)

    for i in range(1,13):
        mask = (test_features['month'] == i)
        mask = predictions & mask
        print i, test_features[mask]['UP_HELI_MAX_max'].mean()


    forecast_mask = (test_features['Run_Date'] == '2012-04-14 00:00:00')
    plot_forecast(test_features[forecast_mask], predictions_proba[forecast_mask])    
