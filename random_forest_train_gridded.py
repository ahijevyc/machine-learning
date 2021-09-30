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
import pickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mpl_toolkits.basemap import *

def computeSTP(row):
    lclterm = ((2000.0-row['MLLCL'])/1000.0)
    lclterm = np.where(row['MLLCL']<1000, 1.0, lclterm)
    lclterm = np.where(row['MLLCL']>2000, 0.0, lclterm)

    shrterm = (row['SHR06']/20.0)
    shrterm = np.where(row['SHR06'] > 30, 1.5, shrterm)
    shrterm = np.where(row['SHR06'] < 12.5, 0.0, shrterm)

    stp = (row['SBCAPE']/1500.0) * lclterm * (row['SRH01']/150.0) * shrterm
    return stp

def computeLR75(row):
    return (row['T700']-row['T500'])

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

    #print 'computing stp'
    #df['stp']   = df.apply(computeSTP, axis=1)   
    #df['lr']   = df.apply(computeLR75, axis=1)   
    #print 'done computing stp'

    #if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    #if model == 'NCAR':
    df['datetime']  = pd.to_datetime(df['Date'])
    #df['Run_Date']  = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    df['year']     = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['hour']     = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear

    #if model == 'NCAR': df = df[df['Forecast_Hour']>12]
    #print df['datetime']

    return df, len(all_files)

def print_scores(labels, predictions, probs=np.array([])):
    cm = confusion_matrix(labels, predictions)
    print(cm)
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
      bs = np.mean(((probs[:,1] - labels)**2))
      climo = np.mean((labels - np.mean(labels)) ** 2)
      bss = 1.0 - bs/climo
    else:
      bs = -999.0
      bss = -999.0

    print('BIAS=%0.3f, POD=%0.3f, FAR=%0.3f, POFD=%0.3f, ETS=%0.3f, BS=%0.3f, BSS=%0.3f'%(bias,pod,far,pofd,ets,bs,bss)) 

def plot_forecast(predictions, prefix=""):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)
    cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/ch_pk_files/rt2015_ch_CONUS.pk', 'r'))

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

def bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - K.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

model = 'NSC'
sdate = dt.datetime(2011,1,1,0,0,0)
edate = dt.datetime(2012,12,31,0,0,0)
dateinc = dt.timedelta(days=1)
df, numfcsts = read_csv_files()
test_year = 2012

print('Training random forest classifier')

#features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC',\
#             'CAPESHEAR', 'STP', 'LR75', 'U850','U700','U500','V850','V700','V500','T850','T700','T500','TD850','TD700','TD500']
features = ['fhr', 'dayofyear', 'lat', 'lon', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'MUCAPE', 'SHR06', 'MLCINH', 'MLLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75']
#large_scale_features = ['U850','U700','U500','V850','V700','V500','T850','T700','T500','TD850','TD700','TD500']
#neighbor_features = [ f+'-%s1'%n for f in large_scale_features for n in ['E','S','N','W'] ]

#features = features + large_scale_features + neighbor_features
print('Number of features', len(features))

#for c in df.columns: print c

#for d in [40,80,120,160,200,240]:
for d in [40,80,120,160,200,240]:
    #labels   = ((df['hail_report_closest_distance'] < d*1000.0) & (df['hail_report_closest_distance'] > 0)) |  \
    labels   = ((df['hailone_report_closest_distance'] < d*1000.0) & (df['hailone_report_closest_distance'] > 0)) |  \
               ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0)) | \
               ((df['torn_report_closest_distance'] < d*1000.0) & (df['torn_report_closest_distance'] > 0))
    #labels   = ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0))
    #labels = df['OSR']

    # pick out training and testing samples from given years
    train_mask = (df['year'] != test_year)
    test_mask = (df['year'] == test_year)
    #train_mask = (df['year'] == 2010) & (df['year'] <= 2014)
    #test_mask = (df['month'] == 4) | (df['month'] == 5) | (df['month'] == 6) | (df['month'] == 7)
    #train_mask = (df['month'] == 1) | (df['month'] == 2) | (df['month'] == 3) | (df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12)

    # extract training and testing features and labels
    train_features, test_features = df[train_mask], df[test_mask]
    train_labels, test_labels = labels[train_mask], labels[test_mask]
    print('train_features shape', train_features.shape)
    print('test_features shape', test_features.shape)

    # only train every 4 hours
    #fhrmask = (df['fhr'] == 14) | (df['fhr'] == 18) | (df['fhr'] == 22) | (df['fhr'] == 26) | (df['fhr'] == 32)
    #train_features, test_features = train_features[fhrmask], test_features[fhrmask]
    #train_labels, test_labels = train_labels[fhrmask], test_labels[fhrmask]

    # set up random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=70, min_samples_split=2, oob_score=True, random_state=10, n_jobs=10)

    rf.fit(train_features[features], train_labels)

    #pickle.dump(rf, open('rf_severe_gridded_%dkm_%s_test%d.pk'%(d,model,test_year), 'wb'))

    ### feature importances
    #importances = rf.feature_importances_
    #indices = np.argsort(importances)[::-1]
    #for f in range(len(features)):
    #    print("feature %s (%f)" % (features[f], importances[indices[f]]))

    #sanity_check = {'Valid_Hour_UTC': range(0,24), 'dayofyear': range(120,144), 'Centroid_Lat':np.arange(25,49), 'Centroid_Lon':[-90]*24 }
    #test_features = pd.DataFrame(data=sanity_check)

    ### predictions using test dataset
    print('Predicting')
    predictions = rf.predict(test_features[features])
    predictions_proba = rf.predict_proba(test_features[features])

    # skill of RF
    print_scores(test_labels, predictions, probs=predictions_proba)
    
    # print histogram and scores for test
    print(np.histogram(predictions_proba[:,1]))
    true_prob, fcst_prob = calibration_curve(test_labels, predictions_proba[:,1], n_bins=10)
    for i in range(true_prob.size): print(true_prob[i], fcst_prob[i])
    print(metrics.roc_auc_score(test_labels, predictions_proba[:,1]))
   
    plot_forecast = False
    if plot_forecast:
        test_features['predict_proba'] = predictions_proba[:,1]
        test_features = test_features.sort_values(by=['predict_proba'])
 
        # skill of UH25 > 50
        #uh_max_values = (test_features['UP_HELI_MAX'] > 50).values
        #print_scores(test_labels, uh_max_values)

        forecast_date = '2012-04-14 00:00:00'
        forecast_mask = (test_features['Date'] == forecast_date) & (test_features['fhr'] > 12)
        plot_forecast(test_features[forecast_mask]) 
    
        for i in range(13,37):
            forecast_mask = (test_features['Date'] == forecast_date) & (test_features['fhr'] == i)
            plot_forecast(test_features[forecast_mask], prefix=str(i))    

