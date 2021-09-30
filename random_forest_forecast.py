#!/usr/bin/env python

import numpy as np
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
from scipy import spatial
from netCDF4 import Dataset, MFDataset 
import os, time
import cPickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mpl_toolkits.basemap import *

def computeshr01(row):
    if model == 'NSC3km-12sec': return np.sqrt(row['USHR1-potential_mean']**2 + row['VSHR1-potential_mean']**2)
    if model == 'NCAR': return np.sqrt(row['UBSHR1-potential_mean']**2 + row['VBSHR1-potential_mean']**2)

def computeshr06(row):
    if model == 'NSC3km-12sec': return np.sqrt(row['USHR6-potential_mean']**2 + row['VSHR6-potential_mean']**2)
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
        if model == 'NSC3km-12sec': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv_preprocessed/track_step_NCARSTORM_d01_%s-0000_13_time2_filtered.csv'%(yyyymmdd)
        if model == 'NCAR': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncar_2016_csv_preprocessed/track_step_ncar_3km_%s_time2.csv'%(yyyymmdd)

        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dt.timedelta(days=1)
    print 'Reading %s files'%(len(all_files))
    
    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    df['shr01'] = df.apply(computeshr01, axis=1)
    df['shr06'] = df.apply(computeshr06, axis=1)

    if model == 'NSC3km-12sec': df['stp']   = df.apply(computeSTP, axis=1)

    if model == 'NSC3km-12sec': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    if model == 'NCAR':
        df['datetime']  = pd.to_datetime(df['Date'])
        df['Run_Date']  = pd.to_datetime(df['Date']) - pd.to_timedelta(df['Forecast_Hour'])
    df['year']     = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    if model == 'NCAR': df = df[df['Forecast_Hour']>12]

    return df, len(all_files)

def print_scores(labels, predictions):
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

    print 'BIAS=%0.3f, POD=%0.3f, FAR=%0.3f, POFD=%0.3f, ETS=%0.3f'%(bias,pod,far,pofd,ets)

def writeOutputSparse():
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    outnc = Dataset('%s/ssr_sparse_grid81_%s_RANDOM-FOREST%dkm_%s.nc'%(out_dir,model,d,yyyymmddhh), 'w')
    outnc.description = 'Surrogate Severe Forecast Data'
    outnc.source = model
    outnc.field = 'RANDOM-FOREST'
    outnc.grid  = 81
    outnc.thresh = 0
    outnc.numens = 1
    outnc.fhours = 37
    outnc.lats = 93
    outnc.lons = 65
    outnc.history = 'Created '+ time.ctime(time.time())

    ssrindx = np.flatnonzero(ssr81_all)
    outnc.createDimension('ssrindx', ssrindx.size)
    #outnc.createDimension('numthresh', len(threshList))

    if ssrindx.size > 0:
      if ssrindx.max() > 4290000000: type = 'u8' #64-bit unsigned int
      else: type = 'u4' # 32-bit unsigned int
    else: type = 'u4'

    ssrloc   = outnc.createVariable('ssrloc', type, ('ssrindx',), zlib=True)
    ssrmag   = outnc.createVariable('ssrmag', 'f4', ('ssrindx',), zlib=True)
    #thresh   = outnc.createVariable('thresh', 'f4', ('numthresh',), zlib=True)
    #ssrs   = outnc.createVariable('ssrs', type, ('ssrindx',))
    #thresh   = outnc.createVariable('thresh', 'f4', ('numthresh',))

    ssr81_all_flat = ssr81_all.flatten()
    #ssrs[:] = ssrindx
    ssrloc[:] = ssrindx
    ssrmag[:] = ssr81_all_flat[ssrindx]
    #thresh[:] = threshList
    outnc.close()

def plot_forecast(storms, predictions, yyyymmdd):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)

    lats, lons, fh = storms['Centroid_Lat'].values, storms['Centroid_Lon'].values, storms['Forecast_Hour'].values
 
    # map storm predictions to 80km grid
    x, y = awips(lons, lats)
    print 'making 80-km grid'
    nngridpts = tree.query(zip(x.ravel(), y.ravel()))
    ssr81_all = np.zeros((1,37,65*93), dtype='f')
    for i in range(len(storms)):
        if predictions[i][1] > 0.5:
            ssr81_all[0,fh[i],nngridpts[1][i]] = 1
    #ssr81_all = np.amax(ssr81_all[:,13:37,:], axis=(0,1))
    ssr81_all = ssr81_all.reshape((1,37,65,93))
    print ssr81_all.sum()

    #x, y = m(lons, lats)
    #cmap = plt.get_cmap('RdGy_r')
    #norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
    #a = m.scatter(x, y, s=40, c=predictions[:,1], lw=0.5, edgecolors='k', cmap=cmap, norm=norm)
    
    # ADD COLORBAR
    #cax = fig.add_axes([0.02,0.1,0.02,0.3])
    #cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    #cb.outline.set_linewidth(0.5)
    #cb.ax.tick_params(labelsize=10)
 
    #plt.savefig('forecast_%s.png'%yyyymmdd)

    return ssr81_all
    
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
lat_coarse,lon_coarse,x_coarse,y_coarse = awips.makegrid(93, 65, returnxy=True)
tree = spatial.KDTree(zip(x_coarse.ravel(),y_coarse.ravel()))
fig, axes, m  = pickle.load(open('/glade/u/home/sobash/RT2015_gpx/rt2015_ch_CONUS.pk', 'r'))

model = 'NSC3km-12sec'
out_dir         = '/glade/work/sobash/SSR/ssr_sparse_grid81_%s_max'%(model)
sdate = dt.datetime(2012,1,1,0,0,0)
edate = dt.datetime(2012,12,31,0,0,0)
dateinc = dt.timedelta(days=1)
df, numfcsts = read_csv_files()
d = 120

if model == 'NSC3km-12sec':
  features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01',\
              'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
               'UP_HELI_MAX_mean', \
               'UP_HELI_MIN_mean', \
               'UP_HELI_MAX01_mean',\
               'UP_HELI_MAX03_mean',\
               'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean',\
               'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
               'PSFC-potential_mean', 'orientation']

  # use these features when training NSC to be used on NCAR ensemble forecasts
  #features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01','MUCAPE-potential_mean','SBCAPE-potential_mean','UP_HELI_MIN_min','UP_HELI_MAX_mean','W_UP_MAX_max','WSPD10MAX_mean',\
   #         'SBCINH-potential_mean','SRH03-potential_mean','SBLCL-potential_mean',\
   #         'UP_HELI_MAX_min', 'W_UP_MAX_min', 'W_DN_MAX_min', 'WSPD10MAX_min','orientation']
if model == 'NCAR':
  features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01','MUCAPE-potential_mean','CAPE_SFC-potential_mean','UP_HELI_MIN_min','UP_HELI_MAX_mean','W_UP_MAX_max','WSPD10MAX_mean',\
            'CIN_SFC-potential_mean','SRH3-potential_mean', 'LCL_HEIGHT-potential_mean',\
            'UP_HELI_MAX_min', 'W_UP_MAX_min', 'W_DN_MAX_min', 'WSPD10MAX_min','orientation']

print 'Reading random forest classifier'
rf = pickle.load(open('rf_severe_%dkm_NSC_test2012.pk'%d, 'rb'))

print 'Predicting'
predictions = rf.predict(df[features])
predictions_proba = rf.predict_proba(df[features])

labels   = ((df['hail_report_closest_distance'] < d*1000.0) & (df['hail_report_closest_distance'] > 0)) |  \
           ((df['wind_report_closest_distance'] < d*1000.0) & (df['wind_report_closest_distance'] > 0)) | \
           ((df['torn_report_closest_distance'] < d*1000.0) & (df['torn_report_closest_distance'] > 0))

print_scores(labels, predictions)

print np.histogram(predictions_proba[:,1])
print calibration_curve(labels, predictions_proba[:,1], n_bins=10)
print metrics.roc_auc_score(labels, predictions_proba[:,1])

tdate = sdate
while tdate <= edate:
    csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv_preprocessed/track_step_NCARSTORM_d01_%s-0000_13_time2_filtered.csv'%(tdate.strftime('%Y%m%d'))
    if model == 'NCAR': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncar_2016_csv_preprocessed/track_step_ncar_3km_%s_time2.csv'%(tdate.strftime('%Y%m%d'))
    if os.path.exists(csv_file):
        print 'plotting', tdate
        yyyymmdd = tdate.strftime('%Y-%m-%d %H:%M:%S')
        yyyymmddhh = tdate.strftime('%Y%m%d%H')
        forecast_mask = (df['Run_Date'] == yyyymmdd)
        #forecast_mask = (df['Run_Date'] >= '2012-01-01 00:00:00')
        ssr81_all = plot_forecast(df[forecast_mask], predictions_proba[forecast_mask], tdate.strftime('%Y%m%d'))
        writeOutputSparse() 
    tdate +=dateinc
