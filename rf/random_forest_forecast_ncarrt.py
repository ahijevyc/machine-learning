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
import cartopy
from cartopy.geodesic import Geodesic
from matplotlib.path import Path


def usa_filter(df):
    # convert storm lat lons to map proj coords
    lats, lons = df['Centroid_Lat'].values, df['Centroid_Lon'].values
    x, y = awips_proj(lons, lats)
    storm_proj = np.array((x.flatten(), y.flatten())).T

    mask = usaPath.contains_points(storm_proj)

    return df[mask]

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
    old_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        
        for i in range(1,11):
            csv_file = '/glade/work/sobash/NSC_objects/track_data_ncargrib_2019_csv/track_step_NCARGRIB_mem%d_%s-0000_13.csv'%(i,yyyymmdd)
            #csv_file = '/glade/work/sobash/NSC_objects/track_data_ncar_2019_csv/track_step_NCAR_mem%d_%s-0000_13.csv'%(i,yyyymmdd)
            if os.path.exists(csv_file): all_files.append(csv_file)

        tdate += dt.timedelta(days=1)

    print 'Reading %s files'%(len(all_files))
    
    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    df['shr01'] = df.apply(computeshr01, axis=1)
    df['shr06'] = df.apply(computeshr06, axis=1)

    if model == 'NSC3km-12sec': df['stp']   = df.apply(computeSTP, axis=1)

    #if model == 'NSC3km-12sec': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    #if model == 'NCAR':
    #    df['datetime']  = pd.to_datetime(df['Date'])
    #    df['Run_Date']  = pd.to_datetime(df['Date']) - pd.to_timedelta(df['Forecast_Hour'])
    df['datetime']  = pd.to_datetime(df['Valid_Date'])
    df['year']     = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    if model == 'NCAR': df = df[df['Forecast_Hour']>12]

    if remove_ocean_storms: df = usa_filter(df)

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
    outnc = Dataset('%s/ssr_sparse_grid81_NCAR2019_RANDOM-FOREST%dkm_%s.nc'%(out_dir,d,yyyymmddhh), 'w')
    #outnc = Dataset('%s/ssr_sparse_grid81_NCAR2019_UP_HELI_MAX_%s.nc'%(out_dir,yyyymmddhh), 'w')
    outnc.description = 'Surrogate Severe Forecast Data'
    outnc.source = model
    #outnc.field = 'UP_HELI_MAX'
    outnc.field = 'RANDOM-FOREST'
    outnc.grid  = 81
    outnc.thresh = 1 
    outnc.numens = num_mem
    outnc.fhours = num_fhr
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

def plot_forecast(storms, yyyymmdd):
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    #cmap = ListedColormap(test)

    lats, lons, fh = storms['Centroid_Lat'].values, storms['Centroid_Lon'].values, storms['Forecast_Hour'].values
    uh = storms['UP_HELI_MAX_max'].values
    predictions = storms['predict_proba'].values
    members = storms['Ensemble_Member'].values

    # map storm predictions to 80km grid
    if grid_ssrs:
        x, y = awips(lons, lats)
        print 'making 80-km grid'
        nngridpts = tree.query(zip(x.ravel(), y.ravel()))
        ssr81_all = np.zeros((num_mem,num_fhr,65*93), dtype='f')

        for n in range(1,num_mem+1):
            member_mask = (members == 'mem%d'%n)
            this_prediction, this_fh, this_nngridpts, this_uh = predictions[member_mask], fh[member_mask], nngridpts[1][member_mask], uh[member_mask]

            for i in range(len(this_prediction)):
                if this_prediction[i] > ssr81_all[n-1,this_fh[i],this_nngridpts[i]]:
                    ssr81_all[n-1,this_fh[i],this_nngridpts[i]] = this_prediction[i]
                #if this_uh[i] > ssr81_all[n-1,this_fh[i],this_nngridpts[i]]:
                #    ssr81_all[n-1,this_fh[i],this_nngridpts[i]] = this_uh[i]

        #ssr81_all = np.amax(ssr81_all[:,13:37,:], axis=(0,1))
        ssr81_all = ssr81_all.reshape((num_mem,num_fhr,65,93))
        print ssr81_all.sum()

    fig, axes, m  = pickle.load(open('/glade/work/sobash/NSC_objects/hwt2019_domain.pk', 'r'))
    x81, y81 = m(lon_coarse, lat_coarse)
    x, y = m(lons, lats)
    cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
    a = m.scatter(x, y, s=40, c=predictions, lw=0.25, edgecolors='k', cmap=cmap, norm=norm)

    #a = m.pcolormesh(x81, y81, np.ma.masked_less(np.amax(ssr81_all[0,13:37,:], axis=0), 0.5), alpha=0.8, edgecolor='None', linewidth=0.05, cmap=cmap, norm=norm, ax=axes)
 
    # ADD TITLE
    fontdict = {'family':'monospace', 'size':12, 'weight':'bold'}
    x0, y1 = axes.transAxes.transform((0,1))
    x0, y0 = axes.transAxes.transform((0,0))
    x1, y1 = axes.transAxes.transform((1,1))
    axes.text(x0, y1+10, 'NCAR Ensemble 24-hr Random Forest Severe Weather Guidance', fontdict=fontdict, transform=None)

    initstr  = sdate.strftime('Init: %a %Y-%m-%d %H UTC')
    validstr1 = (sdate+dt.timedelta(hours=12)).strftime('%a %Y-%m-%d %H UTC')
    validstr2 = (sdate+dt.timedelta(hours=36)).strftime('%a %Y-%m-%d %H UTC')
    validstr = "Valid: %s - %s"%(validstr1, validstr2)

    fontdict = {'family':'monospace', 'size':10 }
    axes.text(x1, y1+20, initstr, horizontalalignment='right', transform=None, fontdict=fontdict)
    axes.text(x1, y1+5, validstr, horizontalalignment='right', transform=None, fontdict=fontdict)

    axes.text(x0, y0-15, 'Circles denote locations of storm centroids with w > 10 m/s for all members', fontdict=fontdict, transform=None)
    axes.text(x0, y0-28, 'Fill color denotes prob. of any severe report w/in 120-km using a random forest trained with 3-km CAM forecasts', fontdict=fontdict, transform=None) 
    axes.text(x0, y0-41, 'Questions/Feedback: sobash@ucar.edu', fontdict=fontdict, transform=None) 

    fontdict = {'family':'monospace', 'size':11, 'weight':'bold' }
    axes.text(x0+600, y0+60, 'Total storms: %d'%len(predictions), fontdict=fontdict, transform=None)
    axes.text(x0+600, y0+45,  'Severe storms: %d'%len(predictions[predictions>0.5]), fontdict=fontdict, transform=None)
    axes.text(x0+600, y0+30,  'Average prob: %0.2f'%np.mean(predictions), fontdict=fontdict, transform=None)
   
    #code for inset plot
    #cax = fig.add_axes([0.11,0.1,0.3,0.1])
    #cax.set_facecolor('#EFEFEF')
    #bars = cax.hist(predictions, np.arange(0,1.01,0.05), color='gray')
    #bars = cax.bar(range(len(bins[:-1])), num_per_bin, color='gray')
    #cax.set_xticklabels([-1,0,0.25,0.5,0.75,1.0])
    #cax.tick_params(direction='in', pad=0.05)
    #for p in ['left','right','top','bottom']: cax.spines[p].set_visible(False)

    # ADD COLORBAR
    cax = fig.add_axes([0.02,0.1,0.02,0.3])
    cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=10)
 
    plt.savefig('random_forest_severe_%s_day1.png'%yyyymmddhh)
    return ssr81_all
    
grid_ssrs = True
if grid_ssrs:
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    lon_coarse,lat_coarse,x_coarse,y_coarse = awips.makegrid(93, 65, returnxy=True)
    tree = spatial.KDTree(zip(x_coarse.ravel(),y_coarse.ravel()))

model = 'NCAR'
remove_ocean_storms = True
out_dir         = '/glade/work/sobash/SSR/ssr_sparse_grid81_NCAR2019_max'
#out_dir         = '/glade/work/sobash/NSC_objects'
 
num_mem = 10
#num_mem = 1
num_fhr = 37

sdate = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H')
edate = sdate
dateinc = dt.timedelta(days=1)
d = 120

if remove_ocean_storms:
    # READ IN US LAT/LON FILE
    fh = open('/glade/u/home/sobash/2013RT/uspoints', 'r')
    pts = fh.readlines()[0].split(',0')
    pts2 = [a.split(',') for a in pts]
    pts3 = np.array(pts2[:-1], dtype=float)
    print pts3.shape

    # AWIPS 80-KM GRID POINTS
    awips_proj = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95)

    # CONVERT US LAT LONS TO MAP PROJ COORDS 
    us_path_proj = awips_proj(pts3[:,0], pts3[:,1])
    usaPath = Path(np.array(us_path_proj).T)

df, numfcsts = read_csv_files()

features = ['Valid_Hour_UTC','dayofyear','Centroid_Lat', 'Centroid_Lon','shr06','shr01',\
              'MUCAPE-potential_mean','SBCAPE-potential_mean','MLCAPE-potential_mean',\
               'UP_HELI_MAX_mean', \
               'UP_HELI_MIN_mean', \
               #'UP_HELI_MAX01_mean',\
               'UP_HELI_MAX03_mean',\
               'W_UP_MAX_mean','W_DN_MAX_mean','WSPD10MAX_mean',\
               'SBCINH-potential_mean','SRH01-potential_mean','SRH03-potential_mean', 'SBLCL-potential_mean','T2-potential_mean','TD2-potential_mean',\
               'PSFC-potential_mean', 'orientation']

print 'Reading random forest classifier'
#rf = pickle.load(open('rf_severe_%dkm_NSC_test2017.pk'%d, 'rb'))
#rf = pickle.load(open('rf_severe_%dkm_NSC_test2018.pk'%d, 'rb')) #all NSC storms
rf = pickle.load(open('rf_severe_%dkm_NSC_test2018_noUH01.pk'%d, 'rb')) #all NSC storms (without UH01, that isnt stored in 2019 NCAR ensemble grib files)
print rf

print 'Predicting'
predictions = rf.predict(df[features])
predictions_proba = rf.predict_proba(df[features])

print np.histogram(predictions_proba[:,1])

df['predict_proba'] = predictions_proba[:,1]
df = df.sort_values(by=['predict_proba'])

num_per_bin = []
bins = np.arange(0,1.01,0.05)
for i in range(len(bins)-1):
    forecast_mask = (df['predict_proba'] >= bins[i]) & (df['predict_proba'] < bins[i+1])
    num_per_bin.append(len(df[forecast_mask].values))
    #prob_per_hour.append(df[forecast_mask]['predict_proba'].mean())
    print bins[i], num_per_bin

tdate = sdate
while tdate <= edate:
    csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv_preprocessed/track_step_NCARSTORM_d01_%s-0000_13_time2_filtered.csv'%(tdate.strftime('%Y%m%d'))
    if model == 'NCAR': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncargrib_2019_csv/track_step_NCARGRIB_mem1_%s-0000_13.csv'%(tdate.strftime('%Y%m%d'))
    yyyymmdd = tdate.strftime('%Y-%m-%d %H:%M:%S')
    yyyymmddhh = tdate.strftime('%Y%m%d%H')

    if os.path.exists(csv_file):
        print 'plotting', tdate
        #forecast_mask = (df['Run_Date'] == yyyymmdd) & (df['Forecast_Hour'] > 12) & (df['Forecast_Hour'] < 36)
        forecast_mask = (df['Run_Date'] == yyyymmdd))
        ssr81_all = plot_forecast(df[forecast_mask], tdate.strftime('%Y%m%d'))
        writeOutputSparse() 
    tdate +=dateinc
