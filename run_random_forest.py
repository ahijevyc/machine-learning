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
import pandas as pd

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    fh = open('/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps/%s.rgb'%name, 'r')
    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def computeshr01(row):
    return np.sqrt(row['USHR1-potential_mean']**2 + row['VSHR1-potential_mean']**2)

def computeshr06(row):
    return np.sqrt(row['USHR6-potential_mean']**2 + row['VSHR6-potential_mean']**2)

def computeSTP(row):
    lclterm = ((2000.0-row['MLLCL-potential_mean'])/1000.0)
    lclterm = np.where(row['MLLCL-potential_mean']<1000, 1.0, lclterm)
    lclterm = np.where(row['MLLCL-potential_mean']>2000, 0.0, lclterm)

    shrterm = (row['shr06']/20.0)
    shrterm = np.where(row['shr06'] > 30, 1.5, shrterm)
    shrterm = np.where(row['shr06'] < 12.5, 0.0, shrterm)

    stp = (row['SBCAPE-potential_mean']/1500.0) * lclterm * (row['SRH01-potential_mean']/150.0) * shrterm
    return stp

def read_csv_files(r):
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        if r == '1km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_1km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        elif r == '3km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print 'Reading %s files'%(len(all_files))

    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    #df['shr01'] = df.apply(computeshr01, axis=1)
    #df['shr06'] = df.apply(computeshr06, axis=1)
    #df['stp'] = df.apply(computeSTP, axis=1)   
    #df['ratio'] = df['RVORT1_MAX_max'] / df['RVORT5_MAX_max']
 
    return df, len(all_files)

sdate = dt.datetime(2010,10,1,0,0,0)
edate = dt.datetime(2017,10,1,0,0,0)
dateinc = dt.timedelta(days=1)

df, numfcsts = read_csv_files('3km')


print df[df['UP_HELI_MAX01_max'] > 14.362][['UP_HELI_MAX01_max', 'Centroid_Lat' ,'Centroid_Lon', 'Run_Date', 'Forecast_Hour']]

