#!/usr/bin/env python

import time, os
import pandas as pd
import numpy as np
import datetime as dt

def log(msg):
    print( time.ctime(time.time()), msg )

def get_features(subset, use_nschrrr_features):
    # complex features
    explicit_features = [ 'UP_HELI_MAX', 'UP_HELI_MAX03', 'RVORT1', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC' ]
    env_features      = [ 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'MLCINH', 'SBLCL', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75' ]
    static_features   = [ 'fhr', 'dayofyear', 'lat', 'lon' ]
    large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']
    simple_max_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'RVORT1', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC']
    simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SBLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']
   
    # use the set of features available in both the NSC and HRRR datasets (no UH01, SBLCL) 
    if use_nschrrr_features:
        explicit_features = [ 'UP_HELI_MAX', 'UP_HELI_MAX03', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC' ]
        env_features      = [ 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'MLCINH', 'SHR01', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75' ]
        simple_max_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC']
        simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']

    simple_max_features = [ f+'-N%dT%d'%(x,t) for f in simple_max_fields for x in [3,5] for t in [1,3,5] ]
    simple_mean_features = [ f+'-N%dT%d'%(x,t) for f in simple_mean_fields for x in [3,5] for t in [1,3,5] ]

    basic_features = static_features + explicit_features + env_features
    
    # all fields
    if subset == 'all': features = static_features + explicit_features + env_features + large_scale_features + simple_max_features + simple_mean_features

    # UH only
    if subset == 'uhonly': features = static_features + ['UP_HELI_MAX', 'UP_HELI_MAX-N3T1', 'UP_HELI_MAX-N3T3', \
                                       'UP_HELI_MAX-N3T5', 'UP_HELI_MAX-N5T1', 'UP_HELI_MAX-N5T3', 'UP_HELI_MAX-N5T5']
    # basic features only
    if subset == 'basic': features = basic_features

    # basic + largescale only
    if subset == 'basiclarge': features = basic_features + large_scale_features

    # environmental features only
    if subset == 'envonly': features = static_features + env_features + large_scale_features + simple_mean_features

    # no upper air features (this also removed the explicit features accidentally...
    if subset == 'noupperair': features = static_features + env_features + simple_mean_features + simple_max_features

    return features

def read_csv_files(sdate, edate, dataset):
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        yyyymmddhh = tdate.strftime('%Y%m%d%H')

        #csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_%s_d01_%s-0000.csv'%(dataset,yyyymmdd)
        csv_file = '/glade/work/sobash/NSC_objects/grid_data/grid_data_%s_d01_%s-0000.par'%(dataset,yyyymmddhh)

        if dataset == 'RT2020': csv_file = '/glade/work/sobash/NSC_objects/RT2020/grid_data_NCARSTORM_d01_%s-0000_verif.csv'%(yyyymmdd)
        #if dataset == 'HRRR': csv_file = '/glade/work/sobash/NSC_objects/HRRR/grid_data/grid_data_HRRR_d01_%s-0000.csv'%(yyyymmddhh)
        if dataset in ['HRRR', 'HRRRX']: csv_file = '/glade/work/sobash/NSC_objects/HRRR/grid_data/grid_data_%s_d01_%s-0000.par'%(dataset,yyyymmddhh)
        if dataset == 'GEFS': csv_file = '/glade/work/sobash/NSC_objects/grid_data_ncarstorm_3km_csv_preprocessed/grid_data_%s_mem%d_d01_%s-0000.csv'%(dataset,mem,yyyymmdd)

        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dt.timedelta(days=1)
    log('Reading %s forecasts'%(len(all_files)))

    #df = pd.concat((pd.read_csv(f, compression='gzip', dtype=type_dict) for f in all_files))
    #df = pd.concat((pd.read_csv(f, dtype=type_dict) for f in all_files))
    df = pd.concat((pd.read_parquet(f) for f in all_files))

    # check if we need to convert to float32

    log('computing fields')
    #if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)   
    #if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    df['datetime']  = pd.to_datetime(df['Date'])
    #df['Run_Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    df['year']      = df['datetime'].dt.year
    df['month']     = df['datetime'].dt.month
    df['hour']      = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear

    return df, len(all_files)

def normalize_multivariate_data(data, features, scaling_values=None):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    log('%s, %s'%(data.shape, data.dtype))
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols, index=features)
        #for i in range(data.shape[-1]): scaling_values.loc[i, ["mean", "std"]] = [data[:, i].mean(), data[:, i].std()]
        for i in range(data.shape[-1]): scaling_values.loc[features[i], ["mean", "std"]] = [data[:, i].mean(), data[:, i].std()]

    for i in range(data.shape[-1]):
        #normed_data[:, i] = (data[:, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
        normed_data[:, i] = (data[:, i] - scaling_values.loc[features[i], "mean"]) / scaling_values.loc[features[i], "std"]
    return normed_data, scaling_values
