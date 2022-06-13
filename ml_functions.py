#!/usr/bin/env python

import datetime as dt
import glob
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import numpy as np
import hwtmode.statisticplot
import scipy.ndimage.filters
from sklearn.calibration import calibration_curve
from sklearn import metrics
from tensorflow import is_tensor
from tensorflow.keras import backend as K
import time, os
import xarray

def log(msg, flush=True):
    print( time.ctime(time.time()), msg , flush=flush)

def brier_skill_score(obs, preds):
    if is_tensor(obs) and is_tensor(preds): 
        bs = K.mean((preds - obs) ** 2)
        obs_climo = K.mean(obs, axis=0) # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
        bs_climo = K.mean((obs - obs_climo) ** 2)
        #bss = 1.0 - (bs/bs_climo+K.epsilon()) # TODO: shouldn't K.epsilon() be grouped with denominator?
        bss = 1.0 - bs/(bs_climo+K.epsilon())
    else:
        bs = np.mean((preds - obs) ** 2)
        obs_climo = np.mean(obs, axis=0) # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
        bs_climo = np.mean((obs - obs_climo) ** 2)
        bss = 1 - bs/bs_climo

    return bss

def rptdist2bool(df, rptdist, twin):
    # get rid of columns that are not associated with the time window (twin)
    dropcol=[]
    for r in ["sighail", "sigwind", "hailone", "wind", "torn"]:
        for h in [0,1,2]:
            if h != twin:
                dropcol.append(f"{r}_rptdist_{h}hr")
    df = df.drop(columns=dropcol)
    # keep track of new Boolean column names
    rptcols = []
    for r in ["sighail", "sigwind", "hailone", "wind", "torn"]:
        rh = f"{r}_rptdist_{twin}hr"
        # Convert severe report distance to boolean (0-rptdist = True)
        df[rh] = (df[rh] >= 0) & (df[rh] < rptdist)
        rptcols.append(rh)

    # Any report
    any_rpt_col = f"any_rptdist_{twin}hr"
    hailwindtorn = [f"{r}_rptdist_{twin}hr" for r in ["hailone","wind","torn"]]
    df[any_rpt_col] = df[hailwindtorn].any(axis="columns")
    rptcols.append(any_rpt_col)
    return df, rptcols

def get_glm(twin,rptdist,date=None):
    assert twin == 2, "get_glm assumes time window is 2, not {twin}"
    logging.info(f"load {twin}h {rptdist}km GLM")
    if date:
        logging.info(f"date={date}")
        glmfiles = sorted(glob.glob(f"/glade/work/ahijevyc/GLM/{date.strftime('%Y%m%d')}*.glm.nc"))
        glm = xarray.open_mfdataset(glmfiles, concat_dim="time_coverage_start", combine="nested")
    else:
        oneGLMfile = True
        if oneGLMfile:
            glm = xarray.open_dataset("/glade/scratch/ahijevyc/temp/GLM_all.nc")
        else:
            glmfiles = sorted(glob.glob("/glade/work/ahijevyc/GLM/2*.glm.nc"))
            #glmtimes = [datetime.datetime.strptime(os.path.basename(x), "%Y%m%d%H.glm.nc") for x in glmfiles] # why is this here?
            logging.info("open_mfdataset")
            glm = xarray.open_mfdataset(glmfiles, concat_dim="time_coverage_start", combine="nested")

    assert (glm.time_coverage_start[1] - glm.time_coverage_start[0]) == np.timedelta64(3600,'s'), 'glm.time_coverage_start interval not 1h'
    logging.info("Add flashes from previous 2 times and next time to current time. 4-hour centered time window")
    glm = glm + glm.shift(time_coverage_start=2) + glm.shift(time_coverage_start=1) + glm.shift(time_coverage_start=-1)

    if rptdist != 40:
        k = int(rptdist/40)
        logging.warning(f"this is not correct. the window overlaps masked points")
        logging.warning(f"TODO: save GLM in non-masked form, or filter it from 40km to 120km while unmasked and making the GLM files.")
        logging.info(f"sum GLM flash counts in {k}x{k} window")
        glm = glm.rolling(x=k, y=k, center=True).sum()

    glm = glm.rename(dict(time_coverage_start="valid_time", y="projection_y_coordinate", x="projection_x_coordinate"))
    return glm


def print_scores(obs, fcst, label, desc="", n_bins=10, debug=False):

    # print scores for this set of forecasts
    # histogram of probability values
    print(np.histogram(fcst, bins=n_bins))

    # reliability curves
    true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=n_bins)
    for o, f in zip(true_prob, fcst_prob): print(o, f)

    print('brier score', np.mean((obs-fcst)**2))

    # BSS
    bss_val = hwtmode.statisticplot.bss(obs, fcst)
    print('bss', bss_val)

    # ROC auc
    auc = metrics.roc_auc_score(obs, fcst)
    print('auc', auc)

    # copied and pasted from ~ahijevyc/bin/reliability_curve_MET.py 
    """calibration curve """
    fig_index=1
    fig = plt.figure(fig_index, figsize=(10, 7))
    ax1 = plt.subplot2grid((3,2), (0,0), rowspan=2)
    reld = hwtmode.statisticplot.reliability_diagram(ax1, obs, fcst, label=label, n_bins=n_bins, debug=debug)
    ax1.tick_params(axis='x', labelbottom=False)
 
    """histogram of counts"""
    ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, sharex=ax1)
    histogram_of_counts = hwtmode.statisticplot.count_histogram(ax2, fcst, label=label, n_bins=n_bins, debug=debug)
    ROC_ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
    roc_curve = hwtmode.statisticplot.ROC_curve(ROC_ax, obs, fcst, label=label, sep=0.1, debug=debug)
    fineprint = f"{desc} {label}\ncreated {str(dt.datetime.now(tz=None)).split('.')[0]}"
    plt.annotate(s=fineprint, xy=(1,1), xycoords=('figure pixels', 'figure pixels'), va="bottom", fontsize=5) 
    ofile = f'{desc}.{label}.png'
    plt.savefig(ofile)
    print("made", os.path.realpath(ofile))
    return true_prob, fcst_prob, bss_val, auc

def upscale(field, nngridpts, type='mean', maxsize=27):
    if type == 'mean':
        field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode='nearest')
        #field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode='constant') # mode shouldn't matter. mask takes over. But it does matter for pre-Apr 21, 2015 T2 field.
        # For some reason, in this case T2 is transformed to 0-4 range
    elif type == 'max':
        field = scipy.ndimage.filters.maximum_filter(field, size=maxsize)
    elif type == 'min':
        field = scipy.ndimage.filters.minimum_filter(field, size=maxsize)

    field_interp = field.flatten()[nngridpts[1]].reshape((65,93))

    return field_interp

def get_features(subset='all'):
    # complex features
    explicit_features = [ 'COMPOSITE_REFL_10CM', 'REFD_MAX', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'RAINNC_1H' ]
    env_features      = [ 'MUCAPE', 'SBCAPE', 'SBCINH', 'SHR01', 'SHR06', 'MLCINH', 'MLLCL', 'SRH01', 'SRH03', 'T2', 'TD2', 'PSFC','CAPESHEAR', 'STP', 'LR75' ]
    env_features.remove('PSFC') # suspect before Sep 2015
    static_features   = [ 'fhr', 'dayofyear', 'lat', 'lon', 'hgt' ]

    large_scale_features = ['U925','U850','U700','U500','V925','V850','V700','V500','T925','T850','T700','T500','TD925','TD850','TD700','TD500']

    simple_max_fields = ['COMPOSITE_REFL_10CM', 'REFD_MAX', 'UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'RAINNC_1H']
    simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'MLCINH', 'SBCINH', 'MLLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']


    nbrs = [3,5]
    if "7x7" in subset:
        nbrs = [3,5,7]
    simple_max_features = [ f+'-N%dT%d'%(x,t) for f in simple_max_fields for x in nbrs for t in [1,3,5] ]
    simple_mean_features = [ f+'-N%dT%d'%(x,t) for f in simple_mean_fields for x in nbrs for t in [1,3,5] ]


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

    if subset[0:11] == 'basic_nbrhd': features = basic_features + simple_mean_features + simple_max_features


    if subset == 'storm':
        features = ["SBCAPE", "UP_HELI_MAX", "W_UP_MAX", "SHR06", "CAPESHEAR", "TD2", "PREC_ACC_NC", "WSPD10MAX", "STP", "GRPL_MAX", "HGT0C", "CAPESHEAR-N3T1"]
    if subset == 'env':
        features = ["SBCAPE", "SHR06", "CAPESHEAR", "TD2", "STP", "HGT0C", "CAPESHEAR-N3T1"]
    if 'ens_mean' in subset:
        features_c = features.copy()
        for fcn in ["std", "max", "min"]:
            if '_'+fcn in subset:
                # add " xxx" versions of each feature
                features += [x + " " + fcn for x in features_c]
                # remove functions of static features
                for static in ["lon", "lat", "hgt", "fhr", "dayofyear"]:
                    features.remove(static + " " + fcn)

    features = list(set(features)) # no duplicates

    features.sort()

    return features

def read_csv_files(sdate, edate, dataset, members=[str(x) for x in range(1,11)], columns=None):
    # read in all CSV files for 1km forecasts
    all_files = []
    for member in members:
        all_files.extend(glob.glob(f'/glade/scratch/ahijevyc/NSC_objects/grid_data_{dataset}_mem{member}_d01_????????-0000.par'))
    all_files = set(all_files) # in case you ask for same member twice
    log("found "+str(len(all_files))+" files")
    all_files = [x for x in all_files if sdate.strftime('%Y%m%d') <= x[-17:-9] <= edate.strftime('%Y%m%d')]
    all_files = sorted(all_files) # important for predictions_labels output
    log(f'Reading {len(all_files)} forecasts from {sdate} to {edate}')

    #df = pd.concat((pd.read_csv(f, compression='gzip', dtype=type_dict) for f in all_files))
    #df = pd.concat((pd.read_csv(f, dtype=type_dict) for f in all_files))
    ext = all_files[0][-4:]
    if ext == ".csv": df = pd.concat((pd.read_csv(f, engine='c') for f in all_files), sort=False)
    elif ext == ".par":
        df = pd.concat((pd.read_parquet(f) for f in all_files), sort=False)
    else:
        print("unexpected extension", ext,"exiting")
        sys.exit(1)
    log('finished reading')
    if 'member' not in df.columns: # started adding member in previous step (random_forest_preprocess_gridded.py)
        log('adding members column')
        import re
        member = [re.search("_mem(\d+)",s).groups()[0] for s in all_files]
        # repeat each element n times. where n is number of rows in a single file's dataframe
        # avoid ValueError: Length of values does not match length of index
        df['member'] = np.repeat(np.int8(member), len(df)/len(all_files))
    #if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)   
    #if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    #df = df.reset_index(level=0).pivot(columns="level_0")
    #df.columns = [' '.join(col).strip() for col in df.columns.values]
    #df = df.reset_index('Date') 
    if 'datetime' not in df.columns:
        log('adding datetime')
        df['datetime']  = pd.to_datetime(df['Date'])
    #df['Run_Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    if 'year' not in df.columns:
        log('adding year')
        df['year']      = df['datetime'].dt.year.astype(np.uint16)
    if 'month' not in df.columns:
        log('adding month')
        df['month']     = df['datetime'].dt.month.astype(np.uint8)
    if 'hour' not in df.columns:
        log('adding hour')
        df['hour']      = df['datetime'].dt.hour.astype(np.uint8)
    if 'dayofyear' not in df.columns:
        log('adding dayofyear')
        df['dayofyear'] = df['datetime'].dt.dayofyear.astype(np.uint16)
    log('leaving read_csv()')



    return df, len(all_files)

def normalize_multivariate_data(data, features, scaling_values=None, nonormalize=False, debug=False):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: Pandas DataFrame (not normalized, could have extraneous columns not in features list)
        features: list of features
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    log(data.shape)
    if hasattr(data, "dtype"):
        log(data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        data = data[features]
        log(data.info())
        scaling_values = pd.DataFrame(columns=scale_cols)
        scaling_values["mean"] = data.mean()
        log(scaling_values.info())
        scaling_values["std"] = data.std()
        log(scaling_values.info())
        if nonormalize:
            print("ml_functions.normalize_multivariate_data(): no normalization. returning scaling_values")
            return None, scaling_values
        data = data.values 


    normed_data = np.zeros(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        #normed_data[:, i] = (data[:, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
        normed_data[:, i] = (data[:, i] - scaling_values.loc[features[i], "mean"]) / scaling_values.loc[features[i], "std"]
        if debug:
            print()
            log(scaling_values.loc[features[i]])
        else:
            print(features[i])
    return normed_data, scaling_values
