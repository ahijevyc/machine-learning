import argparse
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
import sys
from tensorflow import is_tensor
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
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


def configs_match(ylargs, args):
    # Make sure training time range doesn't overlap test range
    trainstart = getattr(ylargs,"trainstart")
    trainend   = getattr(ylargs,"trainend")
    teststart = args.teststart
    testend   = args.testend
    overlap = min([trainend, testend]) - max([trainstart, teststart])
    if overlap >= dt.timedelta(hours=0) and args.kfold == 1:
        logging.warning(f"training and testing time ranges overlap {trainstart}-{trainend}|{teststart}-{testend}")
   
    # Comparing config.yaml training bounds and requested training bounds (args) is not a good test because config.yaml bounds are "actual" training set bounds, which may be 
    # shorter than the requested bounds. 
    for key in ["batchnorm", "batchsize", "debug", "dropout", "epochs", "flash", "glm", "kfold", "layers", "learning_rate", "model", "neurons",
                "optimizer", "reg_penalty", "rptdist", "suite", "twin"]:
        assert getattr(ylargs, key) == getattr(
            args, key), f'this script {key} {getattr(args,key)} does not match yaml {key} {getattr(ylargs,key)}'

    return True



def get_argparser():
    parser = argparse.ArgumentParser(description = "train/test dense neural network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchnorm', action='store_true', help="use batch normalization")
    parser.add_argument('--batchsize', type=int, default=1024, help="nn training batch size") # tf default is 32
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--dropout", type=float, default=0., help='fraction of neurons to drop in each hidden layer (0-1)')
    parser.add_argument('--nfits', type=int, default=5, help="number of times to fit (train) model")
    parser.add_argument('--epochs', default=30, type=int, help="number of training epochs")
    parser.add_argument('--flash', type=int, default=10, help="GLM flash count threshold")
    parser.add_argument('--kfold', type=int, default=5, help="apply kfold cross validation to training set")
    parser.add_argument('--layers', default=2, type=int, help="number of hidden layers")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--model', type=str, choices=["HRRR","NSC1km","NSC3km-12sec","NSC15km"], default="HRRR", help="prediction model")
    parser.add_argument("--glm", action='store_true', help='Use GLM')
    parser.add_argument('--neurons', type=int, nargs="+", default=[16], help="number of neurons in each nn layer")
    parser.add_argument('--optimizer', type=str, choices=['adam','sgd'], default='adam', help="optimizer")
    parser.add_argument('--reg_penalty', type=float, default=0.01, help="L2 regularization factor")
    parser.add_argument('--rptdist', type=int, default=40, help="severe weather report max distance")
    parser.add_argument('--savedmodel', type=str, help="filename of machine learning model")
    parser.add_argument('--trainend', type=lambda s: pd.to_datetime(s), help="training set end")
    parser.add_argument('--trainstart', type=lambda s: pd.to_datetime(s), help="training set start")
    parser.add_argument('--testend', type=lambda s: pd.to_datetime(s), default="20220101T00", help="testing set end")
    parser.add_argument('--teststart', type=lambda s: pd.to_datetime(s), default="20201202T12", help="testing set start")
    parser.add_argument('--suite', type=str, default='default', help="name for suite of training features")
    parser.add_argument('--twin', type=int, default=2, help="time window in hours")
    return parser


def get_optimizer(s, learning_rate = 0.001, **kwargs):
    if s == 'adam':
        o = optimizers.Adam(learning_rate = learning_rate)
    elif s == 'sgd':
        #learning_rate = 0.001 # from sobash
        momentum = 0.99
        nesterov = True
        decay = 1e-4 # no place to specify in optimizers.SGD, v2 of tensorflow
        o = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
    return o


def rptdist2bool(df, args):
    rptdist = args.rptdist
    twin = args.twin
    logging.debug(f"report distance {rptdist}km  time window {twin}h")
    # get rid of columns that are associated with different time window (twin)
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
        df[rh] = (df[rh] >= 0) & (df[rh] < rptdist) # TODO: test speed with .loc[:,rh]. it seemed slower.
        rptcols.append(rh)

    # Any report
    any_rpt_col = f"any_rptdist_{twin}hr"
    hailwindtorn = [f"{r}_rptdist_{twin}hr" for r in ["hailone","wind","torn"]]
    df[any_rpt_col] = df[hailwindtorn].any(axis="columns")
    rptcols.append(any_rpt_col)

    # GLM?
    if args.glm:
        logging.debug(f"at least {args.flash} flashes")
        df["flashes"] = df["flashes"] >= args.flash
        rptcols.append("flashes")

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
            #glmtimes = [dt.datetime.strptime(os.path.basename(x), "%Y%m%d%H.glm.nc") for x in glmfiles] # why is this here?
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

    glm = glm.rename(dict(time_coverage_start="valid_time")) #, y="projection_y_coordinate", x="projection_x_coordinate")) # commented out Aug 10, 2022
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

def get_features(args, subset=None):
    if args.model == "HRRR":
        features = [ "CAPESHEAR", "CAPESHEAR-N3T1", "CAPESHEAR-N3T3", "CAPESHEAR-N3T5", "CAPESHEAR-N5T1", "CAPESHEAR-N5T3", "CAPESHEAR-N5T5", "CREF", "CREF-N3T1", "CREF-N3T3", "CREF-N3T5",
                    "CREF-N5T1", "CREF-N5T3", "CREF-N5T5", "GRPL_MAX", "GRPL_MAX-N3T1", "GRPL_MAX-N3T3", "GRPL_MAX-N3T5", "GRPL_MAX-N5T1", "GRPL_MAX-N5T3", "GRPL_MAX-N5T5", "HAIL_SFC",
                    "HAIL_SFC-N3T1", "HAIL_SFC-N3T3", "HAIL_SFC-N3T5", "HAIL_SFC-N5T1", "HAIL_SFC-N5T3", "HAIL_SFC-N5T5", "HGT0C", "HGT0C-N3T1", "HGT0C-N3T3", "HGT0C-N3T5", "HGT0C-N5T1",
                    "HGT0C-N5T3", "HGT0C-N5T5", "LR75", "LTG1", "LTG1-N3T1", "LTG1-N3T3", "LTG1-N3T5", "LTG1-N5T1", "LTG1-N5T3", "LTG1-N5T5", "LTG2", "LTG2-N3T1", "LTG2-N3T3", "LTG2-N3T5",
                    "LTG2-N5T1", "LTG2-N5T3", "LTG2-N5T5", "LTG3", "LTG3-N3T1", "LTG3-N3T3", "LTG3-N3T5", "LTG3-N5T1", "LTG3-N5T3", "LTG3-N5T5", "Local_Solar_Hour_cos", "Local_Solar_Hour_sin",
                    "MLCAPE", "MLCINH", "MLCINH-N3T1", "MLCINH-N3T3", "MLCINH-N3T5", "MLCINH-N5T1", "MLCINH-N5T3", "MLCINH-N5T5", "MUCAPE", "MUCAPE-N3T1", "MUCAPE-N3T3", "MUCAPE-N3T5", "MUCAPE-N5T1",
                    "MUCAPE-N5T3", "MUCAPE-N5T5", "PREC_ACC_NC", "PREC_ACC_NC-N3T1", "PREC_ACC_NC-N3T3", "PREC_ACC_NC-N3T5", "PREC_ACC_NC-N5T1", "PREC_ACC_NC-N5T3", "PREC_ACC_NC-N5T5", "PSFC",
                    "PSFC-N3T1", "PSFC-N3T3", "PSFC-N3T5", "PSFC-N5T1", "PSFC-N5T3", "PSFC-N5T5", "RVORT1", "RVORT1-N3T1", "RVORT1-N3T3", "RVORT1-N3T5", "RVORT1-N5T1", "RVORT1-N5T3", "RVORT1-N5T5",
                    "SBCAPE", "SBCAPE-N3T1", "SBCAPE-N3T3", "SBCAPE-N3T5", "SBCAPE-N5T1", "SBCAPE-N5T3", "SBCAPE-N5T5", "SBCINH", "SBCINH-N3T1", "SBCINH-N3T3", "SBCINH-N3T5", "SBCINH-N5T1",
                    "SBCINH-N5T3", "SBCINH-N5T5", "SBLCL", "SBLCL-N3T1", "SBLCL-N3T3", "SBLCL-N3T5", "SBLCL-N5T1", "SBLCL-N5T3", "SBLCL-N5T5", "SHR01", "SHR01-N3T1", "SHR01-N3T3", "SHR01-N3T5",
                    "SHR01-N5T1", "SHR01-N5T3", "SHR01-N5T5", "SHR06", "SHR06-N3T1", "SHR06-N3T3", "SHR06-N3T5", "SHR06-N5T1", "SHR06-N5T3", "SHR06-N5T5", "SRH01", "SRH01-N3T1", "SRH01-N3T3",
                    "SRH01-N3T5", "SRH01-N5T1", "SRH01-N5T3", "SRH01-N5T5", "SRH03", "SRH03-N3T1", "SRH03-N3T3", "SRH03-N3T5", "SRH03-N5T1", "SRH03-N5T3", "SRH03-N5T5", "STP", "STP-N3T1",
                    "STP-N3T3", "STP-N3T5", "STP-N5T1", "STP-N5T3", "STP-N5T5", "T2", "T2-N3T1", "T2-N3T3", "T2-N3T5", "T2-N5T1", "T2-N5T3", "T2-N5T5", "T500", "T700", "T850", "T925",
                    "TD2", "TD2-N3T1", "TD2-N3T3", "TD2-N3T5", "TD2-N5T1", "TD2-N5T3", "TD2-N5T5", "TD500", "TD700", "TD850", "TD925", "U500", "U700", "U850", "U925", "UP_HELI_MAX",
                    "UP_HELI_MAX-N1T5", "UP_HELI_MAX-N3T1", "UP_HELI_MAX-N3T3", "UP_HELI_MAX-N3T5", "UP_HELI_MAX-N5T1", "UP_HELI_MAX-N5T3", "UP_HELI_MAX-N5T5", "UP_HELI_MAX02", "UP_HELI_MAX02-N3T1",
                    "UP_HELI_MAX02-N3T3", "UP_HELI_MAX02-N3T5", "UP_HELI_MAX02-N5T1", "UP_HELI_MAX02-N5T3", "UP_HELI_MAX02-N5T5", "UP_HELI_MAX03", "UP_HELI_MAX03-N3T1", "UP_HELI_MAX03-N3T3",
                    "UP_HELI_MAX03-N3T5", "UP_HELI_MAX03-N5T1", "UP_HELI_MAX03-N5T3", "UP_HELI_MAX03-N5T5", "UP_HELI_MAX120", "UP_HELI_MAX120-N1T5", "UP_HELI_MAX80", "UP_HELI_MAX80-N1T5",
                    "UP_HELI_MIN", "UP_HELI_MIN-N3T1", "UP_HELI_MIN-N3T3", "UP_HELI_MIN-N3T5", "UP_HELI_MIN-N5T1", "UP_HELI_MIN-N5T3", "UP_HELI_MIN-N5T5", "V500", "V700", "V850", "V925",
                    "WSPD10MAX", "WSPD10MAX-N3T1", "WSPD10MAX-N3T3", "WSPD10MAX-N3T5", "WSPD10MAX-N5T1", "WSPD10MAX-N5T3", "WSPD10MAX-N5T5", "W_DN_MAX", "W_DN_MAX-N3T1", "W_DN_MAX-N3T3",
                    "W_DN_MAX-N3T5", "W_DN_MAX-N5T1", "W_DN_MAX-N5T3", "W_DN_MAX-N5T5", "W_UP_MAX", "W_UP_MAX-N3T1", "W_UP_MAX-N3T3", "W_UP_MAX-N3T5", "W_UP_MAX-N5T1", "W_UP_MAX-N5T3",
                    "W_UP_MAX-N5T5", "dayofyear_cos", "dayofyear_sin", "forecast_hour", "lat", "lon"]

    elif args.model.startswith("NSC"):
        # Had static list before Nov 24, 2022. But these 33 predictors were missing from the default suite. They weren't even my first commit to github.
        # Why did they disappear? Maybe when I accidentally deleted my work directory in Oct 2021.
        # {'LR75-N5T5', 'REFL_COM-N5T3', 'LR75-N5T1', 'HAILCAST_DIAM_MAX-N5T1', 'HAILCAST_DIAM_MAX-N3T3', 'UP_HELI_MIN-N5T3', 'UP_HELI_MIN-N5T5',
        # 'HAILCAST_DIAM_MAX-N3T1', 'UP_HELI_MIN', 'REFL_COM-N5T1', 'REFL_COM-N5T5', 'MLCINH-N3T1', 'REFL_COM-N3T1', 'REFL_COM-N3T3', 'HAILCAST_DIAM_MAX', 'REFL_COM', 'HAILCAST_DIAM_MAX-N5T3', 
        # 'REFL_COM-N3T5', 'UP_HELI_MIN-N5T1', 'MLCINH-N5T5', 'MLCINH-N3T5', 'UP_HELI_MIN-N3T5', 'MLCINH-N5T3', 'LR75-N3T1', 'HAILCAST_DIAM_MAX-N5T5', 
        # 'HAILCAST_DIAM_MAX-N3T5', 'UP_HELI_MIN-N3T3', 'MLCINH-N5T1', 'LR75-N5T3', 'LR75-N3T3', 'UP_HELI_MIN-N3T1', 'LR75-N3T5', 'MLCINH-N3T3'}

        features = open(f"suite/{args.model}.{args.suite}.txt","r").read().splitlines() # Hopefully these somewhat alphabetically-sorted lists are easier to spot mistakes in.
       
    else:
        logging.error("Can't get feature suite for unexpected model {model}")
        
    if len(set(features)) != len(features):
        logging.warning(f"repeated feature(s) {set([x for x in features if features.count(x) > 1])}")
        features = list(set(features))

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


def savedmodel_default(args, fhr_str=None):
    # Could be 'adam' or SGD from Sobash 2020
    optimizer = get_optimizer(args.optimizer)
    
    if args.batchnorm:
        batchnorm_str = ".bn"
    else:
        batchnorm_str = ""

    glmstr = "" # GLM description 
    if args.glm: glmstr = f"{args.flash}flash_{args.twin}hr." # flash rate threshold and GLM time window
        
    savedmodel  = f"{args.model}.{args.suite}.{glmstr}rpt_{args.rptdist}km_{args.twin}hr.{args.neurons[0]}n.ep{args.epochs}.{fhr_str}."
    savedmodel += f"bs{args.batchsize}.{args.layers}layer.{optimizer._name}.L2{args.reg_penalty}.lr{args.learning_rate}.dr{args.dropout}{batchnorm_str}"
        
    return savedmodel
