import argparse
import datetime as dt
import glob
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import numpy as np
import hwtmode.statisticplot
from hwtmode.data import decompose_circular_feature

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
    # Warn if trimmed training period and requested test periods overlap.
    trainstart = getattr(ylargs,"trainstart")
    trainend   = getattr(ylargs,"trainend")
    teststart = args.teststart
    testend   = args.testend
    overlap = min([trainend, testend]) - max([trainstart, teststart])
    if overlap >= dt.timedelta(hours=0) and args.kfold == 1:
        logging.warning(f"training and testing time ranges overlap {trainstart}-{trainend}|{teststart}-{testend}")
   
    # Comparing config.yaml training period and requested training period (args) is not a good test because config.yaml bounds are "trimmed"
    # to actual training data. config.yaml training period may be subset of the requested training period in args.
    # If any of the actual "trimmed" training period is outside the requested training period, this is a problem.
    # args.trainstart <= trainstart            trainend <= args.trainend
    assert args.trainstart <= trainstart, f"Requested training period {args.trainstart}-{args.trainend} starts after actual 'trimmed' training period {trainstart}-{trainend} starts"
    assert trainend <= args.trainend,     f"Requested training period {args.trainstart}-{args.trainend} ends before actual 'trimmed' training period {trainstart}-{trainend} ends"
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
    parser.add_argument('--ifile', type=str, help="Read this parquet input file. Otherwise guess which one to read.")
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


def full_cmd(args):
    """
    Given a argparse Namespace, return complete argument string suitable for shell command line.
    Format datetimes as strings.
    Just print keyword if its value is Boolean and True.
    Skip keyword and value if its value is Boolean and False.
    Skip keyword and value if value is None.
    Remove brackets from lists.
    """
    s = " ".join(args._get_args())
    for kw,value in args._get_kwargs():
        if isinstance(value, dt.datetime):
            value = value.strftime("%Y%m%dT%H%M")
        if isinstance(value, bool):
            if not value:
                continue
            value = ""
        if isinstance(value, list):
            value = " ".join([str(i) for i in value])
        if value is None:
            continue
        s += f" --{kw} {value}"
    return s + "\n"

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


def grab_predictors(args, idate, idir = '/glade/work/sobash/NSC_objects'):
    model = args.model
    twin = args.twin
    rptdist = args.rptdist
    time_space_windows = [(twin,rptdist)]
    # Define ifiles, a list of input files from glob.glob method
    if model == "HRRR":
        if idate < pd.to_datetime("20191002"):
            logging.error(f"No {model} before 20191002")
            sys.exit(1)
        if idate < pd.to_datetime("20201202"):
            ifile = f'{idir}/HRRR_new/grid_data/grid_data_HRRRX_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
        else:
            ifile = f'{idir}/HRRR_new/grid_data/grid_data_HRRR_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
    elif model.startswith("NSC"):
        ifile = f'/glade/work/sobash/NSC_objects/grid_data_new/grid_data_{model}_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
        # remove largest neighborhood size (fields containing N7 in the name)
    df = pd.read_parquet(ifile, engine="pyarrow")

    # Index df and modeds the same way. 
    logging.info(f"convert df Date to datetime64[ns]")
    df["Date"] = df.Date.astype('datetime64[ns]')
    df = df.rename(columns=dict(yind="y",xind="x",Date="initialization_time",fhr="forecast_hour"))
    logging.info(f"derive valid_time from initialization_time + forecast_hour")
    df["valid_time"] = pd.to_datetime(df["initialization_time"]) + df["forecast_hour"].astype(int) * dt.timedelta(hours=1)
    df = df.set_index(["y","x","initialization_time","forecast_hour"])

    # Derived fields
    df["dayofyear"] = df["valid_time"].dt.dayofyear
    df["Local_Solar_Hour"] = df["valid_time"].dt.hour + df["lon"]/15
    df = decompose_circular_feature(df, "dayofyear", period=365.25)
    df = decompose_circular_feature(df, "Local_Solar_Hour", period=24)
    df = df.reset_index().set_index(["valid_time","y","x"])

    if args.glm:
        earliest_valid_time = df.index.get_level_values(level="valid_time").min()
        latest_valid_time = df.index.get_level_values(level="valid_time").max()
        assert latest_valid_time > pd.to_datetime("20160101"), "DataFrame completely before GLM exists"
        glmds = get_glm(time_space_windows)
        glmds = glmds.sel(valid_time = slice(earliest_valid_time,latest_valid_time)) # Trim GLM to time window of model data
        logging.info(f"Merge flashes with {model} DataFrame")
        # In glmds, x : west to east, y : south to north
        # In sobash df, xind : south to north, yind : west to east.
        glmds = glmds.rename(dict(x="y",y="x")) #dimensions are renamed to match sobash df.
        df = df.merge(glmds.to_dataframe(), left_on=["valid_time","y","x"], right_on=["valid_time","y","x"])
        #Do {model} and GLM overlap at all?"
        assert not df.empty, f"Merged {model}/GLM Dataset is empty."
        # Sanity check--make sure prediction model and GLM grid box lat lons are similar
        assert (df.lon_y - df.lon_x).max() < 0.1, f"{model} and glm longitudes don't match"
        assert (df.lat_y - df.lat_x).max() < 0.1, f"{model} and glm lats don't match"
        df = df.drop(columns=["lon_y","lat_y"])
        df = df.rename(columns=dict(lon_x="lon",lat_x="lat")) # helpful for scale factor pickle file.

    return df



def rptdist2bool(df, args):
    """
    Convert distance to closest storm report to True/False based on distance and time thresholds 
    And convert flash count to True/False based on distance, time, and flash threshold. 
    """
    
    rptdist = args.rptdist
    twin = args.twin
    logging.debug(f"report distance {rptdist}km  time window {twin}h")
    # get rid of storm report distance columns that are associated with different time window (twin)
    dropcol=[]
    for r in ["sighail", "sigwind", "hailone", "wind", "torn"]:
        for h in [0,1,2]:
            if h != twin:
                dropcol.append(f"{r}_rptdist_{h}hr")
    df = df.drop(columns=dropcol)

    rptcols = []
    for r in ["sighail", "sigwind", "hailone", "wind", "torn"]:
        rh = f"{r}_rptdist_{twin}hr"
        # Convert severe report distance to boolean (0-rptdist = True)
        df[rh] = (df[rh] >= 0) & (df[rh] < rptdist) # TODO: test speed with .loc[:,rh]. it seemed slower.
        # new Boolean column name with numeric dist threshold instead of numberless "rptdist".
        newcol = f"{r}_{rptdist}km_{twin}hr"
        df = df.rename(columns={rh:newcol}, errors="raise")
        rptcols.append(newcol)

    # Any report
    any_rpt_col = f"any_{rptdist}km_{twin}hr"
    hailwindtorn = [f"{r}_{rptdist}km_{twin}hr" for r in ["hailone","wind","torn"]]
    df[any_rpt_col] = df[hailwindtorn].any(axis="columns")
    rptcols.append(any_rpt_col)

    # GLM?
    if args.glm:
        # Check if flash threshold is met in one space/time window.
        # new flash variable name has space and time window in it.
        flash_spacetime_win = f"flashes_{rptdist}km_{twin}hr"
        logging.debug(f"at least {args.flash} flashes in {flash_spacetime_win}")
        df[flash_spacetime_win] = df[flash_spacetime_win] >= args.flash
        rptcols.append(flash_spacetime_win)
        # Drop all the other "flash_" columns of various space/time windows.
        dropcol = [x for x in df.columns if x.startswith("flash_") and x != flash_spacetime_win]
        df = df.drop(columns=dropcol)

    return df, rptcols

def get_glm(time_space_windows, date=None):
    # Initialize Dataset to hold GLM flash count for all space/time windows.
    ds = xarray.Dataset()
    for twin, rptdist in time_space_windows:
        suffix = ".glm.nc"
        if rptdist == 20:
            suffix = ".glm.40km.nc"
        if date:
            logging.info(f"date={date}")
            glmfiles = sorted(glob.glob(f"/glade/work/ahijevyc/GLM/{date.strftime('%Y%m%d')}*{suffix}"))
            glm = xarray.open_mfdataset(glmfiles, concat_dim="time_coverage_start", combine="nested")
        else:
            oneGLMfile = True
            if oneGLMfile:
                ifile = f"/glade/scratch/ahijevyc/temp/all{suffix}"
                if os.path.exists(ifile):
                    glm = xarray.open_dataset(ifile)
                else:
                    logging.error(f"{ifile} does not exist. run ncrcat on GLM files")
                    logging.error(f"cd /glade/work/ahijevyc/GLM")
                    logging.error(f"ls 20??/*{suffix} > filelist")
                    logging.error(f"cat filelist|ncrcat --fl_lst_in {ifile}")
                    sys.exit(1)
            else:
                glmfiles = sorted(glob.glob("/glade/work/ahijevyc/GLM/2*{suffix}"))
                logging.info("open_mfdataset")
                glm = xarray.open_mfdataset(glmfiles, concat_dim="time_coverage_start", combine="nested")

        assert (glm.time_coverage_start[1] - glm.time_coverage_start[0]) == np.timedelta64(3600,'s'), 'glm.time_coverage_start interval not 1h'

        if rptdist < 40:
            logging.error("TODO: remake GLM with {rptdist}km grid")
            sys.exit(1)
        newcol = f"flashes_{rptdist}km_{twin}hr"
        logging.info(f"Sum flashes from -{twin} hours to +{twin-1} hour(s) to make {2*twin}-hour time-centered window. Store in new DataArray {newcol}")
        glmsum = xarray.zeros_like(glm)
        for time_shift in range(-twin, twin):
            glmsum += glm.shift(time_coverage_start=-time_shift)

        if rptdist > 40:
            k = int(rptdist/40)
            logging.warning(f"{rptdist}km glm distance threshold is not really available. Spatial window overlaps masked points")
            logging.warning(f"TODO: save GLM in non-masked form, or filter it from 40km to 120km while unmasked and making the GLM files.")
            logging.info(f"Despite reservations, continuing to sum GLM flash counts in rolling {k}x{k} window")
            glmsum = glmsum.rolling(x=k, y=k, center=True).sum()

        glmsum = glmsum.rename(dict(time_coverage_start="valid_time",flashes=newcol))
        ds = ds.assign(variables=glmsum)
    return ds


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

        features = open(f"suite_predictors/{args.model}.{args.suite}.txt","r").read().splitlines() # Hopefully these somewhat alphabetically-sorted lists are easier to spot mistakes in.
       
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


def savedmodel_default(args, fhr_str=None, odir="nn"):
    # optimizer could be 'adam' or SGD from Sobash 2020
    optimizer = get_optimizer(args.optimizer)
    
    if args.batchnorm:
        batchnorm_str = ".bn"
    else:
        batchnorm_str = ""

    glmstr = "" # GLM description 
    if args.glm: glmstr = f"{args.flash}flash." # flash rate threshold and GLM time window
        
    savedmodel  = f"{odir}/nn_{args.model}.{args.suite}.{glmstr}rpt_{args.rptdist}km_{args.twin}hr.{args.neurons[0]}n.ep{args.epochs}.{fhr_str}."
    savedmodel += f"bs{args.batchsize}.{args.layers}layer.{optimizer._name}.L2{args.reg_penalty}.lr{args.learning_rate}.dr{args.dropout}{batchnorm_str}"
        
    return savedmodel
