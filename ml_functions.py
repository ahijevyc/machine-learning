import argparse
import dask.dataframe as dd
import datetime
import glob
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import pickle
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
import time
import os
import xarray

def assertclose(df, *c, lsuffix="", rsuffix="", atol=0.1):
    # use groupby(["x","y"]).mean() here instead of comparing entire df.lon series cause
    # there could be nans
    # Don't compare valid_times if eligible forecast hours were different like storm mode (12-35)
    for column in c:
        lc, rc = column+lsuffix, column+rsuffix
        xymean = df[[lc,rc]].groupby(["y", "x"]).mean(numeric_only=False)
        # Handle times and numbers differently
        if xymean[lc].dtype == '<M8[ns]' and xymean[rc].dtype == '<M8[ns]':
            assert xymean[column+lsuffix].equals(xymean[rc]), (
                f"{lc} and {rc} are not equal {xymean[[lc,rc]]}")
        else:
            logging.debug(f"are {lc} and {rc} close?")
            assert np.allclose(xymean[lc], xymean[rc], atol=atol), (
                f"{lc} and {rc} are not close {xymean[[lc,rc]]}")

# write and read safe yaml. write argparse.Namespace as dictionary and 
# timestamps as strings
# experiment with this in ~ahijevyc/yaml_config.ipynb
from yaml import CSafeDumper
from yaml.representer import SafeRepresenter

class Dumper(CSafeDumper):
    pass

def timestamp_representer(dumper, data):
    return SafeRepresenter.represent_datetime(dumper, pd.Timestamp(data))

def namespace_representer(dumper, data):
    return SafeRepresenter.represent_dict(dumper, data.__dict__)

Dumper.add_representer(pd.Timestamp, timestamp_representer)
Dumper.add_representer(argparse.Namespace, namespace_representer)




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
        bs = np.mean((preds - obs) ** 2, axis=0)
        obs_climo = np.mean(obs, axis=0) # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
        bs_climo = np.mean((obs - obs_climo) ** 2, axis=0)
        bss = 1 - bs/bs_climo

    return bss


def configs_match(ylargs, args):
    # Warn if trimmed training period and requested test periods overlap.
    trainstart = getattr(ylargs,"trainstart")
    trainend   = getattr(ylargs,"trainend")
    teststart = args.teststart
    testend   = args.testend
    overlap = min([trainend, testend]) - max([trainstart, teststart])
    if overlap >= datetime.timedelta(hours=0) and args.kfold == 1:
        logging.warning(f"training and testing time ranges overlap [{trainstart},{trainend}] [{teststart},{testend}]")
   
    # Comparing yaml config training period and requested training period (args) is not a good test 
    # because yaml config bounds are trimmed to actual range of training cases.
    # config training period may be subset of the requested training period in args, but if any of the actual "trimmed" training period
    # beyond the range of the requested training period, this is a problem.
    # args.trainstart <= trainstart            trainend <= args.trainend
    assert args.trainstart <= trainstart, (f"requested start of training period {args.trainstart} is after actual training period [{trainstart},{trainend}]")
    assert trainend <= args.trainend,     (f"requested end of training period {args.trainend} is before actual training period [{trainstart},{trainend}]")
    for key in ["batchnorm", "batchsize", "dropout", "epochs", "flash", "fhr", "glm", "kfold", "learning_rate", "model", "neurons",
                "optimizer", "reg_penalty", "rptdist", "suite", "twin"]:
        assert getattr(ylargs, key) == getattr(
            args, key), f'requested {key} {getattr(args,key)} does not match savedmodel yaml {key} {getattr(ylargs,key)}'

    return True



def get_argparser():
    parser = argparse.ArgumentParser(description = "train/test dense neural network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchnorm', action='store_true', help="use batch normalization")
    parser.add_argument('--batchsize', type=int, default=1024, help="nn training batch size") # tf default is 32
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--dropout", type=float, default=0., help='fraction of neurons to drop in each hidden layer (0-1)')
    parser.add_argument('--epochs', default=30, type=int, help="number of training epochs")
    parser.add_argument('--event', default=None, choices=["cg", "ic", "flash", "sighail", "sigwind", "hailone", "wind", "torn"], help="train for this event only")
    parser.add_argument('--fhr', nargs="+", type=int, default=list(range(1,49)), help="train with these forecast hours. Testing scripts only use this list to verify correct model "
                                                                                      "for testing; no filter applied to testing data. In other words you "
                                                                                      "test on all forecast hours in the testing data, regardless of whether the model was "
                                                                                      "trained with the same forecast hours.")
    parser.add_argument('--fits', nargs="+", type=int, default=None, help="work on specific fit(s) so you can run many in parallel")
    parser.add_argument('--flash', type=int, default=10, help="GLM flash count threshold")
    parser.add_argument('--folds', nargs="+", type=int, default=None, help="work on specific fold(s) so you can run many in parallel")
    parser.add_argument("--glm", action='store_true', help='Use GLM')
    parser.add_argument('--kfold', type=int, default=5, help="apply kfold cross validation to training set")
    parser.add_argument('--idate', type=lambda s: pd.to_datetime(s), help="single initialization time")
    parser.add_argument('--ifile', type=str, help="Read this parquet input file. Otherwise guess which one to read.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--model', type=str, choices=["HRRR","NSC1km","NSC3km-12sec","NSC15km"], default="HRRR", help="prediction model")
    parser.add_argument('--neurons', type=int, nargs="+", default=[16,16], help="number of neurons in each nn layer")
    parser.add_argument('--nfits', type=int, default=5, help="number of times to fit (train) model")
    parser.add_argument('--nprocs', type=int, default=0, help="verify this many forecast hours in parallel")
    parser.add_argument('--optimizer', type=str, choices=['Adam','SGD'], default='Adam', help="optimizer")
    parser.add_argument('--reg_penalty', type=float, default=0.01, help="L2 regularization factor")
    parser.add_argument('--rptdist', type=int, default=40, help="severe weather report max distance")
    parser.add_argument('--savedmodel', type=str, help="filename of machine learning model")
    parser.add_argument('--seed', type=int, default=None, help="random number seed for reproducability")
    parser.add_argument('--trainend', type=lambda s: pd.to_datetime(s), help="training set end")
    parser.add_argument('--trainstart', type=lambda s: pd.to_datetime(s), help="training set start")
    parser.add_argument('--testend', type=lambda s: pd.to_datetime(s), default="20220101T00", help="testing set end")
    parser.add_argument('--teststart', type=lambda s: pd.to_datetime(s), default="20201202T12", help="testing set start")
    parser.add_argument('--suite', type=str, default='default', help="name for suite of training features")
    parser.add_argument('--twin', nargs="+", type=int, default=[1,2,4], help="time window(s) in hours")

    return parser


def full_cmd(args):
    """
    Given a argparse Namespace, reverse engineer the string of arguments that would create it. 
    String is suitable for shell command line.
    Format datetimes as strings.
    Just print keyword if its value is Boolean and True.
    Skip keyword and value if its type is Boolean and value is False.
    Skip keyword and value if value is None.
    Remove brackets from lists.
    """
    s = " ".join(args._get_args())
    for kw,value in args._get_kwargs():
        if isinstance(value, datetime.datetime):
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
    if s == 'Adam':
        o = optimizers.Adam(learning_rate = learning_rate)
    elif s == 'SGD':
        #learning_rate = 0.001 # from sobash
        momentum = 0.99
        nesterov = True
        o = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, **kwargs)
    return o


def load_df(args, idir="/glade/work/sobash/NSC_objects", wbugdir="/glade/campaign/mmm/parc/ahijevyc/wbug_lightning"):
    debug = args.debug
    idate = args.idate
    ifile = args.ifile
    model = args.model
    rptdist = args.rptdist
    twin = args.twin

    # Define input filename.
    if ifile is None:
        if model == "HRRR":
            ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.par'
            ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRXHRRR.par'
            if debug:
                ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.fastdebug.par'
        elif model.startswith("NSC"):
            ifile = f'{model}.par'
            if debug:
                ifile = f'/glade/work/ahijevyc/NSC_objects/{model}_debug.par'
        if idate:
            ifile = os.path.join(
                os.getenv('TMPDIR'), f"{model}.{idate.strftime('%Y%m%d%H-%M%S')}.par")

    logging.info(
        f"load {model} predictors from parquet file {ifile}")
    if os.path.exists(ifile):
        logging.info(f'reading {ifile} {os.path.getsize(ifile)/1024**3:.1f}G')
        df = pd.read_parquet(ifile, engine="pyarrow")
        return df

    logging.info(f"{ifile} doesn't exist, so create it.")
    # Define ifiles, a list of input files from glob.glob method
    if model == "HRRR":
        # HRRRX = experimental HRRR (v4)
        search_str = f'{idir}/HRRR_new/grid_data/grid_data_HRRRX_d01_20*00-0000.par'
        if debug:
            search_str = search_str.replace("*", "2006*")  # just June 2020
        else:
            # append HRRR to HRRRX.
            # HRRR prior to Dec 3 2020 is v3 and HRRR at Dec 3 2020 and afterwards is v4.
            # Dec 3-9, 2020 00z only
            search_str += f' {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_2020120[3-9]*00-0000.par'
            # Dec 10-31, 2020 00z only
            search_str += f' {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_202012[1-3]*00-0000.par'
            # 2021+ 00z only
            search_str += f' {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_202[1-9]*00-0000.par'
        logging.info(f"ifiles search string {search_str}")
        ifiles = []
        for x in search_str.split(" "):
            ifiles.extend(glob.glob(x))
    elif model.startswith("NSC"):
        search_str = f'{idir}/grid_data_new/grid_data_{model}_d01_20*00-0000.par'
        if debug:
            debug_replace = "201[0]*"
            search_str = search_str.replace("20*", debug_replace)
        logging.info(f"ifiles search string {search_str}")
        ifiles = glob.glob(search_str)

    if idate:
        if model == "HRRR":
            assert idate >= pd.to_datetime("20191002"), f"No {model} before 20191002"
            d = f'{idir}/HRRR_new/grid_data/grid_data_HRRRX_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
            if idate >= pd.to_datetime("20201202"):
                d = d.replace("HRRRX","HRRR")
        elif model.startswith("NSC"):
            d = f'{idir}/grid_data_new/grid_data_{model}_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
        else:
            logging.error(f"unexpected model {model} with idate {idate}")
            sys.exit(1)
        ifiles = [d]

    logging.info(f"Reading {len(ifiles)} {model} files")
    # pd.read_parquet only handles one file at a time, so pd.concat
    df = pd.concat(pd.read_parquet(f, engine="pyarrow")
                   for f in ifiles)

    logging.info(f"read {len(df)} rows")

    # Index df and modeds the same way.
    df = df.rename(columns=dict(yind="y", xind="x",
                   Date="initialization_time", fhr="forecast_hour"), copy=False)
    logging.info(
        f"derive valid_time from initialization_time + forecast_hour")
    df["valid_time"] = pd.to_datetime(
        df["initialization_time"]) + pd.to_timedelta(df["forecast_hour"], unit="hours")
    df = df.set_index(["y", "x", "initialization_time", "forecast_hour"])

    if model.startswith("NSC3km"):
        logging.info("model starts with 'NSC3km' so read mode probabilities")
        use_hourly_mode_files = False
        if use_hourly_mode_files:
            search_str = f'/glade/scratch/cbecker/NCAR700_objects/output_object_based/evaluation_zero_filled/20*/label_probabilities_20*00_fh_*.nc'
            ifiles = sorted(glob.glob(search_str))
            logging.info(
                f"Read {len(ifiles)} storm mode files in date range of {model} DataFrame")
            # reset_coords to avoid xarray.core.merge.MergeError: unable to determine if 
            # these variables should be coordinates or not in the merged result: {'valid_time'}
            # set_index to avoid ValueError: Could not find any dimension coordinates to use to order the datasets for concatenation
            modeds = xarray.open_mfdataset(ifiles, 
                    preprocess=lambda x: x.reset_coords(["lon", "lat", 'valid_time']).set_index(time=['init_time', 'forecast_hour']),
                    combine="nested", parallel=True, compat="override", combine_attrs="override")  # parallel is faster
        else:
            # Try nco concat files from ~/bin/modeprob_concat.csh. Faster than reading individual forecast hour files.
            search_str = f'/glade/scratch/ahijevyc/NCAR700_objects/output_object_based/evaluation_zero_filled/20*00.nc'
            if debug:
                search_str = search_str.replace('20*', debug_replace)
            logging.info(f"search_str {search_str}")
            ifiles = sorted(glob.glob(search_str))
            logging.info(
                f"Open and combine {len(ifiles)} storm mode probability files")
            modeds = xarray.open_mfdataset(ifiles)
        logging.info(
            f"put usamask pickle file into xarray DataArray")
        # mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
        mask = pickle.load(open('./usamask.pk', 'rb'))
        height, width = 65, 93
        mask = mask.reshape((height, width))
        mask = xarray.DataArray(mask, dims=["y", "x"])
        logging.info(f"Make x and y indices actual coordinates of mask")
        mask = mask.assign_coords(dict(x=mask.x, y=mask.y))

        modeds = modeds.assign_coords(dict(x=modeds.x, y=modeds.y))
        logging.info(f"apply mask and trim coordinates")
        # even after trimming coordinates with drop=True you still have nans. mask is not a box. It has irregular disjointed edges.
        modeds = modeds.where(mask, drop=True)

        # In modeds, x : west to east, y : south to north
        # In sobash df, xind : south to north, yind : west to east.
        logging.info(
            f"Rename mode prob dimensions to match index names of df {df.index.names}")
        modeds = modeds.rename(dict(x="y", y="x"))
        logging.info(f"mode prob dimensions now {modeds.dims}")

        logging.info(f"convert mode DataArray to DataFrame")
        modedf = modeds.to_dataframe()
        logging.info(
            f"Drop all-NA rows from {len(modedf)} row storm mode DataFrame")
        modedf = modedf.dropna(how="all")
        # modeprobs are na for some forecast hours. So expect fewer rows than df
        logging.info(f"{len(modedf)} remaining")
        logging.info(f"replace 'time' with 'forecast_hour' in index")
        modedf = modedf.set_index(
            "forecast_hour", append=True).droplevel("time")
        logging.info(
            f"join {model} DataFrame with storm mode DataFrame")
        df = df.join(modedf, rsuffix="_mode")
        logging.info(
            f"joined DataFrame has {len(df)} rows")
        assertclose(df, "lat", "lon", rsuffix="_mode")
        df = df.drop(columns=["lat_mode", "lon_mode", "valid_time_mode"])

    # Derived fields
    df["dayofyear"] = df["valid_time"].dt.dayofyear
    df["Local_Solar_Hour"] = df["valid_time"].dt.hour + df["lon"]/15
    df = decompose_circular_feature(df, "dayofyear", period=365.25)
    df = decompose_circular_feature(df, "Local_Solar_Hour", period=24)
    logging.info(f"make valid_time an index so wbug and glm can be joined on valid_time")
    df = df.reset_index().set_index(["valid_time", "y", "x"])

    earliest_valid_time = df.index.get_level_values(
        level="valid_time").min()
    latest_valid_time = df.index.get_level_values(
        level="valid_time").max()

    # Merge weatherbug lightning data
    logging.info("load wbug lightning data")
    wbug = xarray.open_dataset(os.path.join(wbugdir, f"flash_{rptdist}km.nc"))
    # In wbug, x : west to east, y : south to north
    # In sobash df, xind : south to north, yind : west to east.
    # dimensions are renamed to match sobash df.
    #     df     | wbug
    # ==================
    # valid_time | time_coverage_start
    #     y      |  x
    #     x      |  y
    wbug = wbug.rename(dict(x="y",y="x",time_coverage_start="valid_time"))
    wbug = wbug.sel(valid_time=slice(earliest_valid_time, latest_valid_time))
    
    for t in twin:
        if wbug.valid_time.size:
            # weatherbug CG and IC lightning
            # Sum counts in 30-minute blocks
            # Investigate or debug with ~ahijevyc/wbug_sum_time_window.ipynb
            logging.info(f"Sum weatherbug flashes in +/-{t}hr time window")
            ltg_sum = wbug.rolling(valid_time=t*2, center=True).sum()
        else:
            ltg_sum = xarray.full_like(wbug, np.nan)
            logging.warning(f"wbug Dataset has 0 valid_times")            
        name_dict = {l: f"{l}_{rptdist}km_{t}hr" for l in ["cg","ic"]}
        logging.info(f"rename {name_dict}")
        ltg_sum = ltg_sum.rename(name_dict)
        wbug = wbug.assign(ltg_sum)
        
    logging.info(f"join {wbug.valid_time.size} wbug lightning times")
    df = df.join(wbug.to_dataframe(), rsuffix="_y")
    logging.info("done joining wbug lightning data")
    if wbug.valid_time.size:
        # Sanity check--make sure lat lons are similar
        assertclose(df, "lat", "lon", rsuffix="_y")
    df = df.drop(columns=["lon_y", "lat_y", "cg", "ic"])

    # join Geostationary Lightning Mapper (GLM)
    firstglm = pd.to_datetime("20160101")
    if latest_valid_time > firstglm:
        time_space_windows = [(1, rptdist), (2, rptdist), (4, rptdist)]
        glmds = get_glm(time_space_windows,
                        start=earliest_valid_time, end=latest_valid_time)
        # In glmds, x : west to east, y : south to north
        # In sobash df, xind : south to north, yind : west to east.
        # dimensions are renamed to match sobash df.
        glmds = glmds.rename(dict(x="y", y="x"))
        logging.info(f"join glm flashes with {model} DataFrame")
        df = df.join(glmds.to_dataframe(), rsuffix="_y")
        # Do {model} and GLM overlap at all?"
        if df.empty:
            logging.warning(f"joined {model}/GLM DataFrame is empty.")
        logging.debug(
            f"Sanity check--make sure {model} and GLM lat lons are close")
        assertclose(df, "lat", "lon", rsuffix="_y")
        df = df.drop(columns=["lon_y", "lat_y"])
    else:
        logging.warning(f"all {model} data before first GLM {firstglm}")

    logging.info("convert 64-bit to 32-bit columns")
    dtype_dict = {
        k: np.float32 for k in df.select_dtypes(np.float64).columns}
    dtype_dict.update(
        {k: np.int32 for k in df.select_dtypes(np.int64).columns})
    df = df.astype(dtype_dict, copy=False)

    logging.info(f"writing {ifile}")
    df.to_parquet(ifile)
    return df


def rptdist2bool(df, args):
    """
    Return DataFrame with storm report distances and flash counts converted to Boolean.
    These columns have new names that include distance and time window. 
    Derive "any" severe storm report and "cg.ic" labels.
    Also return a list of column names to be used as labels.
    """

    event = args.event
    rptdist = args.rptdist
    twin = args.twin
    logging.debug(f"report distance {rptdist}km {twin}h")

    lsrtypes = ["sighail", "sigwind", "hailone", "wind", "torn"]
    oldtwin = [0,1,2]
    logging.warning(f"using {oldtwin} time windows for lsrtypes until Ryan updates them with [1,2,4]") 
    label_cols = [f"{r}_rptdist_{t}hr" for r in lsrtypes for t in oldtwin]
    # refer to new label names (with f"{rptdist}km" not f"rptdist")
    renamecolumns = {r: r.replace("rptdist", f"{rptdist}km")
                     for r in label_cols}
    logging.info(f'rename columns {renamecolumns}')
    df = df.rename(columns=renamecolumns, copy=False, errors="raise")
    # Replace old label_cols list with list of new names
    label_cols = list(renamecolumns.values())
    logging.info(
        f"Convert severe report distance to boolean [0,{rptdist}km) = True")
    # faster than df[label_cols].loc[:,label_cols]
    df[label_cols] = (0 <= df[label_cols]) & (df[label_cols] < rptdist)


    
    for t in oldtwin:
        # any type of severe storm report
        any_label_str = f"any_{rptdist}km_{t}hr"
        labels_this_twin = [f"{r}_{rptdist}km_{t}hr" for r in lsrtypes]
        logging.debug(f"derive {any_label_str} from {labels_this_twin}")
        df[any_label_str] = df[labels_this_twin].any(axis="columns")
        label_cols.append(any_label_str)

    for t in twin:

        # weatherbug flashes
        logging.info(f"threshold wbug at {args.flash} flashes")
        wbug_cols = [f"{f}_{rptdist}km_{t}hr" for f in ["cg","ic"]]
        df[wbug_cols] = df[wbug_cols] >= args.flash
        either = f"cg.ic_{rptdist}km_{t}hr"
        logging.info(f"{' or '.join(wbug_cols)} = {either}")
        df[either] = df[wbug_cols].any(axis="columns")
        label_cols.extend(wbug_cols)
        label_cols.append(either) # do not extend with single string either. it gets treated like an array of characters
        
        # GLM?
        flash_spacetime_win = f"flashes_{rptdist}km_{t}hr"
        if flash_spacetime_win in df:
            # Check if flash threshold is met in one space/time window.
            # new flash variable name has space and time window in it.
            logging.debug(
                f"at least {args.flash} flashes in {flash_spacetime_win}")
            df[flash_spacetime_win] = df[flash_spacetime_win] >= args.flash
            label_cols.append(flash_spacetime_win)

    if event is not None:
        # Recreate a smaller label_cols list with a single event type.
        label_cols = [f"{event}_{rptdist}km_{t}hr" for t in twin]

    return df, label_cols

def get_glm(time_space_windows, date=None, start=None, end=None):
    # Initialize Dataset to hold GLM flash count for all space/time windows.
    ds = xarray.Dataset()
    for twin, rptdist in time_space_windows:
        logging.info(f"get_glm: time/space window {twin}/{rptdist}")
        suffix = f".glm_{rptdist}km_{twin}hr.nc"
        if date:
            logging.info(f"date={date}")
            glmfiles = sorted(glob.glob(f"/glade/campaign/mmm/parc/ahijevyc/GLM/{date.strftime('%Y')}/{date.strftime('%Y%m%d_%H%M')}{suffix}"))
            glm = xarray.open_mfdataset(glmfiles, concat_dim="time", combine="nested")
        else:
            oneGLMfile = True
            if oneGLMfile:
                ifile = f"/glade/scratch/ahijevyc/temp/all{suffix}"
                if os.path.exists(ifile):
                    glm = xarray.open_dataset(ifile)
                else:
                    logging.error(f"{ifile} does not exist. To create, concatenate GLM files:")
                    logging.error(f"cd /glade/campaign/mmm/parc/ahijevyc/GLM")
                    logging.error(f"find ???? -name '*{suffix}' > filelist")
                    logging.error(f"cat filelist|ncrcat --fl_lst_in {ifile}")
                    sys.exit(1)
            else:
                glmfiles = sorted(glob.glob(f"/glade/campaign/mmm/parc/ahijevyc/GLM/{date.strftime('%Y')}/*{suffix}"))
                logging.info("open_mfdataset")
                glm = xarray.open_mfdataset(glmfiles, concat_dim="time", combine="nested")

        assert (glm.time[1] - glm.time[0]) == np.timedelta64(3600,'s'), 'glm.time interval not 1h'

        time = slice(start, end)
        logging.info(f"trim GLM to time window {time}")
        glm = glm.sel(time=time)

        newcol = f"flashes_{rptdist}km_{twin}hr"
        logging.info(f"Store in new DataArray {newcol}")
        glm = glm.rename(dict(time="valid_time",flashes=newcol))
        ds = ds.assign(variables=glm)
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
    fineprint = f"{desc} {label}\ncreated {str(datetime.datetime.now(tz=None)).split('.')[0]}"
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
    feature_list_file = f"suite_predictors/{args.model}.{args.suite}.txt"

    # Defined list of features in this function before Nov 24, 2022. But these 33 predictors were missing from the default suite. They weren't even in my first commit to github.
    # Why did they disappear? Maybe when I accidentally deleted my work directory in Oct 2021.
    # {'LR75-N5T5', 'REFL_COM-N5T3', 'LR75-N5T1', 'HAILCAST_DIAM_MAX-N5T1', 'HAILCAST_DIAM_MAX-N3T3', 'UP_HELI_MIN-N5T3', 'UP_HELI_MIN-N5T5',
    # 'HAILCAST_DIAM_MAX-N3T1', 'UP_HELI_MIN', 'REFL_COM-N5T1', 'REFL_COM-N5T5', 'MLCINH-N3T1', 'REFL_COM-N3T1', 'REFL_COM-N3T3', 'HAILCAST_DIAM_MAX', 'REFL_COM', 'HAILCAST_DIAM_MAX-N5T3', 
    # 'REFL_COM-N3T5', 'UP_HELI_MIN-N5T1', 'MLCINH-N5T5', 'MLCINH-N3T5', 'UP_HELI_MIN-N3T5', 'MLCINH-N5T3', 'LR75-N3T1', 'HAILCAST_DIAM_MAX-N5T5', 
    # 'HAILCAST_DIAM_MAX-N3T5', 'UP_HELI_MIN-N3T3', 'MLCINH-N5T1', 'LR75-N5T3', 'LR75-N3T3', 'UP_HELI_MIN-N3T1', 'LR75-N3T5', 'MLCINH-N3T3'}

    features = open(feature_list_file, "r").read().splitlines() # Hopefully these somewhat alphabetically-sorted lists are easier to spot mistakes in.
   
    # strip leading and trailing whitespace
    features = [x.strip() for x in features]

    if len(set(features)) != len(features):
        logging.warning(f"repeated feature(s) {set([x for x in features if features.count(x) > 1])}")
        features = list(set(features))

    return features


def make_fhr_str(fhr):
    # abbreviate list of forecast hours with hyphenated ranges of continuous times
    # so model name is not too long for tf.
    fhr.sort()
    seq = []
    final = []
    last = 0

    for index, val in enumerate(fhr):

        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
                final.append(f"f{seq[0]:02d}-f{seq[len(seq)-1]:02d}")
            else:
                final.append(f"f{seq[0]:02d}")
            seq = []
            seq.append(val)
            last = val

        if index == len(fhr) - 1:
            if len(seq) > 1:
                final.append(f"f{seq[0]:02d}-f{seq[len(seq)-1]:02d}")
            else:
                final.append(f"f{seq[0]:02d}")

   
    final_str = '.'.join(map(str, final))
    return final_str


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


def get_savedmodel_path(args, odir="nn"):
    # Use path requested on command line, if available.
    if args.savedmodel is not None:
        return args.savedmodel

    fhr_str = make_fhr_str(args.fhr)

    if args.batchnorm:
        batchnorm_str = ".bn"
    else:
        batchnorm_str = ""

    twin_str = f"_{'.'.join([str(int(t)) for t in args.twin])}hr"

    glmstr = "" # GLM description 
    if args.glm: glmstr = f"{args.flash:02d}flash." # zero-padded flash count threshold
        
    neurons_str = ''.join([f"{x}n" for x in args.neurons]) # [1024] -> "1024n", [16,16] -> "16n16n"
    savedmodel  = f"{odir}/{args.model}.{args.suite}.{glmstr}rpt_{args.rptdist}km{twin_str}.{neurons_str}.ep{args.epochs}.{fhr_str}."
    savedmodel += f"bs{args.batchsize}.{args.optimizer}.L2{args.reg_penalty}.lr{args.learning_rate}.dr{args.dropout}{batchnorm_str}"
        
    return savedmodel
