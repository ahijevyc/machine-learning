"""
machine learning functions
"""
from yaml.representer import SafeRepresenter
from yaml import CSafeDumper
import argparse
import datetime
import glob
import logging
import os
import pdb
import pickle
import sys
import time
from typing import Iterable

import dask.dataframe as ddf
from itertools import repeat
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.ndimage.filters
import xarray
import yaml
from scipy import spatial
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.calibration import calibration_curve
from tensorflow import is_tensor
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

from ahijevyc import G211


def assertclose(df, *c, lsuffix="", rsuffix="", atol=0.1):
    # use groupby(["x","y"]).mean() here instead of comparing entire df.lon series cause
    # there could be nans
    # Don't compare valid_times if eligible forecast hours were different like storm mode (12-35)
    for column in c:
        lc, rc = column + lsuffix, column + rsuffix
        if df[lc].isnull().all():
            logging.warning(f"{lc} is all null. skip assertclose check")
            continue
        if df[rc].isnull().all():
            logging.warning(f"{rc} is all null. skip assertclose check")
            continue
        xymean = df[[lc, rc]].groupby(["y", "x"]).mean(numeric_only=False)
        # Handle times and numbers differently
        if xymean[lc].dtype == "<M8[ns]" and xymean[rc].dtype == "<M8[ns]":
            logging.debug(f"are {lc} and {rc} time columns close?")
            assert xymean[lc].equals(
                xymean[rc]
            ), f"{lc} and {rc} are not equal {xymean[[lc,rc]]}"
        else:
            logging.debug(f"are {lc} and {rc} numeric columns close?")
            if not np.allclose(xymean[lc], xymean[rc], atol=atol):
                logging.error(f"{lc} and {rc} are not close {xymean[[lc,rc]]}")
                logging.error((xymean[lc] - xymean[rc]).abs().max())
                logging.error((xymean[lc] - xymean[rc]).abs().argmax())
                sys.exit(1)


# write and read safe yaml. write argparse.Namespace as dictionary and
# timestamps as strings
# experiment with this in ~ahijevyc/yaml_config.ipynb


class Dumper(CSafeDumper):
    pass


def timestamp_representer(dumper, data):
    return SafeRepresenter.represent_datetime(dumper, pd.Timestamp(data))


def namespace_representer(dumper, data):
    return SafeRepresenter.represent_dict(dumper, data.__dict__)


Dumper.add_representer(pd.Timestamp, timestamp_representer)
Dumper.add_representer(argparse.Namespace, namespace_representer)


def brier_skill_score(obs, preds):
    if is_tensor(obs) and is_tensor(preds):
        bs = K.mean((preds - obs) ** 2)
        # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
        obs_climo = K.mean(obs, axis=0)
        bs_climo = K.mean((obs - obs_climo) ** 2)
        # bss = 1.0 - (bs/bs_climo+K.epsilon()) # TODO: shouldn't K.epsilon() be grouped with denominator?
        bss = 1.0 - bs / (bs_climo + K.epsilon())
    else:
        bs = np.mean((preds - obs) ** 2, axis=0)
        # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
        obs_climo = np.mean(obs, axis=0)
        bs_climo = np.mean((obs - obs_climo) ** 2, axis=0)
        logging.debug(bs_climo)
        epsilon = np.finfo(bs_climo.dtype).eps
        logging.debug(epsilon)
        bss = 1.0 - bs / (bs_climo + epsilon)

    return bss


def configs_match(ylargs, args):
    # Warn if trimmed training period and requested test periods overlap.
    trainstart = getattr(ylargs, "trainstart")
    trainend = getattr(ylargs, "trainend")
    teststart = args.teststart
    testend = args.testend
    overlap = min([trainend, testend]) - max([trainstart, teststart])
    if overlap > datetime.timedelta(hours=0) and args.kfold == 1:
        logging.warning(
            f"training and testing time ranges overlap [{trainstart},{trainend}) [{teststart},{testend}]"
        )

    # Comparing yaml config training period and requested training period (args) is not a good test
    # because yaml config bounds are trimmed to actual available range of training cases.
    # config training period may be subset of the requested training period in args, but if any of the actual "trimmed" training period
    # beyond the range of the requested training period, this is a problem.
    # args.trainstart <= trainstart            trainend <= args.trainend
    assert (
        args.trainstart <= trainstart
    ), f"requested start of training period {args.trainstart} is after actual training period [{trainstart},{trainend}]"
    assert (
        trainend <= args.trainend
    ), f"requested end of training period {args.trainend} is before actual training period [{trainstart},{trainend}]"

    for key in [
        "batchnorm",
        "batchsize",
        "dropout",
        "epochs",
        "flash",
        "fhr",
        "kfold",
        "learning_rate",
        "model",
        "neurons",
        "optimizer",
        "reg_penalty",
        "suite",
    ]:
        assert getattr(ylargs, key) == getattr(
            args, key
        ), f"requested {key} {getattr(args,key)} does not match savedmodel yaml {key} {getattr(ylargs,key)}"
    # Ignore "labels" if this is an old configuration. I used to save labels as top-level list in yaml. Now it is an attribute of ylargs Namespace.
    # TODO: stop ignoring "labels". add it to the keys compared above. (Need to replace old dnn models with new yaml.config first)
    if hasattr(ylargs, "labels"):
        ylabels = getattr(ylargs, "labels")
        assert (
            ylabels == args.labels
        ), f"requested labels {args.labels} no match yaml labels {ylabels}"
    else:
        logging.warning(f"Can't compare labels. not part of ylargs namespace")
    return True


def get_argparser():
    parser = argparse.ArgumentParser(
        description="train/test dense neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batchnorm", action="store_true", help="use batch normalization")
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="nn training batch size"
    )  # tf default is 32
    parser.add_argument(
        "--clobber", action="store_true", help="overwrite any old outfile, if it exists"
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="fraction of neurons to drop in each hidden layer (0-1)",
    )
    parser.add_argument("--epochs", default=30, type=int, help="number of training epochs")
    parser.add_argument("--labels", nargs="+", default=[], help="labels")
    parser.add_argument(
        "--fhr",
        nargs="+",
        type=int,
        default=list(range(1, 49)),
        help="train with these forecast hours. Testing scripts only use this list to verify correct model "
        "for testing; no filter applied to testing data. In other words you "
        "test on all forecast hours in the testing data, regardless of whether the model was "
        "trained with the same forecast hours.",
    )
    parser.add_argument(
        "--fits",
        nargs="+",
        type=int,
        default=None,
        help="work on specific fit(s) so you can run many in parallel",
    )
    parser.add_argument("--flash", type=int, default=10, help="GLM flash count threshold")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="work on specific fold(s) so you can run many in parallel",
    )
    parser.add_argument(
        "--kfold", type=int, default=5, help="apply kfold cross validation to training set"
    )
    parser.add_argument(
        "--idate", type=lambda s: pd.to_datetime(s), help="single initialization time"
    )
    parser.add_argument(
        "--ifile", help="Read this parquet input file. Otherwise guess which one to read."
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--model",
        choices=["HRRR", "NSC1km", "NSC3km-12sec", "NSC15km"],
        default="HRRR",
        help="prediction model",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=[16, 16],
        help="number of neurons in each nn layer",
    )
    parser.add_argument(
        "--nfits", type=int, default=10, help="number of times to fit (train) model"
    )
    parser.add_argument(
        "--nprocs", type=int, default=0, help="verify this many forecast hours in parallel"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["Adam", "SGD"], default="Adam", help="optimizer"
    )
    parser.add_argument(
        "--reg_penalty", type=float, default=0.01, help="L2 regularization factor"
    )
    parser.add_argument("--savedmodel", help="filename of machine learning model")
    parser.add_argument(
        "--seed", type=int, default=None, help="random number seed for reproducability"
    )
    parser.add_argument(
        "--trainstart",
        type=lambda s: pd.to_datetime(s),
        default="19700101",
        help="training set start",
    )
    parser.add_argument(
        "--trainend",
        type=lambda s: pd.to_datetime(s),
        default="20220101",
        help="training set end",
    )
    parser.add_argument(
        "--teststart",
        type=lambda s: pd.to_datetime(s),
        #default="20201202T12",
        default="20210101",
        help="testing set start",
    )
    parser.add_argument(
        "--testend",
        type=lambda s: pd.to_datetime(s),
        default="20220101",
        help="testing set end",
    )
    parser.add_argument(
        "--twin",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="centered time window duration (plus/minus half this number)",
    )
    parser.add_argument(
        "--suite", default="default", help="name for suite of training features"
    )

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
    for kw, value in args._get_kwargs():
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


def get_optimizer(s, learning_rate=0.001, **kwargs):
    if s == "Adam":
        o = optimizers.Adam(learning_rate=learning_rate)
    elif s == "SGD":
        # learning_rate = 0.001 # from sobash
        momentum = 0.99
        nesterov = True
        o = optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, **kwargs
        )
    return o


def get_combined_parquet_input_file(args):
    """
    Return name of parquet file that is combination of all daily input
    """
    if args.ifile is not None:
        return ifile

    ifile = os.path.join(os.getenv("TMPDIR"), f"{args.model}.{args.twin}hr.par")
    if args.debug:
        base, ext = os.path.splitext(ifile)
        ifile = base + ".debug" + ext

    if args.idate:
        ifile = os.path.join(
            os.getenv("TMPDIR"),
            f"{args.model}.{args.twin}hr.{args.idate.strftime('%Y%m%d%H-%M%S')}.par",
        )
        logging.debug(f"get_combined_parquet_input_file: {ifile}")
    return ifile


debug_replace = "202106*"


def get_ifiles(args, idir):
    """
    Return list of daily input files
    """
    idate = args.idate
    model = args.model

    # Define ifiles, a list of input files from glob.glob method
    if model == "HRRR":
        # HRRRX = experimental HRRR (v4)
        search_str = f"{idir}/HRRR_new/grid_data/grid_data_HRRRX_d01_20*00-0000.par"
        if args.debug:
            search_str = search_str.replace("HRRRX", "HRRR").replace("20*", debug_replace)
        else:
            # append HRRR to HRRRX.
            # HRRR prior to Dec 3 2020 is v3 and HRRR at Dec 3 2020 and afterwards is v4.
            # Dec 3-9, 2020 00z only
            search_str += (
                f" {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_2020120[3-9]*00-0000.par"
            )
            # Dec 10-31, 2020 00z only
            search_str += (
                f" {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_202012[1-3]*00-0000.par"
            )
            # 2021 00z only
            # used to include 2022+ but they have no storm reports
            # added 2022 just for lightning
            search_str += f" {idir}/HRRR_new/grid_data/grid_data_HRRR_d01_202[12]*00-0000.par"
        logging.info(f"ifiles search string {search_str}")
        ifiles = []
        for x in search_str.split(" "):
            ifiles.extend(glob.glob(x))
    elif model.startswith("NSC"):
        search_str = f"{idir}/grid_data_new/grid_data_{model}_d01_20*00-0000.par"
        if args.debug:
            search_str = search_str.replace("20*", debug_replace)
        logging.info(f"ifiles search string {search_str}")
        ifiles = glob.glob(search_str)

    if idate:
        if model == "HRRR":
            assert idate >= pd.to_datetime("20191002"), f"No {model} before 20191002"
            d = f'{idir}/HRRR_new/grid_data/grid_data_HRRRX_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
            if idate >= pd.to_datetime("20201202"):
                d = d.replace("HRRRX", "HRRR")
        elif model.startswith("NSC"):
            d = f'{idir}/grid_data_new/grid_data_{model}_d01_{idate.strftime("%Y%m%d%H-%M%S")}.par'
        else:
            logging.error(f"unexpected model {model} with idate {idate}")
            sys.exit(1)
        ifiles = [d]

    return sorted(ifiles)


def load_df(
        args: argparse.Namespace,
        idir: str ="/glade/work/sobash/NSC_objects",
        wbugdir: str ="/glade/campaign/mmm/parc/ahijevyc/ENTLN",
        index_cols: Iterable =["initialization_time", "valid_time", "y", "x"],
):
    """
    Return DataFrame with all input data for a particular model
    and grid size. Contains features, labels, and index_cols
    """
    idate = args.idate
    model = args.model
    twin = args.twin

    ifile = get_combined_parquet_input_file(args)

    feature_list = get_features(args)
    columns = feature_list + args.labels + index_cols

    if os.path.exists(ifile):
        logging.warning(
            f"reading {ifile} {os.path.getsize(ifile)/1024**3:.1f}G "
            f"mtime {time.ctime(os.path.getmtime(ifile))} "
            f"{len(feature_list)} features {len(args.labels)} labels "
            f"and {len(index_cols)} index_cols & dropna."
        )
        # tacking on dropna was 20% faster than doing it separately afterwards.
        df = pd.read_parquet(ifile, engine="pyarrow", columns=columns).dropna()

        return df

    # copied and pasted from dask_HRRR_read.ipynb after testing - Aug 29, 2023
    engine = ddf
    ifiles = get_ifiles(args, idir)
    logging.warning(f"Create {ifile} from {len(ifiles)} {model} files.")
    fmt = "%Y%m%d%H-%M%S.par"
    earliest_valid_time = datetime.datetime.strptime(
        os.path.basename(ifiles[0])[-19:], fmt
    ) + pd.Timedelta(hours=1)
    latest_valid_time = datetime.datetime.strptime(
        os.path.basename(ifiles[-1][-19:]), fmt
    ) + pd.Timedelta(hours=48)
    df = engine.read_parquet(ifiles, engine="pyarrow")

    df = df.rename(
        columns=dict(xind="y", yind="x", Date="initialization_time", fhr="forecast_hour")
    )
    logging.info(f"convert initialization_time to datetime")
    df["initialization_time"] = engine.to_datetime(
        df["initialization_time"], format="%Y-%m-%d %H:%M:%S"
    )
    logging.info(f"derive valid_time from Date + forecast_hour")
    df["valid_time"] = df["initialization_time"] + engine.to_timedelta(
        df["forecast_hour"], unit="hours"
    )
    dayofyear = df["valid_time"].dt.dayofyear
    df["dayofyear_sin"] = np.sin(dayofyear * 2 * np.pi / 365.25)
    df["dayofyear_cos"] = np.cos(dayofyear * 2 * np.pi / 365.25)

    Local_Solar_Hour = df["valid_time"].dt.hour + df["lon"] / 15
    df["Local_Solar_Hour_sin"] = np.sin(Local_Solar_Hour * 2 * np.pi / 24)
    df["Local_Solar_Hour_cos"] = np.cos(Local_Solar_Hour * 2 * np.pi / 24)

    # Earth Networks Total Lightning Network (ENTLN)
    # Previously Weatherbug.
    # cg, ic lightning in rptdist = 20km and 40km grids
    # Counts are in 30-minute bins labeled at the start of the bin.
    ENTLN_dict = {}
    for rptdist in [20, 40]:
        iwbug = os.path.join(wbugdir, f"flash.{rptdist}km_30min.nc")
        logging.info(f"load ENTLN lightning data {iwbug}")
        wbug = xarray.open_dataset(iwbug, chunks={"time_coverage_start": 270})
        wbug["cg.ic"] = wbug.cg + wbug.ic
        wbugtimes = slice(earliest_valid_time, latest_valid_time - pd.Timedelta(minutes=30))
        wbug = wbug.sel(time_coverage_start=wbugtimes)
        # mean of 30-minute lightning blocks in time window
        # debug with notebooks/ENTLN_sum_time_window.ipynb
        logging.info(f"sum ENTLN {rptdist}km flashes in {twin}hr time window")
        ltg_sum = (
            # If you use xr.DataArray.shift() or .rolling() on a time series with dimension time_coverage_start,
            # the time shift or time window varies unless time_coverage_start is evenly spaced with no missing times.
            # Therefore, resample every 30 minutes, using a missing value for the count if the time is missing. 
            # Missing times are filled in, but given a missing value (nan). Therefore the rolling window
            # always has a duration of twin*2 hours.
            # We take the .mean in the rolling time window, requiring at least min_periods=twin time_coverage_starts
            # with non-missing counts. We set center=True so the .rolling window is centered on the time dimension
            # and we rename the time dimension `valid_time`. Then multiply by the time window duration twin*2 hours to get a count.
            wbug.resample(time_coverage_start="30min")
            .first()
            .rolling(
                dim={"time_coverage_start": twin * 2},
                min_periods=twin,  # at least half times must be present
                center=True,
            )
            .mean()
            .rename({"time_coverage_start": "valid_time"})
            * twin
            * 2
        )

        # Append rptdist and twin strings to ENTLN variable names.
        name_dict = {s: f"{s}_{rptdist}km_{twin}hr" for s in ["cg", "ic", "cg.ic"]}
        logging.info(f"rename {name_dict}")
        ltg_sum = ltg_sum.rename(name_dict)
        logging.info(f"add rptdist={rptdist}km to ENTLN_dict")
        ENTLN_dict[rptdist] = ltg_sum

    # At each G211 point, assign ENTLN_dict[20] at nearest half-G211 point
    g211 = G211.GridManager(factor=1)
    lonsG211 = g211.lon.ravel()
    latsG211 = g211.lat.ravel()
    g211x2 = G211.GridManager(factor=2)
    lonsG211x2 = g211x2.lon.ravel()
    latsG211x2 = g211x2.lat.ravel()
    tree = spatial.KDTree(list(zip(lonsG211x2, latsG211x2)))
    dist, indices = tree.query(list(zip(lonsG211, latsG211)))
    ENTLN20_coarse = ENTLN_dict[20].stack(pt=("y", "x")).isel(pt=indices)

    # ENTLN20_coarse has lon and lat values of G211 
    # but its x and y coordinates were iselected with pt=indices, so they 
    # are not simply monotonic 0-92 and 0-64 like in G211 coords.
    logging.info("update half-G211 y and x coordinates with full-G211 y and x coordinates")
    c = ENTLN20_coarse.coords
    c.update(g211.mask.stack(pt=("y", "x")).coords)
    ENTLN20_coarse = ENTLN20_coarse.assign_coords(c).unstack(dim="pt")

    # Used to merge ENTLN with HRRR here, but now I wait until I have GLM too.
    # If either ENTLN or GLM is present (and HRRR is present) we want to keep that time.
    # Before, if either was missing for a particular time, the whole time would be dropped
    # because merge(how="inner") was used on a ENTLN merge and a GLM merge.

    # Geostationary Lightning Mapper (GLM)
    glm40 = get_glm((twin, 40), start=earliest_valid_time, end=latest_valid_time)
    glm20_coarse = (
        get_glm((twin, 20), start=earliest_valid_time, end=latest_valid_time)
        .stack(pt=("y", "x"))
        .isel(pt=indices)
    )

    # TODO: make sure glm40 and glm20_coarse have same times, except for maybe the
    # ragged end, where one might have been pre-processed with more available times.

    c = glm20_coarse.coords
    c.update(g211.mask.stack(pt=("y", "x")).coords)
    logging.info("assign_coords, unstack pt dim")
    # TODO: fix <__array_function__ internals>:200: RuntimeWarning: invalid value encountered in cast
    glm20_coarse = glm20_coarse.assign_coords(c).unstack(dim="pt")

    logging.info("merge ENTLN20_coarse, ENTLN40, glm20_coarse, glm40")
    all_ltg = xarray.merge(
        [ENTLN20_coarse, ENTLN_dict[40], glm20_coarse, glm40], compat="override"
    ).to_dataframe()

    # dask can't handle MultiIndex. use dask.dataframe.compute to convert from dask to
    # regular Pandas DataFrame.
    logging.info("computing dask dataframe, merging")
    df = df.compute().merge(
        all_ltg,
        how="inner",
        left_on=all_ltg.index.names,
        right_index=True,
        suffixes=(None, "_y"),
    )

    sanity_check = False
    if sanity_check:
        ll = ["lat", "lon"]
        xymean = df[["lat", "lon", "lat_y", "lon_y", "y", "x"]].groupby(["y", "x"]).mean()
        assert np.allclose(xymean[ll], xymean[[l + "_y" for l in ll]], atol=0.289)

    df = df.drop(columns=[f"lon_y", f"lat_y"])

    logging.info("convert 64-bit to 32-bit columns")
    df = update_dtype(df)

    logging.info(f"writing {ifile}")
    df.to_parquet(ifile)

    # Saved all columns to parquet, but only return columns subset.
    df = df[columns]

    # Used to test all columns for NA, but we only care about subset columns.
    # For example, mode probs are not available for fhr=2 but we don't need to drop fhr=2 if
    # the other features are complete.
    # We don't want rptdist2bool to convert missing labels to Falses.
    # Before Oct 20, 2023, it did. Now it asserts none are missing first.
    logging.warning(f"Drop na")
    beforedropna = len(df)
    df = df.dropna()
    logging.warning(f"kept {len(df)}/{beforedropna} {len(df)/beforedropna:.0%} rows")

    return df


def rptdist2bool(df, args):
    """
    Derive "any" severe storm report label
    Return DataFrame with storm report distances and flash counts converted to Boolean.
    These columns have new names that include distance and time window.
    """

    twin = args.twin
    for rptdist in [20, 40]:
        labels = args.labels
        # Don't want any missing labels. They will be treated as False instead of NA when thresholded.
        assert (
            not df[labels].isna().any(axis=None)
        ), f"label(s) missing {df[labels].isna().any()}"
        lsrtypes = [
            "sighail",
            "sigwind",
            "hailone",
            "wind",
            "torn",
            "windmg",
            "svrwarn",
            "torwarn",
        ]
        oldtwin = [0, 1, 2]
        logging.warning(
            f"use {oldtwin} time win for {len(lsrtypes)} lsrtypes until parquet renamed [1,2,4]"
        )
        label_cols = [f"{r}_rptdist_{t}hr" for r in lsrtypes for t in oldtwin]
        # only keep those in labels
        label_cols = [x for x in label_cols if x in labels]

        # refer to new label names (with f"{rptdist}km" not f"rptdist")
        new_label_cols = [r.replace("rptdist", f"{rptdist}km") for r in label_cols]
        logging.info(f"Convert severe report distance to boolean [0,{rptdist}km) = True")
        df[new_label_cols] = (0 <= df[label_cols]) & (df[label_cols] < rptdist)

        for t in oldtwin:
            # any type of severe storm report
            any_label_str = f"any_{rptdist}km_{t}hr"
            if any_label_str not in labels:
                continue
            labels_this_twin = [f"{r}_{rptdist}km_{t}hr" for r in ["hailone", "wind", "torn"]]
            logging.debug(f"derive {any_label_str} from {labels_this_twin}")
            df[any_label_str] = df[labels_this_twin].any(axis="columns")

        # ENTLN (previously Weatherbug) flashes
        wbug_cols = [f"{f}_{rptdist}km_{twin}hr" for f in ["cg", "ic", "cg.ic"]]
        logging.info(f"threshold ENTLN at {args.flash} flashes")
        df[wbug_cols] = df[wbug_cols] >= args.flash

        # GLM flashes
        flash_spacetime_win = f"flashes_{rptdist}km_{twin}hr"
        if flash_spacetime_win in labels:
            # Check if flash threshold is met in one space/time window.
            # new flash variable name has space and time window in it.
            logging.debug(f"threshold {flash_spacetime_win} >= {args.flash}")
            df[flash_spacetime_win] = df[flash_spacetime_win] >= args.flash

    return df


def get_glm(
    time_space_window: tuple((float, float)), date=None, start=None, end=None, oneGLMfile=True
):
    """join Geostationary Lightning Mapper (GLM)"""
    firstglm = pd.to_datetime("20180213")
    if end < firstglm:
        logging.warning(
            f"requested GLM time range [{start}-{end}] prior to first GLM day {firstglm}"
        )
    twin, rptdist = time_space_window
    logging.info(f"get_glm: time/space window {twin}/{rptdist} start={start} end={end}")
    suffix = f".glm_{rptdist}km_{twin}hr.nc"
    fmt = "%Y%m%d_%H%M"
    if date:
        logging.info(f"date={date}")
        glmfiles = sorted(
            glob.glob(
                f"/glade/campaign/mmm/parc/ahijevyc/GLM/{date.strftime('%Y')}/{date.strftime(fmt)}{suffix}"
            )
        )
        glm = xarray.open_mfdataset(glmfiles, concat_dim="time", combine="nested")
    else:
        if oneGLMfile:
            tmpdir = Path(os.getenv("TMPDIR"))
            ifile = tmpdir / f"all{suffix}"
            if os.path.exists(ifile):
                logging.info(f"open {ifile} for GLM flashes")
                glm = xarray.open_dataset(ifile)
            else:
                logging.error(f"{ifile} does not exist. To create, concatenate GLM files:")
                logging.error(f"cd /glade/campaign/mmm/parc/ahijevyc/GLM")
                logging.error(
                    f"find 20[0-9][0-9] -name '*{suffix}' | sort | ncrcat -D 2 -o {ifile}"
                )
                sys.exit(1)
        else:
            glmfiles = glob.glob(
                f"/glade/campaign/mmm/parc/ahijevyc/GLM/2[0-9][0-9][0-9]/*{suffix}"
            )
            logging.warning(f"found {len(glmfiles)} files")
            glmfiles = [
                x
                for x in glmfiles
                if datetime.datetime.strptime(os.path.basename(x)[:13], fmt) >= start
                and datetime.datetime.strptime(os.path.basename(x)[:13], fmt) <= end
            ]
            logging.warning(
                f"open_mfdataset {len(glmfiles)} files in time window [{start},{end}]"
            )
            glm = xarray.open_mfdataset(glmfiles, concat_dim="time", combine="nested")

        assert (glm.time[1] - glm.time[0]) == np.timedelta64(
            3600, "s"
        ), "glm.time interval not 1h"

        time = slice(start, end)
        logging.info(f"trim GLM to {time}")
        glm = glm.sel(time=time)

        newcol = f"flashes_{rptdist}km_{twin}hr"
        logging.info(f"Store in new DataArray {newcol}")
        glm = glm.rename(dict(time="valid_time", flashes=newcol))
    if glm.valid_time.size == 0:
        logging.warning(f"GLM Dataset is empty.")
    return glm


def upscale(field, nngridpts, type="mean", maxsize=27):
    if type == "mean":
        field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode="nearest")
        # field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode='constant') # mode shouldn't matter. mask takes over. But it does matter for pre-Apr 21, 2015 T2 field.
        # For some reason, in this case T2 is transformed to 0-4 range
    elif type == "max":
        field = scipy.ndimage.filters.maximum_filter(field, size=maxsize)
    elif type == "min":
        field = scipy.ndimage.filters.minimum_filter(field, size=maxsize)

    field_interp = field.flatten()[nngridpts[1]].reshape((65, 93))

    return field_interp


def get_features(args):
    feature_list_file = (
        f"/glade/work/ahijevyc/NSC_objects/predictor_suites/{args.model}.{args.suite}.txt"
    )

    # Defined list of features in this function before Nov 24, 2022. But these 33 predictors were missing from the default suite. They weren't even in my first commit to github.
    # Why did they disappear? Maybe when I accidentally deleted my work directory in Oct 2021.
    # {'LR75-N5T5', 'REFL_COM-N5T3', 'LR75-N5T1', 'HAILCAST_DIAM_MAX-N5T1', 'HAILCAST_DIAM_MAX-N3T3', 'UP_HELI_MIN-N5T3', 'UP_HELI_MIN-N5T5',
    # 'HAILCAST_DIAM_MAX-N3T1', 'UP_HELI_MIN', 'REFL_COM-N5T1', 'REFL_COM-N5T5', 'MLCINH-N3T1', 'REFL_COM-N3T1', 'REFL_COM-N3T3', 'HAILCAST_DIAM_MAX', 'REFL_COM', 'HAILCAST_DIAM_MAX-N5T3',
    # 'REFL_COM-N3T5', 'UP_HELI_MIN-N5T1', 'MLCINH-N5T5', 'MLCINH-N3T5', 'UP_HELI_MIN-N3T5', 'MLCINH-N5T3', 'LR75-N3T1', 'HAILCAST_DIAM_MAX-N5T5',
    # 'HAILCAST_DIAM_MAX-N3T5', 'UP_HELI_MIN-N3T3', 'MLCINH-N5T1', 'LR75-N5T3', 'LR75-N3T3', 'UP_HELI_MIN-N3T1', 'LR75-N3T5', 'MLCINH-N3T3'}

    # Hopefully these somewhat alphabetically-sorted lists are easier to spot mistakes in.
    features = open(feature_list_file, "r").read().splitlines()

    # strip leading and trailing whitespace
    features = [x.strip() for x in features]

    if len(set(features)) != len(features):
        logging.warning(
            f"repeated feature(s) {set([x for x in features if features.count(x) > 1])}"
        )
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

    final_str = ".".join(map(str, final))
    return final_str


def predct(i, args, df):
    """
    Return DataFrame of predictions for this (ifold, thisfit).
    Used global variable features dataframe, df.
    Used by test_stormrpts_dnn.py and lightning_prob.ipynb.
    Remove and use predct2 instead once you adapt lightning_prob.ipynb
    to use predct2.
    """
    assert False, "Don't use predct, use predct2"
    ifold, thisfit = i
    savedmodel = get_savedmodel_path(args)
    savedmodel_thisfitfold = f"{savedmodel}_{thisfit}/{args.kfold}fold{ifold}"
    logging.warning(f"{i} {savedmodel_thisfitfold}")
    yl = yaml.load(
        open(os.path.join(savedmodel_thisfitfold, "config.yaml"), "r"), Loader=yaml.Loader
    )
    if "labels" in yl:
        labels = yl["labels"]
        # delete labels so we can make DataFrame from rest of dictionary.
        del yl["labels"]
    else:
        labels = getattr(yl["args"], "labels")

    assert configs_match(
        yl["args"], args
    ), f'this configuration {args} does not match yaml file {yl["args"]}'
    del yl["args"]
    # scaling values DataFrame as from .describe()
    sv = pd.DataFrame(yl).set_index("columns").T
    if sv.columns.size != df.columns.size:
        logging.error(f"size of yaml and features columns differ {sv.columns} {df.columns}")
    assert all(
        sv.columns == df.columns
    ), f"columns {df.columns} don't match when model was trained {sv.columns}"

    logging.info(f"loading {savedmodel_thisfitfold}")
    model = load_model(savedmodel_thisfitfold)
    df_fold = df
    if args.kfold > 1:
        cv = KFold(n_splits=args.kfold)
        # Convert generator to list. You don't want a generator.
        # Generator depletes after first run of statjob, and if run serially,
        # next time statjob is executed the entire fold loop is skipped.
        cvsplit = list(cv.split(df))
        itrain, itest = cvsplit[ifold]
        df_fold = df.iloc[itest]
    norm_features = (df_fold - sv.loc["mean"]) / sv.loc["std"]
    # Grab numpy array of predictions.
    Y = model.predict(norm_features.to_numpy(dtype="float32"), batch_size=10000)
    Y = pd.DataFrame(Y, columns=labels, index=df_fold.index)
    return Y


def predct2(i, args, df):
    """
    Return DataFrame of predictions and labels for this (ifold, thisfit).
    Used global variable features dataframe, df.
    Used by test_stormrpts_dnn.py and lightning_prob.ipynb.
    """
    ifold, thisfit = i
    savedmodel = get_savedmodel_path(args)
    savedmodel_thisfitfold = f"{savedmodel}_{thisfit}/{args.kfold}fold{ifold}"
    logging.warning(f"{i} {savedmodel_thisfitfold}")
    yaml_file = os.path.join(savedmodel_thisfitfold, "config.yaml")
    yl = yaml.load(open(yaml_file, "r"), Loader=yaml.Loader)

    # pop `labels` item so we can make DataFrame from rest of dictionary.
    labels = yl.pop("labels", getattr(yl["args"], "labels"))

    assert configs_match(
        yl["args"], args
    ), f'this configuration {args} does not match {yaml_file} {yl["args"]}'
    del yl["args"]
    feature_list = get_features(args)
    assert len(yl["columns"]) == len(
        feature_list
    ), f"size of yaml 'columns' and args feature_list differ {yl['columns']} {feature_list}"
    assert (
        yl["columns"] == feature_list
    ), f"yaml 'columns' and args feature_list differ {yl['columns']} {feature_list}"

    # scaling values DataFrame as from .describe()
    sv = pd.DataFrame(yl).set_index("columns").T

    logging.info(f"loading {savedmodel_thisfitfold}")
    model = load_model(savedmodel_thisfitfold)
    df_fold = df
    if args.kfold > 1:
        cv = KFold(n_splits=args.kfold)
        # Convert generator to list. You don't want a generator.
        # Generator depletes after first run of statjob, and if run serially,
        # next time statjob is executed the entire fold loop is skipped.
        cvsplit = list(cv.split(df))
        itrain, itest = cvsplit[ifold]
        df_fold = df.iloc[itest]
    norm_features = (df_fold[feature_list] - sv.loc["mean"]) / sv.loc["std"]
    # To avoid warning about about tf.function repeat tracing, set model.run_eagerly = True.
    # Otherwise, if false, wrap in tf.function and run trace tf.graph for speedup.
    model.run_eagerly = True
    # Grab numpy array of predictions.
    y_preds = model.predict(norm_features.to_numpy(dtype="float32"), batch_size=10000)
    y_preds = pd.DataFrame(y_preds, columns=labels, index=df_fold.index)
    # predictions (y_pred) and labels (y_label) as MultiIndex columns
    Y = pd.concat([y_preds, df[labels]], axis=1, keys=["y_pred", "y_label"])

    return Y


def read_csv_files(sdate, edate, dataset, members=[str(x) for x in range(1, 11)], columns=None):
    # read in all CSV files for 1km forecasts
    all_files = []
    for member in members:
        all_files.extend(
            glob.glob(
                f"/glade/scratch/ahijevyc/NSC_objects/grid_data_{dataset}_mem{member}_d01_????????-0000.par"
            )
        )
    all_files = set(all_files)  # in case you ask for same member twice
    logging.debug("found " + str(len(all_files)) + " files")
    all_files = [
        x
        for x in all_files
        if sdate.strftime("%Y%m%d") <= x[-17:-9] <= edate.strftime("%Y%m%d")
    ]
    all_files = sorted(all_files)  # important for predictions_labels output
    logging.debug(f"Reading {len(all_files)} forecasts from {sdate} to {edate}")

    # df = pd.concat((pd.read_csv(f, compression='gzip', dtype=type_dict) for f in all_files))
    # df = pd.concat((pd.read_csv(f, dtype=type_dict) for f in all_files))
    ext = all_files[0][-4:]
    if ext == ".csv":
        df = pd.concat((pd.read_csv(f, engine="c") for f in all_files), sort=False)
    elif ext == ".par":
        df = pd.concat((pd.read_parquet(f) for f in all_files), sort=False)
    else:
        print("unexpected extension", ext, "exiting")
        sys.exit(1)
    logging.debug("finished reading")
    # started adding member in previous step (random_forest_preprocess_gridded.py)
    if "member" not in df.columns:
        logging.debug("adding members column")
        import re

        member = [re.search(r"_mem(\d+)", s).groups()[0] for s in all_files]
        # repeat each element n times. where n is number of rows in a single file's dataframe
        # avoid ValueError: Length of values does not match length of index
        df["member"] = np.repeat(np.int8(member), len(df) / len(all_files))
    # if model == 'NSC': df['stp']   = df.apply(computeSTP, axis=1)
    # if model == 'NSC': df['datetime']  = pd.to_datetime(df['Valid_Date'])
    # df = df.reset_index(level=0).pivot(columns="level_0")
    # df.columns = [' '.join(col).strip() for col in df.columns.values]
    # df = df.reset_index('Date')
    if "datetime" not in df.columns:
        logging.debug("adding datetime")
        df["datetime"] = pd.to_datetime(df["Date"])
    # df['Run_Date'] = pd.to_datetime(df['Date']) - pd.to_timedelta(df['fhr'])
    if "year" not in df.columns:
        logging.debug("adding year")
        df["year"] = df["datetime"].dt.year.astype(np.uint16)
    if "month" not in df.columns:
        logging.debug("adding month")
        df["month"] = df["datetime"].dt.month.astype(np.uint8)
    if "hour" not in df.columns:
        logging.debug("adding hour")
        df["hour"] = df["datetime"].dt.hour.astype(np.uint8)
    if "dayofyear" not in df.columns:
        logging.debug("adding dayofyear")
        df["dayofyear"] = df["datetime"].dt.dayofyear.astype(np.uint16)
    logging.debug("leaving read_csv()")

    return df, len(all_files)


def normalize_multivariate_data(data, features, scaling_values=None, nonormalize=False):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: Pandas DataFrame (not normalized, could have extraneous columns not in features list)
        features: list of features
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    logging.debug(data.shape)
    if hasattr(data, "dtype"):
        logging.debug(data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        data = data[features]
        logging.debug(data.info())
        scaling_values = pd.DataFrame(columns=scale_cols)
        scaling_values["mean"] = data.mean()
        logging.debug(scaling_values.info())
        scaling_values["std"] = data.std()
        logging.debug(scaling_values.info())
        if nonormalize:
            logging.warning(
                "ml_functions.normalize_multivariate_data(): no normalization. returning scaling_values"
            )
            return None, scaling_values
        data = data.values

    normed_data = np.zeros(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        # normed_data[:, i] = (data[:, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
        normed_data[:, i] = (
            data[:, i] - scaling_values.loc[features[i], "mean"]
        ) / scaling_values.loc[features[i], "std"]
        logging.debug(scaling_values.loc[features[i]])
        logging.info(features[i])
    return normed_data, scaling_values


def update_dtype(df):
    logging.debug("convert 64-bit to 32-bit columns")
    dtype_dict = {k: np.float32 for k in df.select_dtypes(np.float64).columns}
    dtype_dict.update({k: np.int32 for k in df.select_dtypes(np.int64).columns})
    df = df.astype(dtype_dict)
    return df


def get_savedmodel_path(args, odir="nn"):
    # Use path requested on command line, if available.
    if args.savedmodel is not None:
        return args.savedmodel

    fhr_str = make_fhr_str(args.fhr)

    if args.batchnorm:
        batchnorm_str = ".bn"
    else:
        batchnorm_str = ""

    # zero-padded flash count threshold. GLM description
    glmstr = f"{args.flash:02d}flash."

    # [1024] -> "1024n", [16,16] -> "16n16n"
    neurons_str = "".join([f"{x}n" for x in args.neurons])
    savedmodel = (
        f"{odir}/{args.model}.{args.suite}.{glmstr}{neurons_str}.ep{args.epochs}.{fhr_str}."
    )
    savedmodel += f"bs{args.batchsize}.{args.optimizer}.L2{args.reg_penalty}.lr{args.learning_rate}.dr{args.dropout}{batchnorm_str}"

    return savedmodel


def get_flash_pred(
    args: argparse.Namespace,
    clobber: bool = False,
    levels=["initialization_time", "valid_time", "y", "x"],
    feature_levels=["forecast_hour", "lat", "lon"],
) -> pd.DataFrame:
    """
    Return DataFrame with DNN predictions and observed labels.
    Save levels + feature_levels as levels in the returned MultiIndex.
    These can be used to group and cut by feature values, like lat, lon,
    forecast_hour, valid_time, initialization_time.
    Columns in `levels` are used as indices to group by and ultimately dropped from columns.
    Columns in `feature_levels` are also used as indices to group by but are preserved in the columns.
    """

    tmpdir = Path(os.getenv("TMPDIR"))
    oypreds = tmpdir / f"Y.{args.flash:03d}+{args.twin}hr.{args.teststart.strftime('%Y%m%d%H')}-{args.testend.strftime('%Y%m%d%H')}.par"

    ifile = get_combined_parquet_input_file(args)
    # clobber if combined parquet input file is newer than oypreds.
    if os.path.exists(ifile) and os.path.getmtime(ifile) > os.path.getmtime(oypreds):
        logging.warning(f"combined parquet input file {ifile} newer than oypreds {oypreds}; redo oypreds.")
        clobber = True

    if not clobber and os.path.exists(oypreds):
        logging.warning(f"read saved model output {oypreds}")
        Y = pd.read_parquet(oypreds)
        logging.warning(f"done")
    else:
        df = load_df(args)
        logging.warning(f"valid times: {df.valid_time.min()}-{df.valid_time.max()}")

        logging.warning(f"Use valid times [{args.teststart},{args.testend}) for testing")
        before_filtering = len(df)
        idx = (args.teststart <= df.valid_time) & (df.valid_time < args.testend)
        df = df[idx]
        logging.warning(
            f"kept {len(df)}/{before_filtering} "
            f"{len(df)/before_filtering:.0%} cases in testing time window"
        )

        # Put "initialization_time", "valid_time", "y", and "x" in MultiIndex
        # so we can group by them later.
        logging.warning(f"set_index {levels}")
        df = df.set_index(levels)
        # Append feature levels to index and retain as column.
        logging.warning(f"set_index feature levels {feature_levels}")
        df = df.set_index(feature_levels, drop=False, append=True)
        levels += feature_levels

        logging.warning(f"run model, save results to {oypreds}")
        index = pd.MultiIndex.from_product(
            [range(args.kfold), range(args.nfits)], names=["fold", "fit"]
        )
        # Remember to request multiple cpus and >600G memory when starting jupyter
        with Pool(processes=2) as pool:  # would like to use args.nfit but takes too much memory
            result = pool.starmap(predct2, zip(index, repeat(args), repeat(df)))
        Y = pd.concat(result, keys=index, names=index.names)
        Y.to_parquet(oypreds)
    return Y


def get_args(
    o_thresh: float,
    twin: float,
    trainstart: str = "20191002",
    trainend: str = "20201202",
    teststart: str = "20210101",
    testend: str = "20220101",
    epoch: int = 30,
    optim="Adam",
) -> argparse.Namespace:
    """return argparse.Namespace for ML model"""
    parser = get_argparser()
    # use [0, 1, 2] time windows for ['sighail', 'sigwind', 'hailone', 'wind', 'torn']
    # until we update parquet files with [1,2,4]
    rpttwin = int(np.floor(twin / 2))  # Ryan's original way of naming storm rpt time windows
    args = parser.parse_args(
        args=f"--seed -1 --model HRRR --batchsize 1024 --neurons 1024 --optim {optim} "
        f"--trainstart {trainstart} --trainend {trainend} "
        f"--teststart {teststart} --testend {testend} "
        f"--flash {o_thresh} "
        f"--twin {twin} "
        f"--savedmodel /glade/work/ahijevyc/NSC_objects/nn/lightning/{o_thresh:03d}+.{twin}hr "
        "--labels "
        # f"sighail_{rptdist}km_{rpttwin}hr sigwind_{rptdist}km_{rpttwin}hr "
        # f"hailone_{rptdist}km_{rpttwin}hr wind_{rptdist}km_{rpttwin}hr torn_{rptdist}km_{rpttwin}hr any_{rptdist}km_{rpttwin}hr "
        f"cg_20km_{twin}hr ic_20km_{twin}hr cg.ic_20km_{twin}hr flashes_20km_{twin}hr "
        f"cg_40km_{twin}hr ic_40km_{twin}hr cg.ic_40km_{twin}hr flashes_40km_{twin}hr "
        "--batchnorm "
        "--reg_penalty 0 "
        f" --epoch {epoch} --learning 0.001 --kfold 1".split()
    )

    return args
