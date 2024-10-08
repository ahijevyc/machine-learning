import argparse
import datetime
import glob
from hwtmode.data import decompose_circular_feature
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
from itertools import repeat
import logging
import matplotlib.pyplot as plt
from ml_functions import (
    brier_skill_score,
    configs_match,
    get_argparser,
    get_features,
    get_savedmodel_path,
    load_df,
    predct2,
    rptdist2bool,
)
from multiprocessing import cpu_count, Pool
import numpy as np
import os
import pandas as pd
import pdb
import sklearn
import sys
from tensorflow.keras.models import load_model
import time
import xarray
import yaml

"""
 test neural network(s) in parallel. output truth and predictions from each fit and ensemble mean for each forecast hour
 execcasper --ngpus 13 --mem=50GB # gpus not neeeded for verification
"""


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


parser = get_argparser()

args = parser.parse_args()
logging.info(args)

# Assign arguments to simple-named variables
clobber = args.clobber
debug = args.debug
kfold = args.kfold
nfit = args.nfits
testend = args.testend
teststart = args.teststart
suite = args.suite


if debug:
    logging.basicConfig(level=logging.DEBUG)


### saved model path ###
savedmodel = get_savedmodel_path(args)
logging.info(f"savedmodel={savedmodel}")

for ifold in range(kfold):
    for i in range(0, nfit):
        savedmodel_i = f"{savedmodel}_{i}/{kfold}fold{ifold}"
        assert os.path.exists(savedmodel_i), f"{savedmodel_i} not found"

    nextfit = f"{savedmodel}_{i+1}"
    if os.path.exists(nextfit):
        logging.warning(
            f"next fit exists ({nextfit}). Are you sure nfit only {nfit}?")

##################################

df = load_df(args)

feature_list = get_features(args)

logging.info("convert report distance and flash count to True/False labels")
df = rptdist2bool(df, args)

validtimes = df.valid_time
logging.info(f"range of valid times: {validtimes.min()} - {validtimes.max()}")

# TODO: use valid time to split training and testing sets, here and in Jupyter notebooks and training script(s).
# and possibly other scripts?
logging.info(f"Use initialization times [{teststart}, {testend}) for testing")
before_filtering = len(df)
idx = (teststart <= df.initialization_time) & (
    df.initialization_time < testend)
df = df[idx]
logging.info(
    f"keep {len(df)}/{before_filtering} cases for testing")

itimes = df.initialization_time
teststart = itimes.min()
testend = itimes.max()
ofile = os.path.realpath(
    f"{savedmodel}.{kfold}fold.{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}scores.txt")
assert clobber or not os.path.exists(
    ofile), f"Exiting because output file {ofile} exists. Use --clobber option to override."
logging.info(f"output file will be {ofile}")

# Put "valid_time", "y", and "x" (and some features) in MultiIndex
# so we can group by them later.
# Used here and when calculating ensemble mean.
levels = ["initialization_time", "valid_time", "y", "x",]
df = df.set_index(levels)
feature_levels = ["forecast_hour", "lat", "lon"]
df = df.set_index(feature_levels, drop=False, append=True)
levels = levels + feature_levels

df.info()


def statjob(group, args):
    groupname, Y = group
    logging.info(f"statjob: {groupname}")
    statcurves = (
        "ensmean" in groupname
        and "all" in groupname
        and any([x for x in args.labels if x.startswith("any")])
    )

    # seperate y_pred and labels and drop level 0
    y_pred = Y.xs("y_pred", axis="columns", level=0)
    # labels went from bool to object dtype, so fix it or roc_auc_score will not recognize format
    labels = Y.xs("y_label", axis="columns", level=0).astype(bool)

    bss = brier_skill_score(labels, y_pred)
    base_rate = labels.mean()
    # Default value is np.nan
    # Don't assign Series to auc and aps on same line or they will remain equal even if you change one
    auc = pd.Series(np.nan, index=labels.columns)
    aps = pd.Series(np.nan, index=labels.columns)
    # auc and aps require 2 unique labels, i.e. both True and False
    two = labels.nunique() == 2
    if two.any():
        # average=None returns a metric for each label instead of one group average of all labels
        auc[two] = sklearn.metrics.roc_auc_score(
            labels.loc[:, two], y_pred.loc[:, two], average=None)
        aps[two] = sklearn.metrics.average_precision_score(
            labels.loc[:, two], y_pred.loc[:, two], average=None)
    n = y_pred.count()
    out = pd.DataFrame(
        dict(bss=bss, base_rate=base_rate, auc=auc, aps=aps, n=n))
    out.index.name = "class"
    logging.debug(out)
    if statcurves:
        # use comma. we want a single element, not a list
        anyc, = [x for x in args.labels if x.startswith("any")]
        flashc, = [x for x in args.labels if x.startswith("flash")]
        cgicc, = [x for x in args.labels if x.startswith("cg.ic")]
        cgc, = [x for x in args.labels if x.startswith("cg_")]
        icc, = [x for x in args.labels if x.startswith("ic_")]
        # put more than one event type on same plot
        event_groups = [[anyc, flashc],
                        [anyc, cgicc],
                        [anyc, cgc],
                        [anyc, icc]]

        fig = plt.figure(figsize=(10, 7))
        for event_group in event_groups:
            ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, sharex=ax1)
            ROC_ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
            for event in event_group:
                logging.info(
                    f"{groupname} {event} reliability diagram, histogram, & ROC curve")
                reliability_diagram_obj, = reliability_diagram(
                    ax1, labels[event], y_pred[event])
                counts, bins, patches = count_histogram(
                    ax2, y_pred[event], count_label=False)
                rc = ROC_curve(ROC_ax,
                               labels[event],
                               y_pred[event],
                               fill=False,
                               plabel=False)
            fig.suptitle(f"{suite} {event_group}")
            fig.text(0.5, 0.01, ' '.join(feature_list), wrap=True, fontsize=5)
            ofile = f"{savedmodel}.{event_group}.{groupname}.statcurves{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}.png"
            if not debug:
                fig.savefig(ofile)
                logging.info(os.path.realpath(ofile))
            plt.clf()
    return groupname, out


def applyParallel(dfGrouped, func, args):
    parallel = True
    if parallel:
        with Pool(nfit) as p:
            ret_list = p.starmap(func, [(group, args) for group in dfGrouped])
    else:
        ret_list = [func(group, args) for group in dfGrouped]
    df = pd.concat([x[1] for x in ret_list], keys=[x[0] for x in ret_list])
    return df


index = pd.MultiIndex.from_product(
    [range(kfold), range(nfit)], names=["fold", "fit"])
with Pool(processes=nfit) as p:
    result = p.starmap(predct2, zip(index, repeat(args), repeat(df)))
Y = pd.concat(result, keys=index, names=index.names)

logging.info("average fits for ensmean")
ensmean = Y.groupby(levels).mean()
ensmean = pd.concat([ensmean], keys=["ensmean"], names=["fit"])
ensmean = pd.concat([ensmean], keys=["all"], names=["fold"])

logging.info("concat Y and ensmean")
Y = pd.concat([Y, ensmean], axis="index")
Y = pd.concat([Y], keys=["all"], names=["lat_bin"])
Y = pd.concat([Y], keys=["all"], names=["lon_bin"])

# Aggregate all forecast hours, lat, lon
groupby = ["fit", "fold"]
logging.info(
    f"calculate stats by {groupby} (aggregate all forecast hours, lat, lon)")
# tried as_index=True and group_keys=True but didn't change things. (thought it might keep track of index level names for me)
all_fhr = applyParallel(Y.groupby(groupby), statjob, args)
all_fhr.index.names = (*groupby, "class")
all_fhr = pd.concat([all_fhr], keys=["all"], names=["forecast_hour"])
all_fhr = pd.concat([all_fhr], keys=["all"], names=["lat_bin"])
all_fhr = pd.concat([all_fhr], keys=["all"], names=["lon_bin"])

# Individual forecast hours
groupby = ["fit", "fold", "forecast_hour"]
logging.info(f"calculate stats by {groupby}")
stat = applyParallel(Y.groupby(groupby), statjob, args)
stat.index.names = (*groupby, "class")
stat = pd.concat([stat], keys=["all"], names=["lat_bin"])
stat = pd.concat([stat], keys=["all"], names=["lon_bin"])
# ensure all_fhr and stat have index levels in same order
stat = stat.reorder_levels(all_fhr.index.names)

# Aggregate in forecast_hour and lat/lon blocks with pandas.cut
time_block_hours = 4
cut_time_blocks = pd.cut(
    Y.index.get_level_values("forecast_hour"),
    bins=range(0, max(args.fhr)+1, time_block_hours),
    right=False)
lat_bin = pd.cut(Y.index.get_level_values("lat"), bins=4)
lon_bin = pd.cut(Y.index.get_level_values("lon"), bins=4)
groupby = ["fit", "fold", cut_time_blocks, lat_bin, lon_bin]
logging.info(f"groupby {groupby}")
stat2 = applyParallel(Y.groupby(groupby), statjob, args)
# TODO: hacky; failed to name cut_time_blocks
groupby[-3:] = ("forecast_hour", "lat_bin", "lon_bin")
stat2.index.names = (*groupby, "class")
# ensure all_fhr and stat have index levels in same order
stat2 = stat2.reorder_levels(all_fhr.index.names)


if not debug:
    pd.concat([stat, stat2, all_fhr]).to_csv(ofile)
    logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
