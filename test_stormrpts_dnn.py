import argparse
import dask.dataframe as dd
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
        predct,
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
label_cols = args.labels
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
idx = (teststart <= df.initialization_time) & (df.initialization_time < testend)
df = df[idx]
logging.info(
    f"keep {len(df)}/{before_filtering} cases for testing")

itimes = df.initialization_time
teststart = itimes.min()
testend = itimes.max()
ofile = os.path.realpath(
    f"{savedmodel}.{kfold}fold.{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}scores.txt")
assert clobber or not os.path.exists(ofile), f"Exiting because output file {ofile} exists. Use --clobber option to override."
logging.info(f"output file will be {ofile}")

# Put "valid_time", "y", and "x" in MultiIndex so we can group by them later.
# Otherwise they will be lost when you subset columns by feature_list.
# Used here and when calculating ensemble mean.
levels = ["initialization_time", "valid_time", "y", "x"]
df = df.set_index(levels)


logging.info(f"Split {len(label_cols)} labels away from predictors")
labels = df[label_cols]  # labels converted to Boolean above

df.info()

assert labels.sum().all() > 0, f"at least 1 class has no True labels in testing set {labels.sum()}"


columns_before_filtering = df.columns
df = df[feature_list]
logging.info(
    f"dropped {set(columns_before_filtering) - set(df.columns)}")
logging.info(
    f"kept {len(df.columns)}/{len(columns_before_filtering)} features")


def statjob(group, labels):
    groupname, y_pred = group
    logging.info(f"statjob: {groupname}")
    statcurves = (
            "ensmean" in groupname
            and "all" in groupname
            and any([x for x in label_cols if x.startswith("any")])
    )


    droplevels=list(set(y_pred.index.names) - set(labels.index.names))
    logging.debug(f"drop {droplevels} level(s) from y_pred")
    y_pred = y_pred.droplevel(droplevels)
    labels = labels.loc[y_pred.index]
    bss = brier_skill_score(labels, y_pred)
    base_rate = labels.mean()
    # Default value is np.nan
    # Don't assign Series to auc and aps on same line or they will remain equal even if you change one
    auc = pd.Series(np.nan, index=labels.columns)
    aps = pd.Series(np.nan, index=labels.columns)
    # auc and aps require 2 unique labels, i.e. both True and False
    two = labels.nunique() == 2
    # average=None returns a metric for each label instead of one group average of all labels
    auc[two] = sklearn.metrics.roc_auc_score(          labels.loc[:,two], y_pred.loc[:,two], average=None)
    aps[two] = sklearn.metrics.average_precision_score(labels.loc[:,two], y_pred.loc[:,two], average=None)
    n = y_pred.count()
    out = pd.DataFrame(dict(bss=bss, base_rate=base_rate, auc=auc, aps=aps, n=n))
    out.index.name = "class"
    logging.debug(out)
    if statcurves:
        # use comma. we want a single element, not a list
        anyc, = [x for x in label_cols if x.startswith("any")]
        flashc, = [x for x in label_cols if x.startswith("flash")]
        cgicc, = [x for x in label_cols if x.startswith("cg.ic")]
        cgc, = [x for x in label_cols if x.startswith("cg_")]
        icc, = [x for x in label_cols if x.startswith("ic_")]
        # put more than one event type on same plot
        event_groups = [[anyc, flashc],
                        [anyc, cgicc],
                        [anyc, cgc],
                        [anyc, icc]]

        fig = plt.figure(figsize=(10,7))
        for event_group in event_groups: 
            ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, sharex=ax1)
            ROC_ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
            for event in event_group:
                logging.info(
                    f"{groupname} {event} reliability diagram, histogram, & ROC curve")
                reliability_diagram_obj, = reliability_diagram(
                    ax1, labels[event], y_pred[event])
                counts, bins, patches = count_histogram(ax2, y_pred[event], count_label=False)
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

def applyParallel(dfGrouped, func, labels):
    parallel = True
    if parallel:
        with Pool(nfit) as p:
            ret_list = p.starmap(func, [(group,labels) for group in dfGrouped])
    else:
        ret_list = [func(group,labels) for group in dfGrouped]
    df = pd.concat([x[1] for x in ret_list], keys=[x[0] for x in ret_list])
    return df


index = pd.MultiIndex.from_product([range(kfold), range(nfit)], names=["fold","fit"])
with Pool(processes=nfit) as p:
    result = p.starmap(predct, zip(index, repeat(args), repeat(df)))
y_preds = pd.concat(result, keys=index, names=index.names)

logging.info("average for ensmean") 
ensmean = y_preds.groupby(levels).mean()
ensmean = pd.concat([ensmean], keys=["ensmean"], names=["fit"])
ensmean = pd.concat([ensmean], keys=["all"], names=["fold"])

logging.info("concat y_preds and ensmean")
y_preds = pd.concat([y_preds, ensmean], axis="index")

forecast_leadtime = y_preds.index.get_level_values("valid_time") - y_preds.index.get_level_values("initialization_time")
forecast_leadtime = (forecast_leadtime / pd.Timedelta(hours=1)).astype(int)

### Aggregate all forecast hours
groupby=["fit","fold"]
logging.info(f"calculate stats by {groupby} (aggregate all forecast hours)")
all_fhr = applyParallel(y_preds.groupby(groupby), statjob, labels) # tried as_index=True and group_keys=True but didn't change things. (thought it might keep track of index level names for me)
all_fhr.index.names=(*groupby,"class")
all_fhr = pd.concat([all_fhr], keys=["all"], names=["forecast_hour"])

### Individual forecast hours
groupby=["fit","fold","forecast_hour"]
logging.info(f"calculate stats by {groupby}")
y_preds["forecast_hour"] = forecast_leadtime
# don't want statjob to treat `forecast_hour` as another label like `wind_40km_1hr`
y_preds = y_preds.set_index("forecast_hour", append=True)
stat = applyParallel(y_preds.groupby(groupby), statjob, labels)
stat.index.names=(*groupby,"class")
# ensure all_fhr and stat have index levels in same order
stat = stat.reorder_levels(all_fhr.index.names)

### Aggregate in forecast_hour Time Blocks with pandas.cut
time_block_hours = 4
groupby=["fit","fold","forecast_hour"]
cut_time_blocks = pd.cut(
        forecast_leadtime, 
        bins = range(0, max(args.fhr)+1, time_block_hours),
        right=False)
y_preds["forecast_hour"] = cut_time_blocks
y_preds = y_preds.droplevel("forecast_hour") # Don't want old and new index level "forecast_hour"
y_preds = y_preds.set_index("forecast_hour", append=True)
stat2 = applyParallel(y_preds.groupby(groupby), statjob, labels)
stat2.index.names=(*groupby,"class")
# ensure all_fhr and stat have index levels in same order
stat2 = stat2.reorder_levels(all_fhr.index.names)


if not debug:
    pd.concat([stat, stat2, all_fhr]).to_csv(ofile)
    logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
