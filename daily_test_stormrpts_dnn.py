"""
 copied form test_stormrpts_dnn.py Nov 8 2023
 simplified
 other than parallel processing, no group by fit or fold
 just ensmean by valid date
"""
import argparse
import datetime
import glob
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
from itertools import repeat
import logging
import matplotlib.pyplot as plt
from ml_functions import (
    brier_skill_score,
    get_argparser,
    get_features,
    get_savedmodel_path,
    load_df,
    predct2,
    rptdist2bool,
)
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pdb
import sklearn
import sys
import time
import xarray

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
    f"{savedmodel}.{kfold}fold.{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}dailyscores.txt")
assert clobber or not os.path.exists(
    ofile), f"Exiting because output file {ofile} exists. Use --clobber option to override."
logging.info(f"output file will be {ofile}")


logging.warning("fhr 12-20")
beforefilter = len(df)
df = df[(df.forecast_hour >= 12) & ( df.forecast_hour <= 20)]
logging.warning(f"kept {len(df)}/{beforefilter} cases")

# Put "valid_time", "y", and "x" (and some features) in MultiIndex
# so we can group by them later.
# Used here and when calculating ensemble mean.
levels = ["initialization_time", "valid_time", "y", "x",]
df = df.set_index(levels)

df.info()


def statjob(group, args):
    groupname, Y = group
    logging.info(f"statjob: {groupname}")

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


# Use model to predict test cases.
# Run fits and folds in parallel.
index = pd.MultiIndex.from_product(
    [range(kfold), range(nfit)], names=["fold", "fit"])
with Pool(processes=nfit) as p:
    result = p.starmap(predct2, zip(index, repeat(args), repeat(df)))
Y = pd.concat(result, keys=index, names=index.names)

logging.info("average fits for ensmean")
ensmean = Y.groupby(levels).mean()

# Aggregate by valid date 
ensmean["valid_date"] = ensmean.index.get_level_values("valid_time").date # date part without time and tzone
ensmean = ensmean.set_index("valid_date", append=True)
groupby = "valid_date"
logging.info(f"groupby {groupby}")
stat = applyParallel(ensmean.groupby(groupby), statjob, args)
stat.index.names = (groupby, "class")

if debug:
    pdb.set_trace()
else:
    stat.to_csv(ofile)
    logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
