import argparse
import dask.dataframe as dd
import datetime
import glob
from hwtmode.data import decompose_circular_feature
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, configs_match, get_argparser, get_features, load_df, rptdist2bool, get_savedmodel_path
from multiprocessing import cpu_count, Pool
import numpy as np
import os
import pandas as pd
import pdb
import sklearn
import sys
from sklearn.model_selection import KFold
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

logging.info("convert report distance and flash count to True/False labels")
df = rptdist2bool(df, args)

assert set(df.index.names) == set(['valid_time', 'x', 'y']), f"unexpected index names for df {df.index.names}"


# Make initialization_time a MultiIndex level
df = df.set_index("initialization_time", append=True)


validtimes = df.index.get_level_values(level="valid_time")
logging.info(f"range of valid times: {validtimes.min()} - {validtimes.max()}")


# This is done in train_stormrpts_dnn.py. Important to do here too.
logging.info(f"Sort by valid_time")
# Can't ignore_index=True like train_stormrpts_dnn.py cause we need multiindex, but it shouldn't affect order
df = df.sort_index(level="valid_time")


logging.info(f"Use initialization times {teststart} - {testend} for testing")
before_filtering = len(df)
df = df.loc[:, :, :, teststart:testend]
logging.info(
    f"keep {len(df)}/{before_filtering} cases for testing")

itimes = df.index.get_level_values(level="initialization_time")
teststart = itimes.min()
testend = itimes.max()
ofile = os.path.realpath(
    f"{savedmodel}.{kfold}fold.{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}scores.txt")
assert clobber or not os.path.exists(ofile), f"Exiting because output file {ofile} exists. Use --clobber option to override."

logging.info(f"output file will be {ofile}")

# Used to test all columns for NA, but we only care about the feature subset being complete.
# For example, mode probs are not avaiable for fhr=2 but we don't need to drop fhr=2 if
# the other features are complete.
feature_list = get_features(args)
logging.info(
    f"Retain rows where all {len(feature_list)} requested features are present")
beforedropna = len(df)
df = df.dropna(axis="index", subset=feature_list)
logging.info(f"kept {len(df)}/{beforedropna} cases with no NA features")

logging.info(f"Split {len(label_cols)} labels away from predictors")
labels = df[label_cols]  # labels converted to Boolean above

df.info()

# TODO: is this needed?
labels = labels.droplevel("initialization_time")
# TODO: do we really want to change labels without changing df?
# I know there are duplicate obs (labels) for different initialization times that share the same valid time. big deal.
labels = labels[~labels.index.duplicated(keep="first")]

assert labels.sum().all() > 0, f"at least 1 class has no True labels in testing set {labels.sum()}"


columns_before_filtering = df.columns
df = df[feature_list]
logging.info(
    f"dropped {set(columns_before_filtering) - set(df.columns)}")
logging.info(
    f"kept {len(df.columns)}/{len(columns_before_filtering)} features")


if kfold > 1:
    cv = KFold(n_splits=kfold)
    # Convert generator to list. You don't want a generator.
    # Generator depletes after first run of statjob, and if run serially, next time statjob is executed the entire fold loop is skipped.
    cvsplit = list(cv.split(df))
else:
    # Emulate a 1-split KFold object with all cases in test split.
    cvsplit = [([], np.arange(len(df)))]
def predct(i):
    logging.warning(i)
    ifold, thisfit = i
    savedmodel_thisfitfold = f"{savedmodel}_{thisfit}/{kfold}fold{ifold}"
    yl = yaml.load(open(
        os.path.join(savedmodel_thisfitfold, "config.yaml"), "r"),
        Loader=yaml.Loader)
    if "labels" in yl:
        labels = yl["labels"]
        # delete labels so we can make DataFrame from rest of dictionary.
        del (yl["labels"])
    else:
        labels = getattr(yl["args"], "labels")
        
    assert configs_match(
        yl["args"], args
    ), f'this configuration {args} does not match yaml file {yl["args"]}'
    del (yl["args"])
    # scaling values DataFrame as from .describe()
    sv = pd.DataFrame(yl).set_index("columns").T
    if sv.columns.size != df.columns.size:
        logging.error(
            f"size of yaml and features columns differ {sv.columns} {df.columns}"
        )
    assert all(
        sv.columns == df.columns
    ), f"columns {df.columns} don't match when model was trained {sv.columns}"

    logging.info(f"loading {savedmodel_thisfitfold}")
    model = load_model(
        savedmodel_thisfitfold)
    itrain, itest = cvsplit[ifold]
    df_fold = df.iloc[itest]
    norm_features = (df_fold - sv.loc["mean"]) / sv.loc["std"]
    # Grab numpy array of predictions.
    Y = model.predict(norm_features.to_numpy(
        dtype='float32'), batch_size=10000)
    Y = pd.DataFrame(Y, columns=labels, index=df_fold.index)
    return Y

def statjob(group, labels):
    name, y_pred = group
    logging.info(f"statjob: {name}")
    statcurves = name == ("ensmean","all")

    droplevels=list(set(y_pred.index.names) - set(labels.index.names))
    logging.info(f"drop {droplevels} level(s) from y_pred")
    y_pred = y_pred.droplevel(droplevels)
    labels = labels.loc[y_pred.index]
    bss = brier_skill_score(labels, y_pred)
    if name == (0, 0):
        pd.concat([labels,y_pred], axis="columns").to_csv("fit0fold0group.csv")
    base_rate = labels.mean()
    # Default value is np.nan
    # Don't assign Series to auc and aps on same line or they will remain equal even if you change one
    auc = pd.Series(np.nan, index=labels.columns)
    aps = pd.Series(np.nan, index=labels.columns)
    # auc and aps require 2 unique labels, i.e. both True and False
    two = labels.nunique() == 2
    # average=None returns a metric for each label
    auc[two] = sklearn.metrics.roc_auc_score(          labels.loc[:,two], y_pred.loc[:,two], average=None)
    aps[two] = sklearn.metrics.average_precision_score(labels.loc[:,two], y_pred.loc[:,two], average=None)
    n = y_pred.count()
    out = pd.DataFrame(dict(bss=bss, base_rate=base_rate, auc=auc, aps=aps, n=n))
    out.index.name = "class"
    logging.debug(out)
    if statcurves:
        fig = plt.figure(figsize=(10,7))
        for event in labels.columns:
            logging.info(
                f"{event} reliability diagram, histogram, & ROC curve")
            ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, sharex=ax1)
            ROC_ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
            reliability_diagram_obj, = reliability_diagram(
                ax1, labels[event], y_pred[event])
            counts, bins, patches = count_histogram(ax2, y_pred[event])
            rc = ROC_curve(ROC_ax,
                           labels[event],
                           y_pred[event],
                           fill=False,
                           plabel=False)
            fig.suptitle(f"{suite} {event}")
            fig.text(0.5, 0.01, ' '.join(feature_list), wrap=True, fontsize=5)
            ofile = f"{savedmodel}.{event}.ensmean.statcurves{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}.png"
            if not debug:
                fig.savefig(ofile)
                logging.info(os.path.realpath(ofile))
            plt.clf()
    return name, out

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
    result = p.map(predct, index)
y_preds = pd.concat(result, keys=index, names=index.names)

logging.info("average for ensmean") 
ensmean = y_preds.groupby(["valid_time", "y", "x", "initialization_time"]).mean()
ensmean = pd.concat([ensmean], keys=["ensmean"], names=["fit"])
ensmean = pd.concat([ensmean], keys=["all"], names=["fold"])

logging.info("concat y_preds and ensmean")
y_preds = pd.concat([y_preds, ensmean], axis="index")


groupby=["fit","fold"]
logging.info(f"calculate stats by {groupby} (aggregate all forecast hours)")
# TODO: why does bss for forecast_hour = "all" differ from test_stormrpts_dnn.py?
all_fhr = applyParallel(y_preds.groupby(groupby), statjob, labels) # tried as_index=True and group_keys=True but didn't change things. (thought it might keep track of index level names for me)
all_fhr.index.names=(*groupby,"class")
all_fhr = pd.concat([all_fhr], keys=["all"], names=["forecast_hour"])


groupby=["fit","fold","forecast_hour"]
logging.info(f"calculate stats by {groupby}")
forecast_leadtime = y_preds.index.get_level_values("valid_time") - y_preds.index.get_level_values("initialization_time")
y_preds["forecast_hour"] = (forecast_leadtime / pd.Timedelta(hours=1)).astype(int)
# don't want statjob to treat `forecast_hour` as another label like `wind_40km_1hr`
y_preds = y_preds.set_index("forecast_hour", append=True)
stat = applyParallel(y_preds.groupby(groupby), statjob, labels)
stat.index.names=(*groupby,"class")
# ensure all_fhr and stat have index levels in same order
stat = stat.reorder_levels(all_fhr.index.names)
if not debug:
    pd.concat([stat, all_fhr]).to_csv(ofile)
    logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
