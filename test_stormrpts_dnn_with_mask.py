import argparse
import datetime
import glob
from hwtmode.data import decompose_circular_feature
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, configs_match, get_argparser, get_features, load_df, rptdist2bool, get_savedmodel_path
import multiprocessing
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
 Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
 execcasper --ngpus 13 --mem=50GB # gpus not neeeded for verification
"""


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


parser = get_argparser()
parser.add_argument('field', type=str,
                    help="feature/column/predictor to base mask on")
parser.add_argument('thresh', type=float,
                    help="field threshold (less than this / greater than or equal to this)")

args = parser.parse_args()
logging.info(args)

# Assign arguments to simple-named variables
clobber = args.clobber
debug = args.debug
field = args.field
kfold = args.kfold
nfit = args.nfits
nprocs = args.nprocs
rptdist = args.rptdist
testend = args.testend
thresh = args.thresh
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
df, label_cols = rptdist2bool(df, args)

assert set(df.index.names) == set(['valid_time', 'x', 'y']), f"unexpected index names for df {df.index.names}"


# Make initialization_time a MultiIndex level
df = df.set_index("initialization_time", append=True)


# Define a column level "ctype" based on whether it is in label_cols or not.
ctype = np.array(["feature"] * df.columns.size)

ctype[df.columns.isin(label_cols)] = "label"

# TODO: add "unused" for predictors that are in the parquet file but not the predictor suite.

df.columns = pd.MultiIndex.from_arrays(
    [df.columns, ctype], names=["name", "ctype"])


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
    f"{savedmodel}.{kfold}fold.{field}{thresh}.{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}scores.txt")
if not clobber and os.path.exists(ofile):
    logging.info(
        f"Exiting because output file {ofile} exists. Use --clobber option to override.")
    sys.exit(0)

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

logging.info("Define mask and append to index")

mask = pd.Series(np.select([df[(field, "feature")] >= thresh], [
                 f"{field}>={thresh*100}%"], f"{field}<{thresh*100}%"), name="mask")  # Mask is a string like DNN_1_Supercell>=10%
df = df.set_index(mask, append=True)

df.info()


labels_sum = df.xs("label", axis=1, level="ctype").groupby(level="mask").sum()
assert labels_sum.all().all() > 0, "at least 1 class has no True labels in testing set"
labels_sum


before_filtering = len(df.columns)
# Keep column if it is a feature of this suite or a label.
tokeep = [x for x in df.columns if x[0]
          in get_features(args) or x[1] == "label"]
df = df[tokeep]
logging.info(f"keeping {len(df.columns)}/{before_filtering} predictors")


if kfold > 1:
    cv = KFold(n_splits=kfold)
    # Convert generator to list. You don't want a generator.
    # Generator depletes after first run of statjob, and if run serially, next time statjob is executed the entire fold loop is skipped.
    cvsplit = list(cv.split(df))
else:
    # Emulate a 1-split KFold object with all cases in test split.
    cvsplit = [([], np.arange(len(df)))]


def statjob(fhr, statcurves=None):
    if statcurves is None:
        statcurves = fhr == "all"
    if statcurves:
        fig = plt.figure(figsize=(10, 7))
    # this_fhr for all cases, not just one fold
    if fhr == "all":
        this_fhr = ~df["forecast_hour"].isna()  # all finite forecast hours
    else:
        this_fhr = df["forecast_hour"] == fhr  # Just this fhr
    logging.debug(f"{len(this_fhr)} {fhr} fhr model predictions")
    y_preds = pd.DataFrame()
    stattxt = ""
    for ifold, (itrain, itest) in enumerate(cvsplit):
        df_fold = df.iloc[itest]
        if fhr == "all":
            # all finite forecast hours
            this_fold_fhr = ~df_fold[("forecast_hour", "feature")].isna()
        else:
            this_fold_fhr = df_fold[("forecast_hour",
                                     "feature")] == fhr  # Just this fhr
        df_fold_fhr = df_fold[this_fold_fhr]

        features = df_fold_fhr.xs("feature", axis=1, level="ctype")
        labels = df_fold_fhr.xs("label", axis=1, level="ctype")[label_cols]

        for thisfit in range(nfit):
            savedmodel_thisfitfold = f"{savedmodel}_{thisfit}/{kfold}fold{ifold}"
            logging.debug(f"checking {savedmodel_thisfitfold} column order")
            # yaml.Loader is not safe (yaml.FullLoader is) but legacy Loader handles argparse.namespace object.
            yl = yaml.load(open(
                os.path.join(savedmodel_thisfitfold, "config.yaml"), "r"),
                Loader=yaml.Loader)
            yl_labels = yl["labels"]
            # delete labels so we can make DataFrame from rest of dictionary.
            del (yl["labels"])
            assert configs_match(
                yl["args"], args
            ), f'this configuration {args} does not match yaml file {yl["args"]}'
            del (yl["args"])
            assert all(
                yl_labels == labels.columns
            ), f"labels {labels.columns} don't match when model was trained {yl_labels}"

            # scaling values DataFrame as from .describe()
            sv = pd.DataFrame(yl).set_index("columns").T
            if sv.columns.size != features.columns.size:
                logging.error(
                    f"size of yaml and features columns differ {sv.columns} {features.columns}"
                )
                pdb.set_trace()
            if not all(sv.columns == features.columns):
                logging.info(f"reordering columns")
                features = features.reindex(columns=sv.columns)
            assert all(
                sv.columns == features.columns
            ), f"columns {features.columns} don't match when model was trained {sv.columns}"
            logging.info(f"loading {savedmodel_thisfitfold}")
            model = load_model(
                savedmodel_thisfitfold,
                custom_objects=dict(brier_skill_score=brier_skill_score))
            logging.info(
                f"predicting fhr {fhr}  fit {thisfit}  fold{ifold}...")
            norm_features = (features - sv.loc["mean"]) / sv.loc["std"]
            # Grab numpy array of predictions.
            Y = model.predict(norm_features.to_numpy(dtype='float32'), batch_size=20000)
            # Put prediction numpy array into DataFrame with index (row) and column labels.
            Y = pd.DataFrame(Y, columns=labels.columns, index=features.index)
            # for each report type
            for rpt_type in labels.columns:
                for mask, labels_fhr in labels[rpt_type].groupby(level="mask"):
                    y_pred = Y.xs(mask, level="mask")[
                        rpt_type]  # grab this particular report type
                    bss = brier_skill_score(labels_fhr, y_pred)
                    base_rate = labels_fhr.mean()
                    auc = sklearn.metrics.roc_auc_score(
                        labels_fhr, y_pred) if labels_fhr.any() else np.nan
                    aps = sklearn.metrics.average_precision_score(
                        labels_fhr, y_pred)
                    n = len(y_pred)
                    logging.debug(
                        f"{rpt_type} fit={thisfit} fold={ifold} fhr={fhr} mask={mask} {bss} {base_rate} {auc} {aps} {n}"
                    )
                    stattxt += f"{rpt_type},{thisfit},{ifold},{fhr},{mask},{bss},{base_rate},{auc},{aps},{n}\n"
            # prepend "fit" level to multilevel DataFrame
            Y = pd.concat([Y], keys=[thisfit], names=["fit"])
            # prepend "fold" level
            Y = pd.concat([Y], keys=[ifold], names=["fold"])
            # concatenate this fit/fold to the y_preds DataFrame
            y_preds = pd.concat([y_preds, Y], axis="index")
    # I may have overlapping valid_times from different init_times like fhr=1 from today and fhr=25 from previous day
    # average probability over all nfits initialized at initialization_time and valid at valid_time
    ensmean = y_preds.groupby(level=[
        "valid_time", "y", "x", "initialization_time"
    ]).mean()
    assert "fit" not in ensmean.index.names, "fit should not be a MultiIndex level of ensmean, the average probability over nfits."
    # for statistic curves plot file name
    logging.debug(f"getting ensmean bss, base rate, auc, aps, n")
    for rpt_type in labels.columns:
        for mask, labels_fhr in labels[rpt_type].groupby(level="mask"):
            y_pred = ensmean.xs(mask, level="mask")[rpt_type]
            bss = brier_skill_score(labels_fhr, y_pred)
            base_rate = labels_fhr.mean()
            auc = sklearn.metrics.roc_auc_score(
                labels_fhr, y_pred) if labels_fhr.any() else np.nan
            # average_precision_score
            aps = sklearn.metrics.average_precision_score(labels_fhr, y_pred)
            n = len(y_pred)
            logging.info(
                f"{rpt_type} ensmean fold=all fhr={fhr} {mask} {bss} {base_rate} {auc} {aps} {n}"
            )
            stattxt += f"{rpt_type},ensmean,all,{fhr},{mask},{bss},{base_rate},{auc},{aps},{n}\n"
            if statcurves:
                logging.info(
                    f"{rpt_type} reliability diagram, histogram, & ROC curve")
                ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, sharex=ax1)
                ROC_ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
                reliability_diagram_obj, = reliability_diagram(
                    ax1, labels_fhr, y_pred)
                counts, bins, patches = count_histogram(ax2, y_pred)
                rc = ROC_curve(ROC_ax,
                               labels_fhr,
                               y_pred,
                               fill=False,
                               plabel=False)
                fig.suptitle(f"{suite} {rpt_type}")
                fig.text(0.5,
                         0.01,
                         ' '.join(features.columns),
                         wrap=True,
                         fontsize=5)
                ofile = f"{thissavedmodel}.{rpt_type}.{mask}.statcurves{teststart.strftime('%Y%m%d%H')}-{testend.strftime('%Y%m%d%H')}.f{fhr}.png"
                fig.savefig(ofile)
                logging.info(os.path.realpath(ofile))
                plt.clf()
    return stattxt


if debug:
    stattxt = statjob('all')
    pdb.set_trace()
    sys.exit(0)

# allow for datasets like storm mode probabilities, that don't start at fhr=1
fhrs = df[("forecast_hour", "feature")].unique()
fhrs = list(fhrs)
fhrs.insert(0, "all")  # put "all" first because it takes the longest
if nprocs > 1:
    # Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
    # execcasper --ncpus=13 --mem=50GB # gpus not neeeded for verification
    pool = multiprocessing.Pool(processes=nprocs)
    # used to set chunksize > 1, but because "all" takes so much longer, the chunk that includes "all" is screwed. Use default chunksize=1
    # Tried imap_unordered, but I like reproducability. Plus I could not sort fhrs
    # when string "all" was mixed with integers.
    data = pool.map(statjob, fhrs)
    pool.close()
else:
    data = []
    for fhr in fhrs:
        data.append(statjob(fhr))

with open(ofile, "w") as fh:
    fh.write('class,fit,fold,forecast_hour,mask,bss,base_rate,auc,aps,n\n')
    fh.write(''.join(data))

logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
