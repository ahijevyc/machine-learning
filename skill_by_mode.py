import argparse
import datetime
import glob
from hwtmode.data import decompose_circular_feature
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, get_argparser, get_features, get_glm, rptdist2bool, savedmodel_default
import multiprocessing
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import sklearn
import sys
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
import time
import xarray
import yaml

"""
 test neural network(s) in parallel. output truth and predictions from each member and ensemble mean for each forecast hour
 Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
 execcasper --ngpus 13 --mem=50GB # gpus not neeeded for verification
"""


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def configs_match(ylargs, args):
    if args.kfold > 1:
        # once I started KFold. Training and testing cases are all before train_test_split_time with KFold.
        pass
    else:
        assert ylargs.splittime == args.splittime, f"yaml train_test_split_time {ylargs.splittime} does not match value from this script {args.splittime}"
    for key in ["batchnorm", "batchsize", "debug", "dropout", "epochs", "flash", "glm", "kfold", "layers", "learning_rate", "model", "neurons",
                "optimizer", "reg_penalty", "rptdist", "suite", "twin"]:
        if key == "debug" and debug:
            continue  # if running in debug mode, don't require debug in yaml file to match
        assert getattr(ylargs, key) == getattr(
            args, key), f'this script {key} {getattr(args,key)} does not match yaml {key} {getattr(ylargs,key)}'

    return True


parser = get_argparser()
parser.add_argument('--nprocs', type=int, default=0,
                    help="verify this many forecast hours in parallel")
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
flash = args.flash
glm = args.glm
kfold = args.kfold
model = args.model
nfit = args.nfits
nprocs = args.nprocs
rptdist = args.rptdist
savedmodel = args.savedmodel
thresh = args.thresh
train_test_split_time = args.splittime
suite = args.suite
twin = args.twin


if debug:
    logging.basicConfig(level=logging.DEBUG)


### saved model name ###
if savedmodel:
    pass
else:
    # use model trained on f01-f48 regardless of the hour you are testing
    savedmodel = savedmodel_default(args, fhr_str='f01-f48')
logging.info(f"savedmodel={savedmodel}")


for ifold in range(kfold):
    for i in range(0, nfit):
        savedmodel_i = f"nn/nn_{savedmodel}_{i}/{kfold}fold{ifold}"
        assert os.path.exists(savedmodel_i), f"{savedmodel_i} not found"

    nextfit = f"nn/nn_{savedmodel}_{i+1}"
    if os.path.exists(nextfit):
        logging.warning(
            f"next fit exists ({nextfit}). Are you sure nfit only {nfit}?")


odir = os.path.join("/glade/scratch", os.getenv("USER"))
if glm:
    odir = os.path.join(odir, "GLM")
if not os.path.exists(odir):
    logging.info(f"making directory {odir}")
    os.mkdir(odir)


ofile = os.path.realpath(
    f"nn/nn_{savedmodel}.{kfold}fold.{field}{thresh}.scores.txt")
if not clobber and os.path.exists(ofile):
    logging.info(
        f"Exiting because output file {ofile} exists. Use --clobber option to override.")
    sys.exit(0)

logging.info(f"output file will be {ofile}")

##################################


logging.info(f"Read {model} predictors")
if model == "HRRR":
    ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.par'
    if debug:
        ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.fastdebug.par'
    scalingfile = "/glade/work/ahijevyc/NSC_objects/HRRR/scaling_values_all_HRRRX.pk"
    nfhr = 48
elif model == "NSC3km-12sec":
    ifile0 = f'{model}.par'
    if debug:
        ifile0 = f'/glade/work/ahijevyc/NSC_objects/fastdebug.par'
    scalingfile = f"scaling_values_{model}_{train_test_split_time:%Y%m%d_%H%M}.pk"
    nfhr = 36


if os.path.exists(ifile0):
    logging.info(f'reading {ifile0}')
    df = pd.read_parquet(ifile0, engine="pyarrow")
else:
    logging.error(f"why is there no parquet file for {model}?")
    logging.error(
        f"Do you need to run train_stormrpts_dnn.py to make {ifile0}?")
    sys.exit(1)


df, rptcols = rptdist2bool(df, rptdist, twin)

if glm:
    df["flashes"] = df["flashes"] >= flash
    rptcols.append("flashes")

# This script expects MultiIndex valid_time, projection_y_coordinate, projection_x_coordinate (t, ew, ns)
if df.index.names == ["valid_time", "projection_y_coordinate", "projection_x_coordinate"]:
    pass
else:
    xs = df.index.get_level_values(level="x")
    ys = df.index.get_level_values(level="y")
    if xs.min() == 21 and xs.max() == 80 and ys.min() == 12 and ys.max() == 46:
        rdict = {"y": "projection_x_coordinate",
                 "x": "projection_y_coordinate"}
        logging.info(f"renaming axes {rdict}")
        # NSC3km-12sec saved as forecast_hour, y, x
        df = df.rename_axis(index=rdict)
    elif ys.min() == 21 and ys.max() == 80 and xs.min() == 12 and xs.max() == 46:
        rdict = {"x": "projection_x_coordinate",
                 "y": "projection_y_coordinate"}
        logging.info(f"renaming axes {rdict}")
        # NSC3km-12sec saved as forecast_hour, y, x
        df = df.rename_axis(index=rdict)
    else:
        logging.error(
            "unexpected x and y coordinates. check mask, training script, parquet file...")
        sys.exit(1)


assert set(df.index.names) == set(['valid_time', 'projection_x_coordinate',
                                   'projection_y_coordinate']), f"unexpected index names for df {df.index.names}"

# This script expects "forecast_hour" spelled out.
df = df.rename(columns={"fhr": "forecast_hour",
               "init_time": "initialization_time"})

# Make initialization_time a MultiIndex level
df = df.set_index("initialization_time", append=True)

# Define column level "label" based on whether it is in rptcols or not.
df = df.T.set_index(df.columns.isin(rptcols), append=True).T
df.columns = df.columns.rename("label", level=1)

validtimes = df.index.get_level_values(level="valid_time")
logging.info(f"range of valid times: {validtimes.min()} - {validtimes.max()}")


# This is done in train_stormrpts_dnn.py. Important to do here too.
logging.info(f"Sort by valid_time")
# Can't ignore_index=True like train_stormrpts_dnn.py cause we need multiindex, but it shouldn't affect order
df = df.sort_index(level="valid_time")


logging.info(f"Drop initialization times at or after {train_test_split_time}")
before_filtering = len(df)
df = df.loc[:, :, :, train_test_split_time:]
logging.info(
    f"keep {len(df)}/{before_filtering} cases with init times at or later than {train_test_split_time}")

logging.info("Add mask to index")
df["mask"] = df[field] >= thresh
df = df.set_index("mask", append=True)

df.info()

# TODO: figure out syntax for this
# print(labels.sum())
#assert labels.sum().all() > 0, "at least 1 class has no True labels in testing set"


before_filtering = len(df.columns)
# Keep column if it is a feature of this suite or a label.
tokeep = [x for x in df.columns if x[0] in get_features(args) or x[1]]
df = df[tokeep]
logging.info(f"keeping {len(df.columns)}/{before_filtering} predictors")


logging.info(f"normalize with training cases mean and std in {scalingfile}")
# conda activate tf if AttributeError: Can't get attribute 'new_block' on...
sv = pd.read_pickle(scalingfile)
if "fhr" in sv:
    logging.info("change fhr to forecast_hour in scaling DataFrame")
    sv = sv.rename(columns={"fhr": "forecast_hour"})

# You might have scaling factors for columns that you dropped already, like -N7 columns.
extra_sv_columns = set(sv.columns) - set(df.columns.get_level_values(level=0))
if extra_sv_columns:
    logging.info(
        f"dropping {len(extra_sv_columns)} extra scaling factor columns {extra_sv_columns}")
    sv = sv.drop(columns=extra_sv_columns)

if kfold > 1:
    cv = KFold(n_splits=kfold)
    cvsplit = cv.split(df)
else:
    # Emulate a 1-split KFold object.
    # Put all cases in test split. They are already after train_test_split_time.
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
    # test model
    y_preds = pd.DataFrame()
    stattxt = ""
    for ifold, (train_index, test_index) in enumerate(cvsplit):
        df_fold = df.iloc[test_index]
        if fhr == "all":
            # all finite forecast hours
            this_fold_fhr = ~df_fold["forecast_hour"].isna()
        else:
            this_fold_fhr = df_fold["forecast_hour"] == fhr  # Just this fhr
        df_fold_fhr = df_fold[this_fold_fhr]

        for imask in [True, False]:
            df_fold_fhr_mask = df_fold_fhr.xs(imask, level="mask")

            # Tried normalizing the whole DataFrame df before this function, but you don't want to normalize fhr first. You need them as integers.
            logging.debug(f"normalize fhr={fhr}")
            pdb.set_trace()
            df_fold_fhr_mask = (df_fold_fhr_mask -
                                sv.loc["mean"]) / sv.loc["std"]

            for thisfit in range(nfit):
                savedmodel_thisfitfold = f"nn/nn_{savedmodel}_{thisfit}/{kfold}fold{ifold}"
                logging.info(f"checking {savedmodel_thisfitfold} column order")
                # not safe (yaml.FullLoader is) but legacy Loader handles argparse.namespace object.
                yl = yaml.load(open(os.path.join(
                    savedmodel_thisfitfold, "config.yaml"), "r"), Loader=yaml.Loader)
                yl_labels = yl["labels"]
                # delete labels so we can make DataFrame from rest of dictionary.
                del(yl["labels"])
                assert configs_match(
                    yl["args"], args), f'this configuration {args} does not match yaml file {yl["args"]}'
                del(yl["args"])
                assert all(yl_labels == df_fold_fhr_mask.xs(
                    True, axis=1, level="label").columns), f"labels don't match when model was trained {yl_labels}"
                yl = pd.DataFrame(yl).set_index("columns").T
                if "fhr" in yl:
                    logging.info("Rename fhr to forecast_hour in yaml columns")
                    yl = yl.rename(columns={"fhr": "forecast_hour"})
                if len(yl.columns) != len(df_fold_fhr_mask.columns):
                    print(
                        f"length of yaml and df_fold_fhr_mask columns differ {yl.columns} {df_fold_fhr_mask.columns}")
                    pdb.set_trace()
                assert np.isclose(yl.reindex(columns=sv.columns), sv.loc[yl.index]).all(
                ), f"pickle and yaml scaling factors don't match up\n{sv}\n{yl}"
                if not all(yl.columns == df_fold_fhr_mask.columns):
                    logging.info(f"reordering columns")
                    df_fold_fhr_mask = df_fold_fhr_mask.reindex(
                        columns=yl.columns)
                assert all(
                    yl.columns == df_fold_fhr_mask.columns), f"columns {df.columns} don't match when model was trained {columns}"
                logging.info(f"loading {savedmodel_thisfitfold}")
                model = load_model(savedmodel_thisfitfold, custom_objects=dict(
                    brier_skill_score=brier_skill_score))
                logging.info(
                    f"predicting fhr {fhr}  fit {thisfit}  fold{ifold}...")
                # Grab numpy array of predictions.
                Y = model.predict(df_fold_fhr_mask.xs(
                    False, axis=1, level="label").to_numpy(dtype='float32'))
                # Put prediction numpy array into DataFrame with index (row) and column labels.
                Y = pd.DataFrame(Y, columns=df_fold_fhr_mask.xs(
                    True, axis=1, level="label").columns, index=df_fold_fhr_mask.index)
                # for each report type
                for rpt_type in df_fold_fhr_mask.xs(True, axis=1, level="label").columns:
                    y_pred = Y[rpt_type]  # grab this particular report type
                    labels_fhr = df_fold_fhr_mask.xs(
                        True, axis=1, level="label")
                    bss = brier_skill_score(labels_fhr, y_pred)
                    base_rate = labels_fhr.mean()
                    auc = sklearn.metrics.roc_auc_score(
                        labels_fhr, y_pred) if labels_fhr.any() else np.nan
                    aps = sklearn.metrics.average_precision_score(
                        labels_fhr, y_pred)
                    logging.info(
                        f"{rpt_type} fit={thisfit} fold={ifold} fhr={fhr} mask={imask} {bss} {base_rate} {auc} {aps}")
                    stattxt += f"{rpt_type},{thisfit},{ifold},{fhr},{imask},{bss},{base_rate},{auc},{aps}\n"
                # prepend "fit" level to multilevel DataFrame
                Y = pd.concat([Y], keys=[thisfit], names=["fit"])
                # prepend "fold" level
                Y = pd.concat([Y], keys=[ifold], names=["fold"])
                # prepend "mask" level
                Y = pd.concat([Y], keys=[imask], names=["mask"])
                # append this fit/fold to the y_preds DataFrame
                y_preds = y_preds.append(Y)
    # I may have overlapping valid_times from different init_times like fhr=1 from today and fhr=25 from previous day
    # average probability over all nfits initialized at initialization_time and valid at valid_time
    ensmean = y_preds.groupby(level=["mask", "valid_time", "projection_y_coordinate",
                              "projection_x_coordinate", "initialization_time"]).mean()
    assert "fit" not in ensmean.index.names, "fit should not be a MultiIndex level of ensmean, the average probability over nfits."
    # for statistic curves plot file name
    thissavedmodel = savedmodel.replace('f01-f48', f'f{fhr}')
    logging.debug(f"getting ensmean bss, base rate, auc, aps")
    for imask in [True, False]:
        for rpt_type in labels.columns:
            y_pred = ensmean[rpt_type]
            pdb.set_trace()
            labels_fhr = labels.loc[this_fhr, rpt_type]
            assert labels_fhr.index.equals(
                y_pred.index), f'{rpt_type} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
            bss = brier_skill_score(labels_fhr, y_pred)
            base_rate = labels_fhr.mean()
            auc = sklearn.metrics.roc_auc_score(
                labels_fhr, y_pred) if labels_fhr.any() else np.nan
            # average_precision_score
            aps = sklearn.metrics.average_precision_score(labels_fhr, y_pred)
            logging.info(
                f"{rpt_type} ensmean fold=all fhr={fhr} {bss} {base_rate} {auc} {aps}")
            stattxt += f"{rpt_type},ensmean,all,{fhr},{bss},{base_rate},{auc},{aps}\n"
            if statcurves:
                logging.info(
                    f"{rpt_type} reliability diagram, histogram, & ROC curve")
                ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, sharex=ax1)
                ROC_ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
                reliability_diagram_obj, = reliability_diagram(
                    ax1, labels_fhr, y_pred)
                counts, bins, patches = count_histogram(ax2, y_pred)
                rc = ROC_curve(ROC_ax, labels_fhr, y_pred,
                               fill=False, plabel=False)
                fig.suptitle(f"{suite} {rpt_type}")
                fig.text(0.5, 0.01, ' '.join(
                    df.columns), wrap=True, fontsize=5)
                ofile = f"nn/{thissavedmodel}.{rpt_type}.{imask}.statcurves.png"
                fig.savefig(ofile)
                logging.info(os.path.realpath(ofile))
                plt.clf()
    return stattxt


if debug:
    stattxt = statjob('all', statcurves=True)
    pdb.set_trace()
    sys.exit(0)

fhrs = range(1, nfhr+1)
# allow for datasets like storm mode probabilities, that don't start at fhr=1
fhrs = df.forecast_hour.unique()
fhrs = list(fhrs)
fhrs.insert(0, "all")  # put "all" first because it takes the longest
if nprocs:
    # Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
    # execcasper --ncpus=13 --mem=50GB # gpus not neeeded for verification
    chunksize = int(np.ceil(len(fhrs)/float(nprocs)))
    pool = multiprocessing.Pool(processes=nprocs)
    data = pool.map(statjob, fhrs, chunksize)
    pool.close()
else:
    data = []
    for fhr in fhrs:
        data.append(statjob(fhr))


with open(ofile, "w") as fh:
    fh.write('class,mem,fold,fhr,bss,base rate,auc,aps\n')
    fh.write(''.join(data))

logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")