import argparse
import datetime
import glob
from hwtmode.data import decompose_circular_feature
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, get_argparser, get_glm, rptdist2bool, savedmodel_default
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
    for key in ["batchnorm","batchsize","debug","dropout","epochs","flash","glm","kfold","layers","learning_rate","model","neurons","optimizer","reg_penalty","rptdist", "suite", "twin"]:
        if key == "debug" and debug: continue # if running in debug mode, don't require debug in yaml file to match
        assert getattr(ylargs,key) == getattr(args,key), f'this script {key} {getattr(args,key)} does not match yaml {key} {getattr(ylargs,key)}'

    return True

parser = get_argparser()
parser.add_argument('--nprocs', type=int, default=0, help="verify this many forecast hours in parallel")

args = parser.parse_args()
logging.info(args)

# Assign arguments to simple-named variables
clobber               = args.clobber
debug                 = args.debug
flash                 = args.flash
glm                   = args.glm
kfold                 = args.kfold
model                 = args.model
nfit                  = args.nfits
nprocs                = args.nprocs
rptdist               = args.rptdist
savedmodel            = args.savedmodel
train_test_split_time = args.splittime
suite                 = args.suite
twin                  = args.twin


if debug:
    logging.basicConfig(level=logging.DEBUG)


### saved model name ###
if savedmodel:
    pass
else:
    savedmodel = savedmodel_default(args, fhr_str='f01-f48') # use model trained on f01-f48 regardless of the hour you are testing
logging.info(f"savedmodel={savedmodel}")


if kfold:
    logging.info(f"Do not apply train_test_split_time if KFold was used. KFold will do the splitting.")
    train_test_split_time = pd.to_datetime("19010101")
for ifold in range(kfold):
    for i in range(0,nfit):
        savedmodel_i = f"nn/nn_{savedmodel}_{i}/{kfold}fold{ifold}"
        assert os.path.exists(savedmodel_i), f"{savedmodel_i} not found"

    nextfit = f"nn/nn_{savedmodel}_{i+1}"
    if os.path.exists(nextfit):
        logging.warning(f"next fit exists ({nextfit}). Are you sure nfit only {nfit}?")


odir = os.path.join("/glade/scratch", os.getenv("USER"))
if glm: odir = os.path.join(odir, "GLM")
if not os.path.exists(odir):
    logging.info(f"making directory {odir}")
    os.mkdir(odir)

ofile = os.path.realpath(f"nn/nn_{savedmodel}.{kfold}fold.scores.txt")
if not clobber and os.path.exists(ofile):
    logging.info(f"Exiting because output file {ofile} exists. Use --clobber option to override.")
    sys.exit(0)

logging.info(f"output file will be {ofile}")

##################################


logging.info(f"Read {model} predictors")
if model == "HRRR":
    ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.par'
    if debug: ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.fastdebug.par'
    scalingfile = "/glade/work/ahijevyc/NSC_objects/HRRR/scaling_values_all_HRRRX.pk"
    nfhr = 48
elif model == "NSC3km-12sec":
    ifile0 = f'{model}{glmstr}.par'
    scalingfile = f"scaling_values_{model}_{train_test_split_time:%Y%m%d_%H%M}.pk"
    nfhr = 36


if os.path.exists(ifile0):
    logging.info(f'reading {ifile0}')
    df = pd.read_parquet(ifile0, engine="pyarrow")
else:
    logging.error(f"why is there no parquet file for {model}?")
    logging.error(f"Do you need to run train_stormrpts_dnn.py to make {ifile0}?")
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
        rdict = {"y":"projection_x_coordinate", "x":"projection_y_coordinate" }
        logging.info(f"renaming axes {rdict}")
        df = df.rename_axis(index=rdict) # NSC3km-12sec saved as forecast_hour, y, x 
    elif ys.min() == 21 and ys.max() == 80 and xs.min() == 12 and xs.max() == 46:
        rdict = {"x":"projection_x_coordinate", "y":"projection_y_coordinate" }
        logging.info(f"renaming axes {rdict}")
        df = df.rename_axis(index=rdict) # NSC3km-12sec saved as forecast_hour, y, x 
    else:
        logging.error("unexpected x and y coordinates. check mask, training script, parquet file...")
        sys.exit(1)

# This script expects "forecast_hour" spelled out.
df = df.rename(columns={"fhr": "forecast_hour", "init_time": "initialization_time"})

if "HAILCAST_DIAM_MAX" in df and (df["HAILCAST_DIAM_MAX"] == 0).all():
    logging.info("HAILCAST_DIAM_MAX all zeros. Dropping.")
    df = df.drop(columns="HAILCAST_DIAM_MAX")


assert set(df.index.names) == set(['valid_time', 'projection_x_coordinate', 'projection_y_coordinate']), f"unexpected index names for df {df.index.names}"

labels = df[rptcols] # converted to Boolean above
df = df.drop(columns=rptcols)

validtimes = df.index.get_level_values(level="valid_time")
logging.info(f"range of valid times: {validtimes.min()} - {validtimes.max()}")
df.info()
print(labels.sum())

# Only keep cases with initialization times later than train_test_split_time.
if "initialization_time" in df:
    logging.info("save and drop initialization_time column")
    idate = df.initialization_time.astype('datetime64[ns]')
    df = df.drop(columns="initialization_time")
else:
    logging.info("derive initialization times")
    idate = df.index.get_level_values(level="valid_time") - df["forecast_hour"] * datetime.timedelta(hours=1) 
test_idx = idate >= train_test_split_time
before_filtering  = len(df)
df = df[test_idx]
labels = labels[test_idx]
logging.info(f"keep {len(df)}/{before_filtering} cases with init times at or later than {train_test_split_time}")

assert labels.sum().all() > 0, "at least 1 class has no True labels in testing set"

logging.info(f"normalize with training cases mean and std in {scalingfile}")
sv = pd.read_pickle(scalingfile) # conda activate tf if AttributeError: Can't get attribute 'new_block' on...
if "fhr" in sv:
    logging.info("change fhr to forecast_hour in scaling DataFrame")
    sv = sv.rename(columns={"fhr": "forecast_hour"})

logging.info("Make sure no 7x7 neighborhood predictors are present")
assert all("-N7" not in x for x in df.columns), "-N7 in df. expected them to be dropped already"
logging.info("no 7x7")

storm_mode_columns = [x for x in df.columns if "SS_" in x or "NN_" in x]
if "with_storm_mode" not in suite and len(storm_mode_columns) > 0:
    logging.info(f"Dropping {len(storm_mode_columns)} storm mode columns.")
    df = df.drop(columns=storm_mode_columns)


# You might have scaling factors for columns that you dropped already, like -N7 columns.
extra_sv_columns = set(sv.columns) - set(df.columns)
if extra_sv_columns:
    logging.info(f"dropping {len(extra_sv_columns)} extra scaling factor columns {extra_sv_columns}")
    sv = sv.drop(columns=extra_sv_columns)


def statjob(fhr,statcurves=False):
    if statcurves:
        fig = plt.figure(figsize=(10,7))
    # test model
    y_preds = pd.DataFrame()
    if fhr == "all":
        this_fhr = ~df["forecast_hour"].isna() # all finite forecast hours
    else:
        this_fhr = df["forecast_hour"] == fhr # Just this fhr
    df_fhr = df[this_fhr]
    labels_fhrs = labels[this_fhr]
    logging.debug(f"normalize {fhr}")
    df_fhr = (df_fhr - sv.loc["mean"]) / sv.loc["std"]
    stattxt = ""
    cv = KFold(n_splits=kfold)

    for thisfit in range(nfit):
        for ifold, (train_index, test_index) in enumerate(cv.split(df_fhr)):
            df_fhr_fold = df_fhr.iloc[test_index]
            labels_fhrs_fold = labels_fhrs.iloc[test_index]
            savedmodel_thisfitfold = f"nn/nn_{savedmodel}_{thisfit}/{kfold}fold{ifold}"
            logging.debug(f"reading {savedmodel_thisfitfold} column order")
            yl = yaml.load(open(os.path.join(savedmodel_thisfitfold, "config.yaml"),"r"), Loader=yaml.Loader) # not safe (yaml.FullLoader is) but legacy Loader handles argparse.namespace object. 
            yl_labels = yl["labels"]
            del(yl["labels"]) # delete labels so we can make DataFrame from rest of dictionary.
            assert configs_match(yl["args"], args), f'this configuration {args} does not match yaml file {yl["args"]}'
            del(yl["args"]) 
            assert all(yl_labels == labels.columns), f"labels {label.columns} don't match when model was trained {yl_labels}"
            yl = pd.DataFrame(yl).set_index("columns").T
            if "fhr" in yl:
                logging.info("Rename fhr to forecast_hour in yaml columns")
                yl = yl.rename(columns={"fhr": "forecast_hour"})
            if len(yl.columns) != len(df_fhr_fold.columns):
                print(f"length of yaml and df_fhr_fold columns differ {yl.columns} {df_fhr_fold.columns}")
                pdb.set_trace()
            assert np.isclose(yl.reindex(columns=sv.columns), sv.loc[yl.index]).all(), f"pickle and yaml scaling factors don't match up\n{sv}\n{yl}"
            if not all(yl.columns == df_fhr_fold.columns):
                logging.info(f"reordering columns")
                df_fhr_fold = df_fhr_fold.reindex(columns=yl.columns)
            assert all(yl.columns == df_fhr_fold.columns), f"columns {df.columns} don't match when model was trained {columns}"
            logging.info(f"loading {savedmodel_thisfitfold}")
            model = load_model(savedmodel_thisfitfold, custom_objects=dict(brier_skill_score=brier_skill_score))
            logging.info(f"predicting fit {thisfit} fold{ifold}...")
            Y = model.predict(df_fhr_fold.to_numpy(dtype='float32')) # Grab numpy array of predictions.
            Y = pd.DataFrame(Y, columns=labels_fhrs_fold.columns, index=df_fhr_fold.index) # Convert prediction numpy array to DataFrame with index (row) and column labels.
            for rpt_type in labels_fhrs_fold.columns: # for each report type
                y_pred = Y[rpt_type] # grab this particular report type
                labels_fhr = labels_fhrs_fold[rpt_type]
                assert labels_fhr.index.equals(y_pred.index), f'fit {thisfit} fold{ifold} {rpt_type} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
                bss = brier_skill_score(labels_fhr, y_pred)
                base_rate = labels_fhr.mean()
                auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
                logging.info(f"{rpt_type} fit={thisfit} fold={ifold} fhr={fhr} {bss} {base_rate} {auc}")
                stattxt += f"{rpt_type},{thisfit},{ifold},{fhr},{bss},{base_rate},{auc}\n"
            Y = pd.concat([Y], keys=[thisfit], names=["fit"]) # prepend "fit" level to multilevel DataFrame
            Y = pd.concat([Y], keys=[ifold], names=["fold"]) # prepend "fold" level
            y_preds = y_preds.append(Y) # append this fit/fold to the y_preds DataFrame
    # I may have overlapping valid_times from different init_times like fhr=1 from today and fhr=25 from previous day
    ensmean = y_preds.groupby(level=["valid_time","projection_y_coordinate","projection_x_coordinate"]).mean() # average probability over all nfits and initialization_times valid at valid_time 
    assert "fit" not in ensmean.index.names, "fit should not be a MultiIndex level of ensmean, the average probability over nfits."
    thissavedmodel = savedmodel.replace('f01-f48', f'f{fhr}') # for statistic curves plot file name
    logging.debug(f"getting bss, base rate, auc")
    for rpt_type in labels_fhrs.columns:
        y_pred = ensmean[rpt_type]
        labels_fhr = labels_fhrs[rpt_type]
        # probability averaged over all nfits and initialization_times valid at valid_time, eliminating multiple values for multiple initialization times. Do same for labels_fhrs.
        # First and last label should be same for a particular valid_time, y, and x coordinate.
        first_label = labels_fhr.groupby(level=["valid_time","projection_y_coordinate","projection_x_coordinate"]).first()
        last_label  = labels_fhr.groupby(level=["valid_time","projection_y_coordinate","projection_x_coordinate"]).last()
        assert first_label.equals(last_label), f"discrepancies in first and last forecasts valid at same valid time {(first != last).sum()}" 
        labels_fhr = first_label # just need one instance of the duplicate labels
        assert labels_fhr.index.equals(y_pred.index), f'{rpt_type} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
        bss = brier_skill_score(labels_fhr, y_pred)
        base_rate = labels_fhr.mean()
        auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
        # TODO: add average_precision_score
        logging.info(f"{rpt_type} ensmean fold=all fhr={fhr} {bss} {base_rate} {auc}")
        stattxt += f"{rpt_type},ensmean,all,{fhr},{bss},{base_rate},{auc}\n"
        if statcurves:
            logging.info(f"{rpt_type} reliability diagram, histogram, & ROC curve")
            ax1 = plt.subplot2grid((3,2), (0,0), rowspan=2)
            ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, sharex=ax1)
            ROC_ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
            reliability_diagram_obj, = reliability_diagram(ax1, labels_fhr, y_pred)
            counts, bins, patches = count_histogram(ax2, y_pred)
            rc = ROC_curve(ROC_ax, labels_fhr, y_pred, fill=False, plabel=False)
            fig.suptitle(f"{suite} {rpt_type}")
            fig.text(0.5, 0.01, ' '.join(df.columns), wrap=True, fontsize=5)
            ofile = f"nn/{thissavedmodel}.{rpt_type}.statcurves.png"
            fig.savefig(ofile)
            logging.info(os.path.realpath(ofile))
            plt.clf()
    return stattxt

if debug:
    statjob('all', statcurves=True)
    sys.exit(0)

fhrs = range(1,nfhr+1)
fhrs = df.forecast_hour.unique() # allow for datasets, like storm mode probabilities, that don't start at fhr=1
fhrs = list(fhrs)
fhrs.insert(0,"all") # put "all" first because it takes the longest
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
    fh.write('class,mem,fold,fhr,bss,base rate,auc\n')
    fh.write(''.join(data))

logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
