import argparse
import datetime
import glob
from hwtmode.statisticplot import count_histogram, reliability_diagram, ROC_curve
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, rptdist2bool, get_glm
import multiprocessing
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import sklearn
from tensorflow.keras.models import load_model
import sys
import time
import xarray
import yaml

"""
 Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
 execcasper --ngpus 13 --mem=50GB # gpus not neeeded for verification
"""


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def parse_args():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "test neural network(s) in parallel. output truth and predictions from each member and ensemble mean for each forecast hour",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', type=int, default=512, help="nn training batch size") # tf default is 32
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument('--nfits', type=int, default=10, help="number of times to fit (train) model")
    parser.add_argument('--epochs', default=30, type=int, help="number of training epochs")
    parser.add_argument('--flash', type=int, default=10, help="GLM flash count threshold")
    parser.add_argument('--layers', default=2, type=int, help="number of hidden layers")
    parser.add_argument('--model', type=str, choices=["HRRR","NSC3km-12sec"], default="HRRR", help="prediction model")
    parser.add_argument("--glm", action='store_true', help='Use GLM')
    parser.add_argument('--savedmodel', type=str, help="filename of machine learning model")
    parser.add_argument('--neurons', type=int, nargs="+", default=[16], help="number of neurons in each nn layer")
    parser.add_argument('--nprocs', type=int, default=0, help="verify this many forecast hours in parallel")
    parser.add_argument('--rptdist', type=int, default=40, help="severe weather report max distance")
    parser.add_argument('--splittime', type=lambda s: pd.to_datetime(s), default="202012021200", help="train with storms before this time; test this time and after")
    parser.add_argument('--suite', type=str, default='default', choices=["default","with_storm_mode"], help="name for suite of training features")
    parser.add_argument('--twin', type=int, default=2, help="time window in hours")
    args = parser.parse_args()
    return args

args = parse_args()
# Assign arguments to simple-named variables
batchsize             = args.batchsize
clobber               = args.clobber
debug                 = args.debug
epochs                = args.epochs
flash                 = args.flash
nfit                  = args.nfits
layer                 = args.layers
model                 = args.model
neurons               = args.neurons
glm                   = args.glm
nprocs                = args.nprocs
rptdist               = args.rptdist
savedmodel            = args.savedmodel
train_test_split_time = args.splittime
suite                 = args.suite
twin                  = args.twin


if debug:
    logging.basicConfig(level=logging.DEBUG)


logging.info(args)

### saved model name ###

trained_models_dir = '/glade/work/ahijevyc/NSC_objects'
if savedmodel:
    pass
else:
    # use model trained on f01-f48 regardless of the hour you are testing
    fhr_str = 'f01-f48'
    glmstr = "" # GLM description 
    if glm: glmstr = f"{flash}flash_{twin}hr." # flash rate threshold and GLM time window
    savedmodel = f"{model}.{suite}.{glmstr}rpt_{rptdist}km_{twin}hr.{neurons[0]}n.ep{epochs}.{fhr_str}.bs{batchsize}.{layer}layer"
logging.info(f"savedmodel={savedmodel}")

for i in range(0,nfit):
    savedmodel_i = f"nn/nn_{savedmodel}_{i}"
    assert os.path.exists(savedmodel_i), f"{savedmodel_i} not found"

nextfit = f"nn/nn_{savedmodel}_{i+1}"
if os.path.exists(nextfit):
    logging.warning(f"next fit exists ({nextfit}). Are you sure nfit only {nfit}?")

odir = os.path.join("/glade/scratch", os.getenv("USER"))
if glm: odir = os.path.join(odir, "GLM")
if not os.path.exists(odir):
    logging.info(f"making directory {odir}")
    os.mkdir(odir)

ofile = os.path.realpath(f"nn/nn_{savedmodel}.scores.txt")
logging.info(f"output file will be {ofile}")

##################################


logging.info(f"Read {model} predictors")
if model == "HRRR":
    ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/sobash_test_Mar-Oct2021.par'
    ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/sobash.noN7_Mar-Oct2021.par'
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
    if model == "HRRR":
        ifiles = glob.glob(f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_20210[3-9]??00-0000.par')
        ifiles.extend(glob.glob(f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_202110??00-0000.par'))
    else:
        logging.error(f"why is there no parquet file for {model}?")
        logging.error(f"Do you need to run train_stormrpts_dnn.py to make {ifil0}?")
        sys.exit(1)

    if glm:
        latest_valid_time = df.index.max()[0]
        assert latest_valid_time > pd.to_datetime("20160101"), "DataFrame completely before GLM exists"
        glmds = get_glm(twin, rptdist)
        glmds = glmds.drop_vars(["lon","lat"]) # Don't interfere with HRRR lat and lon.
        logging.info("Merge flashes with df")
        #Do {model} and GLM overlap at all?"
        df = df.merge(glmds.to_dataframe(), on=df.index.names)
        assert not df.empty, f"Merged {model}/GLM Dataset is empty."

    logging.info(f'writing parquet {ifile0}')
    df.to_parquet(ifile0)
    logging.info(f'wrote parquet {ifile0}')

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

logging.info(f"normalize with training cases mean and std in {scalingfile}.")
sv = pd.read_pickle(scalingfile) # conda activate tf if AttributeError: Can't get attribute 'new_block' on...
if "fhr" in sv:
    logging.info("change fhr to forecast_hour in scaling DataFrame")
    sv = sv.rename(columns={"fhr": "forecast_hour"})

logging.info("Make sure no 7x7 neighborhood predictors are present")
assert all("-N7" not in x for x in df.columns), "-N7 in df. expected them to be dropped already"

if "with_storm_mode" not in suite:
    logging.info("making predictors don't include storm mode")
    storm_mode_columns = [x for x in df.columns if "SS_" in x or "NN_" in x]
    df = df.drop(columns=storm_mode_columns)


# You might have scaling factors for columns that you dropped already, like -N7 columns.
extra_sv_columns = set(sv.columns) - set(df.columns)
if extra_sv_columns:
    logging.warning(f"dropping {len(extra_sv_columns)} extra scaling factor columns {extra_sv_columns}")
    sv = sv.drop(columns=extra_sv_columns)


def statjob(fhr,statcurves=False):
    if statcurves:
        fig = plt.figure(figsize=(10,7))
    # test model
    y_preds = pd.DataFrame()
    this_fhr = df["forecast_hour"] == fhr # Just this fhr
    df_fhr = df[this_fhr]
    labels_fhrs = labels[this_fhr]
    logging.debug(f"normalize {fhr}")
    df_fhr = (df_fhr - sv.loc["mean"]) / sv.loc["std"]
    stattxt = ""
    for thisfit in range(nfit):
        savedmodel_thisfit = f"nn/nn_{savedmodel}_{thisfit}"
        logging.debug(f"reading {savedmodel_thisfit} column order")
        yl = yaml.load(open(os.path.join(savedmodel_thisfit, "config.yaml"),"r"), Loader=yaml.Loader) # not "safe" (yaml.FullLoader is) but legacy Loader handles argparse.namespace object. 
        yl_labels = yl["labels"]
        del(yl["labels"]) # delete labels so we can make DataFrame from rest of dictionary.
        assert yl["args"].splittime == train_test_split_time, f"yaml train_test_split_time {yl['args']['train_test_split_time']} does not match value from this script {train_test_split_time}"
        del(yl["args"]) 
        assert all(yl_labels == labels.columns), f"labels {label.columns} don't match when model was trained {yl_labels}"
        yl = pd.DataFrame(yl).set_index("columns").T
        if "fhr" in yl:
            logging.info("change fhr to forecast_hour in yaml columns")
            yl = yl.rename(columns={"fhr": "forecast_hour"})
        if len(yl.columns) != len(df_fhr.columns):
            print(f"length of yaml and df_fhr columns differ {yl.columns} {df_fhr.columns}")
            pdb.set_trace()
        assert np.isclose(yl.reindex(columns=sv.columns), sv.loc[yl.index]).all(), f"pickle and yaml scaling factors don't match up {sv} {yl}"
        if not all(yl.columns == df_fhr.columns):
            logging.info(f"reordering columns")
            df_fhr = df_fhr.reindex(columns=yl.columns)
        assert all(yl.columns == df_fhr.columns), f"columns {df.columns} don't match when model was trained {columns}"
        logging.info(f"loading {savedmodel_thisfit}")
        model = load_model(savedmodel_thisfit, custom_objects=dict(brier_skill_score=brier_skill_score))
        logging.info(f"predicting...")
        y_pred = model.predict(df_fhr.to_numpy(dtype='float32')) # Grab numpy array of predictions.
        y_pred = pd.DataFrame(y_pred, columns=labels_fhrs.columns, index=df_fhr.index) # Convert prediction numpy array to DataFrame with index (row) and column labels.
        y_pred = pd.concat([y_pred], keys=[thisfit], names=["fit"]) # make prediction DataFrame a multilevel DataFrame by prepending the index with another level called "fit"
        y_preds = y_preds.append(y_pred) # append this fit to the y_preds DataFrame
        for rpt_type in labels_fhrs.columns: # for each report type
            y_pred = y_preds.xs(thisfit, level="fit")[rpt_type] # Grab this particular report type
            labels_fhr = labels_fhrs[rpt_type]
            assert labels_fhr.index.equals(y_pred.index), f'fit {thisfit} {rpt_type} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
            bss = brier_skill_score(labels_fhr, y_pred)
            base_rate = labels_fhr.mean()
            auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
            logging.info(f"{rpt_type} fit={thisfit} fhr={fhr} {bss} {base_rate} {auc}")
            stattxt += f"{rpt_type},{thisfit},{fhr},{bss},{base_rate},{auc}\n"
    # TODO: do I have to worry about overlapping valid_times from different init_times?
    ensmean = y_preds.groupby(level=["valid_time","projection_y_coordinate","projection_x_coordinate"]).mean() # average probability over nfits 
    assert "fit" not in ensmean.index.names, "fit should not be a MultiIndex level of ensmean, the average probability over nfits."
    # write predictions from each member and ensemble mean and the observed truth. (for debugging). 
    ofile = os.path.join(odir, f"f{fhr:02d}.csv")
    pd.concat([labels_fhrs.astype(int)], keys=["obs"], names=["fit"]).append(y_preds).to_csv(ofile)
    thissavedmodel = savedmodel.replace('f01-f48', f'f{fhr:02d}') # for statistic curves plot file name
    logging.debug(f"getting bss, base rate, auc")
    for rpt_type in labels_fhrs.columns:
        y_pred = ensmean[rpt_type]
        labels_fhr = labels_fhrs[rpt_type]
        labels_fhr = labels_fhr.reindex_like(y_pred) # valid_time is sorted in y_pred, not labels_fhr
        assert labels_fhr.index.equals(y_pred.index), f'{rpt_type} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
        bss = brier_skill_score(labels_fhr, y_pred)
        base_rate = labels_fhr.mean()
        auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
        logging.info(f"{rpt_type} ensmean fhr={fhr} {bss} {base_rate} {auc}")
        stattxt += f"{rpt_type},ensmean,{fhr},{bss},{base_rate},{auc}\n"
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
    sys.exit(0)

fhrs = range(1,nfhr+1)
fhrs = df.forecast_hour.unique() # allow for datasets, like storm mode probabilities, that don't start at fhr=1
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
    fh.write('class,mem,fhr,bss,base rate,auc\n')
    fh.write(''.join(data))

logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
