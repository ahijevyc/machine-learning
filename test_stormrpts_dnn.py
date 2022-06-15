import argparse
import datetime
import glob
from hwtmode.data import decompose_circular_feature
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
    parser.add_argument('--flash', type=int, default=10, help="GLM flash threshold")
    parser.add_argument('--layers', default=2, type=int, help="number of hidden layers")
    parser.add_argument('--model', type=str, choices=["HRRR","NSC3km-12sec"], default="HRRR", help="prediction model")
    parser.add_argument("--glm", action='store_true', help='Use GLM')
    parser.add_argument('--savedmodel', type=str, help="filename of machine learning model")
    parser.add_argument('--neurons', type=int, nargs="+", default=[16], help="number of neurons in each nn layer")
    parser.add_argument('--nprocs', type=int, default=12, help="verify this many forecast hours in parallel")
    parser.add_argument('--rptdist', type=int, default=40, help="severe weather report max distance")
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
    if "noN7" in suite: ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/sobash.noN7_Mar-Oct2021.par'
    scalingfile = "/glade/work/ahijevyc/NSC_objects/HRRR/scaling_values_all_HRRRX.pk"
    nfhr = 48
elif model == "NSC3km-12sec":
    ifile0 = f'{model}.{suite}_test.par'
    scalingfile = f"scaling_values_{model}.pk"
    nfhr = 36


if os.path.exists(ifile0):
    logging.info(f'reading {ifile0}')
    df = pd.read_parquet(ifile0, engine="pyarrow")
else:
    if model == "HRRR":
        ifiles = glob.glob(f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_20210[3-9]??00-0000.par')
        ifiles.extend(glob.glob(f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_202110??00-0000.par'))
    elif model == "NSC3km-12sec":
        logging.info(f"ingest vx period for {model} (after 20160701)")
        ifiles = glob.glob(f'/glade/work/sobash/NSC_objects/grid_data/grid_data_{model}_d01_20160[789]??00-0000.par')
        ifiles.extend(glob.glob(f'/glade/work/sobash/NSC_objects/grid_data/grid_data_{model}_d01_20161???00-0000.par'))
        ifiles.extend(glob.glob(f'/glade/work/sobash/NSC_objects/grid_data/grid_data_{model}_d01_201[789]????00-0000.par'))

    logging.info(f"Reading {len(ifiles)} {model} files")
    df = pd.concat( pd.read_parquet(ifile, engine="pyarrow") for ifile in ifiles)

    N7_columns = [x for x in df.columns if "-N7" in x]
    logging.info(f"drop {len(N7_columns)} N7 columns: {N7_columns}")
    df = df.drop(columns=N7_columns)

    df = df.rename(columns=dict(xind="projection_y_coordinate",yind="projection_x_coordinate"))
    df["valid_time"] = pd.to_datetime(df["Date"]) + df["fhr"] * datetime.timedelta(hours=1)
    df = df.drop(columns=["Date"]) # don't need initialization date after deriving valid_time

    df["dayofyear"] = df["valid_time"].dt.dayofyear
    df["Local_Solar_Hour"] = df["valid_time"].dt.hour + df["lon"]/15
    df = decompose_circular_feature(df, "dayofyear", period=365.25)
    df = decompose_circular_feature(df, "Local_Solar_Hour", period=24)
    df = df.set_index(["valid_time","projection_y_coordinate","projection_x_coordinate"])

    if glm:
        glmds = get_glm(twin,rptdist)
        glmds = glmds.drop_vars(["lon","lat"]) # Don't interfere with HRRR lat and lon.
        logging.info("Merge flashes with df")
        df = df.merge(glmds.to_dataframe(), on=df.index.names)

    logging.info(f'writing parquet {ifile0}')
    df.to_parquet(ifile0)
    logging.info(f'wrote parquet {ifile0}')

df, rptcols = rptdist2bool(df, rptdist, twin)

if glm:
    df["flashes"] = df["flashes"] >= flash
    rptcols.append("flashes")

labels = df[rptcols] # converted to Boolean above
df = df.drop(columns=rptcols)


if (df["HAILCAST_DIAM_MAX"] == 0).all():
    logging.info("HAILCAST_DIAM_MAX all zeros. Dropping.")
    df = df.drop(columns="HAILCAST_DIAM_MAX")

validtimes = df.index.get_level_values(level="valid_time")
logging.info(f"first valid time: {validtimes.min()}  last valid time: {validtimes.max()}")
df.info()
print(labels.sum())

logging.info("normalize with training cases mean and std.")
df_desc = pd.read_pickle(scalingfile) # conda activate tf if AttributeError: Can't get attribute 'new_block' on...

if "noN7" in suite:
    assert all("-N7" not in x for x in df.columns), "-N7 in df. expected them to be dropped already"
    N7_columns = [x for x in df_desc.columns if "-N7" in x]
    logging.debug(f"drop {len(N7_columns)} N7 columns: {N7_columns}")
    df_desc = df_desc.drop(columns=N7_columns)


assert labels.sum().all() > 0, "at least 1 class has no True labels in testing set"


def statjob(fhr,statcurves=False):
    if statcurves:
        fig = plt.figure(figsize=(10,7))
    # test model
    y_preds = pd.DataFrame()
    this_fhr = df.fhr == fhr # Just this fhr
    df_fhr = df[this_fhr]
    labels_fhrs = labels[this_fhr]
    logging.debug(f"normalize {fhr}")
    df_fhr = (df_fhr - df_desc.loc["mean"]) / df_desc.loc["std"]
    stattxt = ""
    for thisfit in range(nfit):
        savedmodel_thisfit = f"nn/nn_{savedmodel}_{thisfit}"
        logging.debug(f"reading {savedmodel_thisfit} column order")
        yl = yaml.load(open(os.path.join(savedmodel_thisfit, "columns.yaml"),"r"), Loader=yaml.FullLoader)
        yl_labels = yl["labels"]
        del(yl["labels"]) # delete labels so we can make DataFrame from rest of dictionary.
        assert all(yl_labels == labels.columns), f"labels {label.columns} don't match when model was trained {yl_labels}"
        yl = pd.DataFrame(yl).set_index("columns").T
        assert np.isclose(yl.reindex(columns=df_desc.columns), df_desc.loc[yl.index]).all(), "pickle and yaml scaling factors don't match up {df_desc} {yl}"
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
        for rptcol in labels_fhrs.columns: # for each report type
            y_pred = y_preds.xs(thisfit, level="fit")[rptcol] # Grab this particular report type
            labels_fhr = labels_fhrs[rptcol]
            assert labels_fhr.index.equals(y_pred.index), f'fit {thisfit} {rptcol} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
            bss = brier_skill_score(labels_fhr, y_pred)
            base_rate = labels_fhr.mean()
            auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
            logging.info(f"{rptcol} fit={thisfit} fhr={fhr} {bss} {base_rate} {auc}")
            stattxt += f"{rptcol},{thisfit},{fhr},{bss},{base_rate},{auc}\n"
    ensmean = y_preds.groupby(level=["valid_time","projection_y_coordinate","projection_x_coordinate"]).mean() # average probability over nfits 
    assert "fit" not in ensmean.index.names, "fit should not be a MultiIndex level of ensmean, the average probability over nfits."
    # write predictions from each member and ensemble mean and the observed truth. (for debugging). 
    ofile = os.path.join(odir, f"f{fhr:02d}.csv")
    pd.concat([labels_fhrs.astype(int)], keys=["obs"], names=["fit"]).append(y_preds).to_csv(ofile)
    thissavedmodel = savedmodel.replace('f01-f48', f'f{fhr:02d}')
    logging.debug(f"getting bss, base rate, auc")
    for rptcol in labels_fhrs.columns:
        y_pred = ensmean[rptcol]
        labels_fhr = labels_fhrs[rptcol]
        labels_fhr = labels_fhr.reindex_like(y_pred) # valid_time is sorted in y_pred, not labels_fhr
        assert labels_fhr.index.equals(y_pred.index), f'{rptcol} label and prediction indices differ {labels_fhr.index} {y_pred.index}'
        bss = brier_skill_score(labels_fhr, y_pred)
        base_rate = labels_fhr.mean()
        auc = sklearn.metrics.roc_auc_score(labels_fhr, y_pred) if labels_fhr.any() else np.nan
        logging.info(f"{rptcol} ensmean fhr={fhr} {bss} {base_rate} {auc}")
        stattxt += f"{rptcol},ensmean,{fhr},{bss},{base_rate},{auc}\n"
        if statcurves:
            logging.info(f"{rptcol} reliability diagram, histogram, & ROC curve")
            ax1 = plt.subplot2grid((3,2), (0,0), rowspan=2)
            ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, sharex=ax1)
            ROC_ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
            reliability_diagram_obj, = reliability_diagram(ax1, labels_fhr, y_pred)
            counts, bins, patches = count_histogram(ax2, y_pred)
            rc = ROC_curve(ROC_ax, labels_fhr, y_pred, fill=False, plabel=False)
            fig.suptitle(f"{suite} {rptcol}")
            fig.text(0.5, 0.01, ' '.join(df.columns), wrap=True, fontsize=5)
            ofile = f"nn/{thissavedmodel}.{rptcol}.statcurves.png"
            fig.savefig(ofile)
            logging.info(os.path.realpath(ofile))
            plt.clf()
    return stattxt

if debug:
    data = statjob(12,statcurves=True)
    sys.exit(0)

fhrs = range(1,nfhr+1)
# Verify nprocs forecast hours in parallel. Execute script on machine with nprocs+1 cpus
# execcasper --ncpus=13 --mem=50GB # gpus not neeeded for verification
chunksize = int(np.ceil(len(fhrs)/float(nprocs)))
pool = multiprocessing.Pool(processes=nprocs)
data = pool.map(statjob, fhrs, chunksize)
pool.close()

with open(ofile, "w") as fh:
    fh.write('class,mem,fhr,bss,base rate,auc\n')
    fh.write(''.join(data))

logging.info(f"wrote {ofile}. Plot with \n\npython nn_scores.py {ofile}")
