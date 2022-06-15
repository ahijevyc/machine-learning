#!/usr/bin/env python

def baseline_model(input_dim=None, name=None,numclasses=None, neurons=16, layer=2, optimizer='adam', dropout=0):

    # Discard any pre-existing version of the model.
    model = Sequential(name=name)
    model.add(Dense(neurons, input_dim=input_dim, activation='relu', name="storm_and_env_features"))
    for i in range(layer-1):
        model.add(Dropout(rate=dropout))
        model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(numclasses, activation='sigmoid')) # used softmax in HWT_mode to add to 1

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss="binary_crossentropy" # in HWT_mode, I used categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=[MeanSquaredError(), brier_skill_score, AUC(), "accuracy"])

    return model

import argparse
import datetime
import G211
import glob
from hwtmode.data import decompose_circular_feature
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, rptdist2bool, get_glm
import numpy as np
import os
import pandas as pd
import pdb
import pickle
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.models import Sequential, load_model
import sys
import time
import xarray
import yaml




def f0i(i):
    return f"f{i:02d}"

def make_fhr_str(fhr):
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
               final.append(f0i(seq[0]) + '-' + f0i(seq[len(seq)-1]))
            else:
               final.append(f0i(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(fhr) - 1:
            if len(seq) > 1:
                final.append(f0i(seq[0]) + '-' + f0i(seq[len(seq)-1]))
            else:
                final.append(f0i(seq[0]))

   
    final_str = '.'.join(map(str, final))
    return final_str


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
def main():
    import pandas as pd # started getting UnboundLocalError: local variable 'pd' referenced before assignment Mar 1 2022 even though I import pandas above

    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "train neural network",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', type=int, default=512, help="nn training batch size") # tf default is 32
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.0], help='fraction of neurons to drop in each hidden layer (0-1)')
    parser.add_argument('--fhr', nargs="+", type=int, default=list(range(1,49)), help="forecast hour")
    parser.add_argument('--fits', nargs="+", type=int, default=None, help="work on specific fit(s) so you can run many in parallel")
    parser.add_argument('--nfits', type=int, default=10, help="number of times to fit (train) model")
    parser.add_argument('--epochs', default=30, type=int, help="number of training epochs")
    parser.add_argument('--flash', type=int, default=10, help="GLM flash threshold")
    parser.add_argument('--layers', default=2, type=int, help="number of hidden layers")
    parser.add_argument('--model', type=str, choices=["HRRR","NSC3km-12sec"], default="HRRR", help="prediction model")
    parser.add_argument("--glm", action='store_true', help='Use GLM')
    parser.add_argument('--savedmodel', type=str, help="filename of machine learning model")
    parser.add_argument('--neurons', type=int, nargs="+", default=[16], help="number of neurons in each nn layer")
    parser.add_argument('--rptdist', type=int, default=40, help="severe weather report max distance")
    parser.add_argument('--splittime', type=lambda s: pd.to_datetime(s), default="202012021200", help="train with storms before this time; test this time and after")
    parser.add_argument('--suite', type=str, default='default', choices=["default","with_storm_mode"], help="name for suite of training features")
    parser.add_argument('--twin', type=int, default=2, help="time window in hours")


    # Assign arguments to simple-named variables
    args = parser.parse_args()
    batchsize             = args.batchsize
    clobber               = args.clobber
    debug                 = args.debug
    dropouts              = args.dropouts
    epochs                = args.epochs
    flash                 = args.flash
    fhr                   = args.fhr
    fits                  = args.fits
    nfit                  = args.nfits
    glm                   = args.glm
    layer                 = args.layers
    model                 = args.model
    neurons               = args.neurons
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
        fhr_str = make_fhr_str(fhr) # abbreviate list of forecast hours with hyphens (where possible) so model name is not too long for tf. 
        glmstr = "" # no GLM description 
        if glm: glmstr = f"{flash}flash_{twin}hr." # flash rate threshold and GLM time window
        savedmodel = f"{model}.{suite}.{glmstr}rpt_{rptdist}km_{twin}hr.{neurons[0]}n.ep{epochs}.{fhr_str}.bs{batchsize}.{layer}layer"
    logging.info(f"savedmodel={savedmodel}")

    ##################################

    #mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
    mask = pickle.load(open('./usamask.pk', 'rb'))
    height, width =  65,93
    mask = mask.reshape((height,width))
    mask = xarray.DataArray(mask,dims=["y","x"])
    if False:
        ax = plt.axes(projection = G211.g211)
        xs = G211.xs
        ys = G211.ys


    logging.info(f"Read {model} predictors. Use parquet file, if it exists. If it doesn't exist, create it.")
    if model == "HRRR":
        alsoHRRRv4 = False
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.par'
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.par'
        if alsoHRRRv4: ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRXHRRR.32bit.noN7.par'
        scalingfile = f"/glade/work/ahijevyc/NSC_objects/{model}/scaling_values_all_HRRRX.pk"
    elif model == "NSC3km-12sec":
        ifile = f'{model}.par'
        scalingfile = f"scaling_values_{model}.pk"

    if os.path.exists(ifile):
        logging.info(f'reading {ifile}')
        df = pd.read_parquet(ifile, engine="pyarrow")
    else:
        # Define ifiles, a list of input files from glob.glob method
        if model == "HRRR":
            search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRRX_d01_20*00-0000.par' # just 00z 
            ifiles = glob.glob(search_str)
            if alsoHRRRv4:
                search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_202[01]*00-0000.par' # HRRR, not HRRRX. no 2022 (yet) 
                ifiles.extend(glob.glob(search_str))
        elif model == "NSC3km-12sec":
            search_str = f'/glade/work/sobash/NSC_objects/grid_data/grid_data_{model}_d01_201*00-0000.par'
            ifiles = glob.glob(search_str)

        # remove larger neighborhood size (fields containing N7 in the name)
        df = pd.read_parquet(ifiles[0], engine="pyarrow")
        columns = df.columns
        N7_columns = [x for x in df.columns if "-N7" in x]
        if "noN7" in suite:
            logging.debug(f"ignoring {len(N7_columns)} N7 columns: {N7_columns}")
            columns = set(df.columns) - set(N7_columns)
        # all columns including severe reports
        logging.info(f"Reading {len(ifiles)} {model} files {search_str}")
        df = pd.concat( pd.read_parquet(ifile, engine="pyarrow", columns=columns) for ifile in ifiles)
        logging.info("done")
        df["valid_time"] = pd.to_datetime(df["Date"]) + df["fhr"] * datetime.timedelta(hours=1)
        df["dayofyear"] = df["valid_time"].dt.dayofyear
        df["Local_Solar_Hour"] = df["valid_time"].dt.hour + df["lon"]/15
        df = decompose_circular_feature(df, "dayofyear", period=365.25)
        df = decompose_circular_feature(df, "Local_Solar_Hour", period=24)
        df = df.rename(columns=dict(Date="initialization_time", xind="projection_y_coordinate",yind="projection_x_coordinate"))
        dtype_dict =      {k:np.float32 for k in df.select_dtypes(np.float64).columns}
        dtype_dict.update({k:np.int32   for k in df.select_dtypes(np.int64).columns})
        logging.info("convert 64-bit to 32-bit columns")
        df = df.astype(dtype_dict, copy=False)
        df = df.set_index(["valid_time","projection_y_coordinate","projection_x_coordinate"])
        logging.info(f"writing {ifile}")
        df.to_parquet(ifile)

    if glm:
        latest_valid_time = df.index.max()[0]
        assert latest_valid_time > pd.to_datetime("20160101"), "DataFrame completely before GLM exists"
        glmds = get_glm(twin, rptdist)
        logging.info("Merge flashes with df")
        #Do {model} and GLM overlap at all?"
        df = df.merge(glmds.to_dataframe(), on=df.index.names)
        assert not df.empty, f"Merged Dataset is empty."
    
    df, rptcols = rptdist2bool(df, rptdist, twin)

    plotclimo=False
    if plotclimo:
        mdf = df.groupby(["lon_x","lat_x"]).sum() # for debug plot
        fig, axes = plt.subplots(nrows=3,ncols=2)
        for label,ax in zip(rptcols, axes.flatten()):
            im = ax.scatter(mdf.index.get_level_values("lon_x"), mdf.index.get_level_values("lat_x"), c=mdf[label]) # Don't use mdf (sum) for lon/lat
            ax.set_title(label)
            fig.colorbar(im, ax=ax)
        plt.show()

    # speed things up without multiindex
    df = df.reset_index(drop=True)

    if glm:
        # Sanity check--make sure prediction model and GLM grid box lat lons are similar
        assert (df.lon_y - df.lon_x).max() < 0.1, f"{model} and glm longitudes don't match"
        assert (df.lat_y - df.lat_x).max() < 0.1, f"{model} and glm lats don't match"
        df = df.drop(columns=["lon_y","lat_y"])
        df = df.rename(columns=dict(lon_x="lon",lat_x="lat")) # helpful for scale factor pickle file.
        df["flashes"] = df["flashes"] >= flash
        rptcols.append("flashes")


    labels = df[rptcols] # converted to Boolean above
    df = df.drop(columns=rptcols)

    if (df["HAILCAST_DIAM_MAX"] == 0).all():
        logging.info("HAILCAST_DIAM_MAX all zeros. Dropping.")
        df = df.drop(columns="HAILCAST_DIAM_MAX")


    df.info()
    labels.info()

    # Split into training and testing cases
    # HRRRv3 to v4 at 20201202 0z.
    if train_test_split_time:
        idate = df.initialization_time.astype('datetime64[ns]')
        df = df.drop(columns="initialization_time")
        logging.info(f"train test split time {train_test_split_time}")
        test_idx  = (idate > train_test_split_time) & (idate < pd.to_datetime("20211031")) # Mar 1 - Oct 31 2021 
        train_idx = ~test_idx
        df_train = df[train_idx]
        df_test  = df[test_idx]
        train_labels = labels[train_idx]
        test_labels  = labels[test_idx]
    else:
        df_train, df_test, train_labels, test_labels = train_test_split(df, labels, test_size=0.1, shuffle=False)

    assert all(train_labels.sum()) > 0, "some classes have no True labels in training set"


    logging.info("normalize data")
    if os.path.exists(scalingfile):
        logging.info(f"using pickle file {scalingfile}")
        sv = pickle.load(open(scalingfile, "rb")).astype(np.float32)
    else:
        logging.info(f"calculate mean and std, save to {scalingfile}")
        sv = df_train.describe()
        sv.to_pickle(scalingfile)

    # You might have scaling factors for columns that you dropped already, like -N7 columns.
    extra_sv_columns = set(sv.columns) - set(df.columns)
    if extra_sv_columns:
        logging.warning(f"dropping {len(extra_sv_columns)} extra scaling factor columns {extra_sv_columns}")
        sv = sv.drop(columns=extra_sv_columns)

    # Check for zero standard deviation. (can't normalize)
    stdiszero = sv.loc["std"] == 0
    if stdiszero.any():
        logging.error(f"{sv.columns[stdiszero]} std equals zero")
    df_train = (df_train - sv.loc["mean"]) / sv.loc["std"]
    df_test  = (df_test  - sv.loc["mean"]) / sv.loc["std"]
    logging.info('done normalizing')

    df_train.info()

    if df_train.describe().isna().any().any():
        logging.error(f"nan(s) in {df_train.columns[df_train.mean().isna()]}")
    # train model
    if not fits: # if user did not ask for specific fits, assume fits range from 0 to nfit-1.
        fits = range(0,nfit)
    for i in fits:
        model_i = f"nn/nn_{savedmodel}_{i}"
        if not clobber and os.path.exists(model_i):
            logging.info(f"{model_i} exists")
        else:
            logging.info(f"fitting {model_i}")
            model = baseline_model(input_dim=df_train.columns.size,numclasses=train_labels.columns.size, neurons=neurons[0], layer=layer, name=f"fit_{i}")
            history = model.fit(df_train.to_numpy(dtype='float32'), train_labels.to_numpy(dtype='float32'), class_weight=None, sample_weight=None, batch_size=batchsize,
                epochs=epochs, validation_data=(df_test.to_numpy(dtype='float32'), test_labels.to_numpy(dtype='float32')), verbose=2)
            logging.debug(f"saving {model_i}")
            model.save(model_i)
            # Save order of columns, scaling factors 
            with open(os.path.join(model_i, "columns.yaml"), "w") as file:
                yaml.dump(
                        dict(columns=df_train.columns.to_list(),
                            mean=sv.loc["mean"].reindex(df_train.columns).to_list(),
                            std=sv.loc["std"].reindex(df_train.columns).to_list(),
                            labels=train_labels.columns.to_list()
                            ), file)


if __name__ == "__main__":
    main()
