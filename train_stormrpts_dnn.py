import argparse
import datetime
import G211
import glob
from hwtmode.data import decompose_circular_feature
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, get_argparser, get_glm, get_optimizer, rptdist2bool, savedmodel_default
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import random
import re
import sys
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
#import tensorflow.keras.backend # maybe delete? 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import L2
import time
import visualizecv # custom script by ahijevyc modified from sklearn web page
import xarray
import yaml

def baseline_model(input_dim=None, name=None,numclasses=None, neurons=16, layer=2, kernel_regularizer=None, 
        optimizer='adam', dropout=0, batch_normalize=False, learningrate=0.01):

    # Discard any pre-existing version of the model.
    model = Sequential(name=name)
    model.add(Dense(neurons, input_dim=input_dim, activation='relu', name="storm_and_env_features"))
    for i in range(layer-1): # TODO: figure out how to add dropout, batch normalization, and kernel regularization, even with only one layer. 
        model.add(Dropout(rate=dropout))
        if batch_normalize:
            model.add(BatchNormalization())
        model.add(Dense(neurons, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(Dense(numclasses, activation='sigmoid')) # used softmax in HWT_mode to add to 1

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss="binary_crossentropy" # in HWT_mode, I used categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=[MeanSquaredError(), brier_skill_score, AUC(), "accuracy"])

    return model


def modedate(ifiles, target_dates):
    target_dates = pd.to_datetime(target_dates)
    start, end = target_dates.min(), target_dates.max()
    pattern = r'/20\d\d[01][0-9][0123][0-9][012][0-9][0-5][0-9]/'
    filtered_ifiles = []
    for ifile in ifiles:
        yyyymmddhhmm = re.search(pattern, ifile).group().lstrip("/").rstrip("/")
        itime = pd.to_datetime(yyyymmddhhmm, format='%Y%m%d%H%M')
        if itime >= start and itime <= end:
            filtered_ifiles.append(ifile)
    return filtered_ifiles



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

    parser = get_argparser()
    parser.add_argument('--fhr', nargs="+", type=int, default=list(range(1,49)), help="forecast hour")
    parser.add_argument('--fits', nargs="+", type=int, default=None, help="work on specific fit(s) so you can run many in parallel")
    parser.add_argument('--folds', nargs="+", type=int, default=None, help="work on specific fold(s) so you can run many in parallel")
    parser.add_argument('--seed', type=int, default=None, help="random number seed for reproducability")

    args = parser.parse_args()
    logging.info(args)

    # Assign arguments to simple-named variables
    batchnorm             = args.batchnorm 
    batchsize             = args.batchsize
    clobber               = args.clobber
    debug                 = args.debug
    dropout               = args.dropout
    epochs                = args.epochs
    flash                 = args.flash
    fhr                   = args.fhr
    fits                  = args.fits
    folds                 = args.folds
    nfit                  = args.nfits
    glm                   = args.glm
    kfold                 = args.kfold
    layer                 = args.layers
    learning_rate         = args.learning_rate
    reg_penalty           = args.reg_penalty
    model                 = args.model
    neurons               = args.neurons
    optimizer             = args.optimizer
    rptdist               = args.rptdist
    savedmodel            = args.savedmodel
    seed                  = args.seed
    train_test_split_time = args.splittime
    suite                 = args.suite
    twin                  = args.twin


    if debug:
        logging.basicConfig(level=logging.DEBUG)

    if seed:
        logging.info(f"random seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Could be 'adam' or SGD from Sobash 2020
    optimizer = get_optimizer(optimizer, learning_rate=learning_rate)

    ### saved model name ###
    if savedmodel:
        pass
    else:
        savedmodel = savedmodel_default(args, fhr_str=make_fhr_str(fhr)) # abbreviate list of forecast hours with hyphens (where possible) so model name is not too long for tf. 
    logging.info(f"savedmodel={savedmodel}")

    ##################################

    #mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
    mask = pickle.load(open('./usamask.pk', 'rb'))
    height, width =  65,93
    mask = mask.reshape((height,width))
    mask = xarray.DataArray(mask,dims=["y","x"])
    logging.info(f"Make x and y indices actual coordinates of mask")
    mask = mask.assign_coords(dict(x=mask.x,y=mask.y))
    if False:
        ax = plt.axes(projection = G211.g211)
        xs = G211.xs
        ys = G211.ys


    logging.info(f"Read {model} predictors. Use parquet file, if it exists. If it doesn't exist, create it.")
    if model == "HRRR":
        alsoHRRRv4 = False
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.par'
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.par'
        if debug: ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.noN7.fastdebug.par'
        if alsoHRRRv4: ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRXHRRR.32bit.noN7.par'
        scalingfile = f"/glade/work/ahijevyc/NSC_objects/{model}/scaling_values_all_HRRRX.pk"
    elif model == "NSC3km-12sec":
        ifile = f'{model}{glmstr}.par'
        scalingfile = f"scaling_values_{model}_{train_test_split_time:%Y%m%d_%H%M}.pk"

    if os.path.exists(ifile):
        logging.info(f'reading {ifile}')
        df = pd.read_parquet(ifile, engine="pyarrow")
    else:
        # Define ifiles, a list of input files from glob.glob method
        if model == "HRRR":
            search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRRX_d01_20*00-0000.par' # just 00z 
            if debug:
                search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRRX_d01_2020060*00-0000.par' # just 00z 
            ifiles = glob.glob(search_str)
            if alsoHRRRv4:
                search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_202[01]*00-0000.par' # HRRR, not HRRRX. no 2022 (yet) 
                ifiles.extend(glob.glob(search_str))
        elif model == "NSC3km-12sec":
            search_str = f'/glade/work/sobash/NSC_objects/grid_data/grid_data_{model}_d01_20*00-0000.par'
            ifiles = glob.glob(search_str)

        # remove largest neighborhood size (fields containing N7 in the name)
        df = pd.read_parquet(ifiles[0], engine="pyarrow")
        columns2read = df.columns
        N7_columns = [x for x in df.columns if "-N7" in x]
        logging.info(f"ignoring {len(N7_columns)} N7 columns: {N7_columns}")
        columns2read = set(df.columns) - set(N7_columns)

        # Read list of columns columns2read (includes severe reports)
        logging.info(f"Reading {len(ifiles)} {model} files {search_str}")
        df = pd.concat( pd.read_parquet(ifile, engine="pyarrow", columns=columns2read) for ifile in ifiles) # pd.read_parquet only handles one file at a time, so pd.concat
        logging.info("done")

        # Index df and modeds the same way. 
        logging.info(f"convert df Date to datetime64[ns]")
        df["Date"] = df.Date.astype('datetime64[ns]')
        df = df.rename(columns=dict(yind="y",xind="x",Date="initialization_time",fhr="forecast_hour"))
        df["valid_time"] = pd.to_datetime(df["initialization_time"]) + df["forecast_hour"].astype(int) * datetime.timedelta(hours=1)
        df = df.set_index(["y","x","initialization_time","forecast_hour"])

        if model.startswith("NSC3km"):
            # Read mode probabilities
            search_str = f'/glade/scratch/cbecker/NCAR700_objects/output_object_based/evaluation_zero_filled/20*/label_probabilities_20*00_fh_*.nc'
            ifiles = sorted(glob.glob(search_str))
            logging.info(f"Found {len(ifiles)} storm mode files")
            ifiles = modedate(ifiles, df.initialization_time)
            logging.info(f"Read {len(ifiles)} storm mode files in date range of {model} DataFrame")
            modeds = xarray.open_mfdataset( ifiles, preprocess=lambda x: x.set_index(time=['init_time','forecast_hour']) )
            logging.info(f"Unstack time dimension")
            modeds = modeds.unstack('time')
            logging.info(f"Make x and y indices actual coordinates, so we don't lose track when we mask and drop")
            modeds = modeds.assign_coords(dict(x=modeds.x,y=modeds.y))
            logging.info(f"Use CONUS mask and drop points outside CONUS")
            modeds = modeds.where(mask, drop=True)
            logging.info(f"reset xarray Dataset coordinates [lon, lat] to variables")
            modeds = modeds.reset_coords(["lon","lat"]) 

            # In modeds, x : west to east, y : south to north
            # In sobash df, xind : south to north, yind : west to east.
            modeds = modeds.rename(dict(x="y",y="x",init_time="initialization_time")) #dimensions are renamed to match df.

            logging.info(f"merge {model} DataFrame with mode Dataset in xarray")
            ds = df.to_xarray().merge(modeds, join="inner", compat="override") # if you don't use join="inner" you get a lot of NaNs. override to ignore lat/lon mismatch (TODO: why mismatch?)
            df = ds.to_dataframe().dropna(axis="index") # for some reason there are still nans even after using join="inner" above.

        # Derived fields
        df["dayofyear"] = df["valid_time"].dt.dayofyear
        df["Local_Solar_Hour"] = df["valid_time"].dt.hour + df["lon"]/15
        df = decompose_circular_feature(df, "dayofyear", period=365.25)
        df = decompose_circular_feature(df, "Local_Solar_Hour", period=24)
        logging.info("convert 64-bit to 32-bit columns")
        dtype_dict =      {k:np.float32 for k in df.select_dtypes(np.float64).columns}
        dtype_dict.update({k:np.int32   for k in df.select_dtypes(np.int64).columns})
        df = df.astype(dtype_dict, copy=False)
        df = df.reset_index().set_index(["valid_time","y","x"])

        earliest_valid_time = df.index.get_level_values(level="valid_time").min()
        latest_valid_time = df.index.get_level_values(level="valid_time").max()
        assert latest_valid_time > pd.to_datetime("20160101"), "DataFrame completely before GLM exists"
        glmds = get_glm(twin, rptdist)
        glmds = glmds.sel(valid_time = slice(earliest_valid_time,latest_valid_time)) # Trim GLM to time window of model data
        logging.info(f"Merge flashes with {model} DataFrame")
        # In glmds, x : west to east, y : south to north
        # In sobash df, xind : south to north, yind : west to east.
        glmds = glmds.rename(dict(x="y",y="x")) #dimensions are renamed to match sobash df.
        df = df.merge(glmds.to_dataframe(), left_on=["valid_time","y","x"], right_on=["valid_time","y","x"])
        #Do {model} and GLM overlap at all?"
        assert not df.empty, f"Merged {model}/GLM Dataset is empty."
        # Sanity check--make sure prediction model and GLM grid box lat lons are similar
        assert (df.lon_y - df.lon_x).max() < 0.1, f"{model} and glm longitudes don't match"
        assert (df.lat_y - df.lat_x).max() < 0.1, f"{model} and glm lats don't match"
        df = df.drop(columns=["lon_y","lat_y"])
        df = df.rename(columns=dict(lon_x="lon",lat_x="lat")) # helpful for scale factor pickle file.
    
        logging.info(f"writing {ifile}")
        df.to_parquet(ifile)

    # Convert distance to closest storm report to True/False based on distance and time thresholds 
    df, rptcols = rptdist2bool(df, rptdist, twin)

    if glm:
        df["flashes"] = df["flashes"] >= flash
        rptcols.append("flashes")


    plotclimo=False
    if plotclimo:
        mdf = df.groupby(["lon","lat"]).sum() # for debug plot
        fig, axes = plt.subplots(nrows=3,ncols=2)
        for label,ax in zip(rptcols, axes.flatten()):
            im = ax.scatter(mdf.index.get_level_values("lon"), mdf.index.get_level_values("lat"), c=mdf[label]) # Don't use mdf (sum) for lon/lat
            ax.set_title(label)
            fig.colorbar(im, ax=ax)
        plt.show()

    logging.info(f"Sort by valid_time and speed things up by dropping multiindex")
    df = df.sort_index(level="valid_time", ignore_index=True)



    if "HAILCAST_DIAM_MAX" in df and (df["HAILCAST_DIAM_MAX"] == 0).all():
        logging.info("HAILCAST_DIAM_MAX all zeros. Dropping.")
        df = df.drop(columns="HAILCAST_DIAM_MAX")



    if "with_storm_mode" not in suite:
        logging.info("making sure predictors don't include storm mode")
        storm_mode_columns = [x for x in df.columns if "SS_" in x or "NN_" in x]
        df = df.drop(columns=storm_mode_columns)






    # Split labels away from predictors
    labels = df[rptcols] # converted to Boolean above
    df = df.drop(columns=rptcols)






    df.info()
    print(labels.sum())




    # Split into training and testing cases

    # HRRRv3 to v4 at 20201202 0z.
    idate = df.initialization_time.astype('datetime64[ns]') # used for train_test_split_time 
    df = df.drop(columns="initialization_time")
    logging.info(f"train test split time {train_test_split_time}")
    test_idx  = (idate >= train_test_split_time) & (idate < pd.to_datetime("20211031")) # Mar 1 - Oct 31 2021 # changed > to >= on Jun 28, 2022
    train_idx = ~test_idx
    df_train = df[train_idx]
    df_test  = df[test_idx]
    train_labels = labels[train_idx]
    test_labels  = labels[test_idx]

    assert all(train_labels.sum()) > 0, "some classes have no True labels in training set"


    logging.info("normalize data using training set")
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

    cv = KFold(n_splits=kfold) 

    plot_splits= False
    if plot_splits:
        fig, ax = plt.subplots()
        y = train_labels["any_rptdist_2hr"]
        n_splits = kfold
        visualizecv.plot_cv_indices(cv, df_train, y, None, ax, n_splits)
        plt.show()


    for ifold, (train_split, test_split) in enumerate(cv.split(df_train, train_labels)): # train_labels has no effect for KFold, but for GroupKFold, it, along with groups argument would affect folds.
        if folds and ifold not in folds: continue # just do specific folds (needed for OOM issue in 1024-neuron GPU cases)

        # train model
        if not fits: # if user did not ask for specific fits, assume fits range from 0 to nfit-1.
            fits = range(0,nfit)
        for i in fits:
            model_i = f"nn/nn_{savedmodel}_{i}/{kfold}fold{ifold}"
            if not clobber and os.path.exists(model_i) and os.path.exists(f"{model_i}/config.yaml"):
                logging.info(f"{model_i} exists")
            else:
                logging.info(f"fitting {model_i}")
                model = baseline_model(input_dim=df_train.columns.size, numclasses=train_labels.columns.size, neurons=neurons[0], layer=layer, name=f"fit_{i}",
                        kernel_regularizer=L2(l2=reg_penalty), optimizer=optimizer, dropout=dropout)
                history = model.fit(df_train.iloc[train_split].to_numpy(dtype='float32'), train_labels.iloc[train_split].to_numpy(dtype='float32'), class_weight=None, 
                        sample_weight=None, batch_size=batchsize, epochs=epochs, verbose=2)
                logging.debug(f"saving {model_i}")
                model.save(model_i)
                del(history)
                del(model)
                # Save order of columns, scaling factors, all arguments.
                with open(os.path.join(model_i, "config.yaml"), "w") as yfile:
                    yaml.dump(
                            dict(columns=df_train.columns.to_list(),
                                mean=sv.loc["mean"].reindex(df_train.columns).to_list(),
                                std=sv.loc["std"].reindex(df_train.columns).to_list(),
                                labels=train_labels.columns.to_list(),
                                args=args,
                                ), yfile)


if __name__ == "__main__":
    main()
