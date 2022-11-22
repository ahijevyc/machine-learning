import argparse
import datetime
import G211
import glob
from hwtmode.data import decompose_circular_feature
import logging
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score, get_argparser, get_features, get_glm, get_optimizer, rptdist2bool, savedmodel_default
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
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.regularizers import L2
import visualizecv # custom script by ahijevyc modified from sklearn web page
import xarray
import yaml

def baseline_model(input_dim=None, name=None, numclasses=None, neurons=16, layer=2, kernel_regularizer=None, 
        optimizer='adam', dropout=0, batch_normalize=False, learningrate=0.01):

    # Discard any pre-existing version of the model.
    model = tf.keras.models.Sequential(name=name)
    # Previously used Dense here by providing input_dim, neurons, and activation args. Dense also made an input (layer). 
    model.add(tf.keras.Input(shape=input_dim)) 
    for i in range(layer): # add dropout, batch normalization, and kernel regularization, even with only one layer. 
        model.add(tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=kernel_regularizer))
        model.add(Dropout(rate=dropout))
        if batch_normalize: model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(numclasses, activation='sigmoid')) # used softmax in HWT_mode to add to 1

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss="binary_crossentropy" # in HWT_mode, I used categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=[MeanSquaredError(), brier_skill_score, AUC(), "accuracy"])

    return model


def modedate(modeprob_files, model_dates):
    model_dates = pd.to_datetime(np.unique(model_dates))
    pattern = r'/20\d\d[01][0-9][0123][0-9][012][0-9][0-5][0-9]'
    filtered_ifiles = []
    started_with = len(modeprob_files)
    for ifile in modeprob_files:
        yyyymmddhhmm = re.search(pattern, ifile).group().lstrip("/")
        itime = pd.to_datetime(yyyymmddhhmm, format='%Y%m%d%H%M')
        if itime in model_dates:
            filtered_ifiles.append(ifile)
        else:
            logging.debug(f"ignoring modeprob file {itime}. no matching model file")
    logging.info(f"Kept {len(filtered_ifiles)}/{started_with} mode prob files")
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
    batchnorm      = args.batchnorm 
    batchsize      = args.batchsize
    clobber        = args.clobber
    debug          = args.debug
    dropout        = args.dropout
    epochs         = args.epochs
    flash          = args.flash
    fhr            = args.fhr
    fits           = args.fits
    folds          = args.folds
    glm            = args.glm
    kfold          = args.kfold
    layer          = args.layers
    learning_rate  = args.learning_rate
    model          = args.model
    neurons        = args.neurons
    nfit           = args.nfits
    optimizer      = args.optimizer
    reg_penalty    = args.reg_penalty # L2
    rptdist        = args.rptdist
    savedmodel     = args.savedmodel
    seed           = args.seed
    trainend       = args.trainend
    trainstart     = args.trainstart
    suite          = args.suite
    twin           = args.twin


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
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.par'
        ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRXHRRR.32bit.par'
        if debug: ifile = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.fastdebug.par'
    elif model.startswith("NSC"):
        ifile = f'{model}.par'

    if os.path.exists(ifile):
        logging.info(f'reading {ifile}')
        df = pd.read_parquet(ifile, engine="pyarrow")
    else:
        # Define ifiles, a list of input files from glob.glob method
        if model == "HRRR":
            search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRRX_d01_20*00-0000.par' # just 00z 
            if debug:
                search_str = search_str.replace("*", "2006*") # just June 2020
            ifiles = glob.glob(search_str)
            search_str = f'/glade/work/sobash/NSC_objects/HRRR_new/grid_data/grid_data_HRRR_d01_202*00-0000.par' # append HRRR to HRRRX.
            ifiles.extend(glob.glob(search_str))
        elif model.startswith("NSC"):
            search_str = f'/glade/work/sobash/NSC_objects/grid_data_new/grid_data_{model}_d01_20*00-0000.par'
            if debug:
                search_str = f'/glade/work/sobash/NSC_objects/grid_data_new/grid_data_{model}_d01_201504*00-0000.par' # smaller subset for debugging
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

        # Index df and modeds the same way. 
        logging.info(f"convert df Date to datetime64[ns]")
        df["Date"] = df.Date.astype('datetime64[ns]')
        df = df.rename(columns=dict(yind="y",xind="x",Date="initialization_time",fhr="forecast_hour"))
        logging.info(f"derive valid_time from initialization_time + forecast_hour")
        df["valid_time"] = pd.to_datetime(df["initialization_time"]) + df["forecast_hour"].astype(int) * datetime.timedelta(hours=1)
        df = df.set_index(["y","x","initialization_time","forecast_hour"])

        if model.startswith("NSC3km"):
            # Read mode probabilities
            use_hourly_files = False
            if use_hourly_files:
                search_str = f'/glade/scratch/cbecker/NCAR700_objects/output_object_based/evaluation_zero_filled/20*/label_probabilities_20*00_fh_*.nc'
                ifiles = sorted(glob.glob(search_str))
                logging.info(f"Found {len(ifiles)} storm mode probability files")
                ifiles = modedate(ifiles, df.index.get_level_values("initialization_time"))
                logging.info(f"Read {len(ifiles)} storm mode files in date range of {model} DataFrame")
                # reset_coords to avoid xarray.core.merge.MergeError: unable to determine if these variables should be coordinates or not in the merged result: {'valid_time'}
                # set_index to avoid ValueError: Could not find any dimension coordinates to use to order the datasets for concatenation
                modeds = xarray.open_mfdataset( ifiles, preprocess=lambda x: x.reset_coords(["lon","lat",'valid_time']).set_index(time=['init_time','forecast_hour']),
                        combine="nested", parallel=True, compat="override", combine_attrs="override") # parallel is faster
            # Try nco concat files from ~/bin/modeprob_concat.csh. Faster than reading individual forecast hour files.
            search_str = f'/glade/scratch/ahijevyc/NCAR700_objects/output_object_based/evaluation_zero_filled/20??????0000.nc'
            ifiles = sorted(glob.glob(search_str))
            logging.info(f"Ignore mode prob files for times that are not present in features DataFrame")
            ifiles = modedate(ifiles, df.index.get_level_values("initialization_time"))
            logging.info(f"Open and combine {len(ifiles)} storm mode probability files")
            # Tried open_mfdataset, but got missing values for all but the first initialization time.
            modeds = xarray.combine_nested([xarray.open_dataset(ifile) for ifile in ifiles], concat_dim="time")
            no_time_dim = ["lat","lon","forecast_hour"]
            logging.info("Remove time dimension from {remove_time_dimension}")
            for e in no_time_dim:
                modeds[e] = modeds[e].isel(time=0)
            modeds = modeds.swap_dims(dict(record="forecast_hour",time="init_time")) 
            logging.info(f"Make x and y indices actual coordinates so we can apply similarly-structured CONUS mask")
            modeds = modeds.assign_coords(dict(x=modeds.x,y=modeds.y))
            logging.info(f"Use CONUS mask and drop points outside CONUS")
            modeds = modeds.where(mask, drop=True) #  even with drop=True you still have nans. mask is not a box. It has irregular disjointed edges.

            # In modeds, x : west to east, y : south to north
            # In sobash df, xind : south to north, yind : west to east.
            logging.info(f"Rename mode prob dimensions to match index names of df {df.index.names}")
            modeds = modeds.rename(dict(x="y",y="x",init_time="initialization_time"))
            logging.info(f"mode prob dimensions now {modeds.dims}")

            logging.info(f"merge {model} DataFrame with mode Dataset in xarray")
            # slash df fhrs to match modeds. modeds has fhr 12-35. Why? It is faster but would be nice to keep forecast hours 1-11. 
            # df = df.sort_index(level=[0,1,2,3]).loc[(slice(None),slice(None),slice(None),slice(12,35))]
            # Tried join="left" but it ignored all the modeds columns. Tried "inner" but it ignored df fhrs 1-11 and 36.
            ds = df.to_xarray().merge(modeds, join="outer", compat="override") #  override to ignore the lat/lon mismatch (assumed small)
            logging.info("Convert xarray Dataset to pandas DataFrame")
            df = ds.to_dataframe()
            # drop row if all columns are na. modeprobs are na for some forecast hours.
            logging.info(f"Drop rows with all NAs from {len(df)} row DataFrame")
            df = df.dropna(how="all")
            logging.info(f"{len(df)} remaining")

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

        if glm:
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
    df, rptcols = rptdist2bool(df, args)

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


    # HRRRv3 to v4 at 20201202 0z.
    logging.info(f"Use initialization times {trainstart} through {trainend} for training")
    before_filtering = len(df)
    train_idx  = (trainstart <= df.initialization_time) & (df.initialization_time <= trainend)
    df = df[train_idx]
    setattr(args, 'trainstart', df.initialization_time.min())
    setattr(args, 'trainend', df.initialization_time.max())
    logging.info(f"After filtering, trainstart={args.trainstart} trainend={args.trainend}")
    df = df.drop(columns="initialization_time")
    logging.info(f"keep {len(df)}/{before_filtering} cases for training")


    before_filtering = len(df)
    # Used to test all columns for NA, but we only care about the feature subset being complete. 
    # For example, mode probs are not avaiable for fhr=2 but we don't need to drop fhr=2 if
    # the other features are complete. 
    features = get_features(args)
    logging.info(f"Retain rows where all {len(features)} requested features are present")
    df = df.loc[df[features].notna().all(axis="columns"),:]
    logging.info(f"kept {len(df)}/{before_filtering} cases with no NA features")


    logging.info(f"Split {len(rptcols)} labels away from predictors")
    labels = df[rptcols] # labels converted to Boolean above
    df = df.drop(columns=rptcols)

    df.info()
    print(labels.sum())

    assert all(labels.sum()) > 0, "some classes have no True labels in training set"


    # This filtering of features must occur after initialization time is used and discarded, and after labels are separated and saved. 
    before_filtering = df.columns
    df = df[features]
    logging.info(f"dropped features {set(before_filtering) - set(df.columns)}")
    logging.info(f"kept {len(df.columns)}/{len(before_filtering)} features")


    logging.info(f"calculating mean and std scaling values")
    sv = df.describe()

    # Check for zero standard deviation. (can't normalize)
    stdiszero = sv.loc["std"] == 0
    if stdiszero.any():
        logging.error(f"{sv.columns[stdiszero]} std equals zero")
    logging.info(f"normalize data")
    df = (df - sv.loc["mean"]) / sv.loc["std"]
    logging.info('done normalizing')

    df.info()

    if df.describe().isna().any().any():
        logging.error(f"nan(s) in {df.columns[df.mean().isna()]}")

    if kfold > 1:
        cv = KFold(n_splits=kfold) 
        cvsplit = cv.split(df, labels) # labels has no effect for KFold, but for GroupKFold, it, along with groups argument would affect folds.
        plot_splits= False
        if plot_splits:
            fig, ax = plt.subplots()
            y = labels["any_rptdist_2hr"]
            n_splits = kfold
            visualizecv.plot_cv_indices(cv, df, y, None, ax, n_splits)
            plt.show()
    else:
        # Use everything for training. train_split is every index; test_split is empty
        cvsplit = [(np.arange(len(df)), [])]



    for ifold, (train_split, test_split) in enumerate(cvsplit): 
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
                tf.keras.backend.clear_session()
                model = baseline_model(input_dim=df.columns.size, numclasses=labels.columns.size, neurons=neurons[0], layer=layer, name=f"fit_{i}",
                        kernel_regularizer=L2(l2=reg_penalty), optimizer=optimizer, dropout=dropout)
                history = model.fit(df.iloc[train_split].to_numpy(dtype='float32'), labels.iloc[train_split].to_numpy(dtype='float32'), class_weight=None, 
                        sample_weight=None, batch_size=batchsize, epochs=epochs, verbose=2)
                logging.debug(f"saving {model_i}")
                model.save(model_i)
                del(history)
                del(model)
                # Save order of columns, scaling factors, all arguments.
                with open(os.path.join(model_i, "config.yaml"), "w") as yfile:
                    yaml.dump(
                            dict(columns=df.columns.to_list(),
                                mean=sv.loc["mean"].reindex(df.columns).to_list(),
                                std=sv.loc["std"].reindex(df.columns).to_list(),
                                labels=labels.columns.to_list(),
                                args=args,
                                ), yfile)



if __name__ == "__main__":
    main()
