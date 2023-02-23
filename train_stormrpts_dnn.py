import datetime
import logging
import matplotlib.pyplot as plt
from ml_functions import Dumper, get_argparser, get_features, get_optimizer, load_df, rptdist2bool, get_savedmodel_path
import numpy as np
import os
import pandas as pd
import pdb
import random
import re
import sys
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.regularizers import L2
import visualizecv  # custom script by ahijevyc modified from sklearn web page
import xarray
import yaml


def baseline_model(input_dim=None, name=None, numclasses=None, neurons=[16,16], kernel_regularizer=None,
                   optimizer_name='Adam', dropout=0, batch_normalize=False, learning_rate=0.01):

    # Discard any pre-existing version of the model.
    model = tf.keras.models.Sequential(name=name)
    # Previously used Dense here by providing input_dim, neurons, and activation args. Dense also made an input (layer).
    model.add(tf.keras.Input(shape=input_dim))
    # add dropout, batch normalization, and kernel regularization, even with only one layer.
    for n in neurons:
        model.add(tf.keras.layers.Dense(n, activation='relu',
                  kernel_regularizer=kernel_regularizer))
        model.add(Dropout(rate=dropout))
        if batch_normalize:
            model.add(BatchNormalization())
    # used softmax in HWT_mode to add to 1
    model.add(tf.keras.layers.Dense(numclasses, activation='sigmoid'))

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss = "binary_crossentropy"  # in HWT_mode, I used categorical_crossentropy
    optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=[
                  MeanSquaredError(), AUC(), "accuracy"])

    return model


# maybe delete this function. It returns a list of model_dates that have a corresponding mode file.
def modedate(modeprob_files, model_dates):
    logging.warning("I didn't think this was needed")
    sys.exit(1)
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
            logging.debug(
                f"ignoring modeprob file {itime}. no matching model file")
    logging.info(f"Kept {len(filtered_ifiles)}/{started_with} mode prob files")
    return filtered_ifiles


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():

    parser = get_argparser()

    args = parser.parse_args()
    logging.info(args)

    # Assign arguments to simple-named variables
    batchnorm = args.batchnorm
    batchsize = args.batchsize
    clobber = args.clobber
    debug = args.debug
    dropout = args.dropout
    epochs = args.epochs
    fhr = args.fhr
    fits = args.fits
    folds = args.folds
    glm = args.glm
    kfold = args.kfold
    learning_rate = args.learning_rate
    neurons = args.neurons
    nfit = args.nfits
    optimizer_name = args.optimizer
    reg_penalty = args.reg_penalty  # L2
    rptdist = args.rptdist
    seed = args.seed
    testend = args.testend
    teststart = args.teststart
    trainend = args.trainend
    trainstart = args.trainstart
    suite = args.suite

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    if seed:
        logging.info(f"random seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Error if requested training and test period overlap and kfold == 1.
    overlap = min([trainend, testend]) - max([trainstart, teststart])
    assert overlap < datetime.timedelta(
        hours=0) or kfold > 1, f"training and testing periods overlap {trainstart}-{trainend}|{teststart}-{testend}"

    ### saved model name ###
    savedmodel = get_savedmodel_path(args)
    logging.info(f"savedmodel={savedmodel}")

    ##################################


    df = load_df(args)

    # Convert distance to closest storm report to True/False based on distance and time thresholds
    # And convert flash count to True/False based on distance, time, and flash thresholds
    df, label_cols = rptdist2bool(df, args)

    plotclimo = False
    if plotclimo:
        mdf = df.groupby(["lon", "lat"]).sum()  # for debug plot
        fig, axes = plt.subplots(nrows=3, ncols=2)
        for label, ax in zip(label_cols, axes.flatten()):
            im = ax.scatter(mdf.index.get_level_values("lon"), mdf.index.get_level_values(
                "lat"), c=mdf[label])  # Don't use mdf (sum) for lon/lat
            ax.set_title(label)
            fig.colorbar(im, ax=ax)
        plt.show()

    logging.info(
        f"Sort by valid_time and speed things up by dropping multiindex")
    df = df.sort_index(level="valid_time", ignore_index=True)

    # HRRRv3 to v4 at 20201202 0z.
    logging.info(
        f"Use initialization times {trainstart} through {trainend} for training")
    before_filtering = len(df)
    train_idx = (trainstart <= df.initialization_time) & (
        df.initialization_time <= trainend)
    df = df[train_idx]
    setattr(args, 'trainstart', df.initialization_time.min())
    setattr(args, 'trainend', df.initialization_time.max())
    logging.info(
        f"After trimming, trainstart={args.trainstart} trainend={args.trainend}")
    logging.info(f"keep {len(df)}/{before_filtering} cases for training")

    beforedropna = len(df)
    # Used to test all columns for NA, but we only care about the feature subset being complete.
    # For example, mode probs are not available for fhr=2 but we don't need to drop fhr=2 if
    # the other features are complete.
    feature_list = get_features(args)
    logging.info(
        f"Retain rows where all {len(feature_list)} requested features are present")
    df = df.dropna(axis="index", subset=feature_list)
    logging.info(
        f"kept {len(df)}/{beforedropna} cases with no NA features")

    before_filtering = len(df)
    logging.info(f"Retain rows with requested forecast hours {fhr}")
    df = df.loc[df["forecast_hour"].isin(fhr)]
    logging.info(
        f"kept {len(df)}/{before_filtering} rows with requested forecast hours")

    logging.info(f"Split {len(label_cols)} labels away from predictors")
    labels = df[label_cols]  # labels converted to Boolean above

    df.info()
    print(labels.sum())

    assert all(
        labels.sum()) > 0, "some classes have no True labels in training set"

    # This extraction of features must occur after initialization time is used and after labels are separated and saved.
    before_filtering = df.columns
    df = df[feature_list]
    logging.info(f"dropped {set(before_filtering) - set(df.columns)}")
    logging.info(f"kept {len(df.columns)}/{len(before_filtering)} features")

    logging.info(f"calculating mean and standard dev scaling values")
    sv = df.describe()

    # Check for zero standard deviation. (can't normalize)
    stdiszero = sv.loc["std"] == 0
    if stdiszero.any():
        logging.error(f"{sv.columns[stdiszero]} standard dev equals zero")
    logging.info(f"normalize data")
    df = (df - sv.loc["mean"]) / sv.loc["std"]
    logging.info('done normalizing')

    if df.describe().isna().any().any():
        logging.error(f"nan(s) in {df.columns[df.mean().isna()]}")

    if kfold > 1:
        cv = KFold(n_splits=kfold)
        # labels has no effect for KFold, but for GroupKFold, it, along with groups argument would affect folds.
        cvsplit = cv.split(df, labels)
        plot_splits = False
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
        if folds and ifold not in folds:
            # just do specific folds (needed for OOM issue in 1024-neuron GPU cases)
            continue

        # train model
        # if user did not ask for specific fits, assume fits range from 0 to nfit-1.
        if not fits:
            fits = range(0, nfit)
        for i in fits:
            model_i = f"{savedmodel}_{i}/{kfold}fold{ifold}"
            if not clobber and os.path.exists(model_i) and os.path.exists(f"{model_i}/config.yaml"):
                logging.info(f"{model_i} exists")
            else:
                logging.info(f"fitting {model_i}")
                model = baseline_model(input_dim=df.columns.size, numclasses=labels.columns.size, neurons=neurons, name=f"fit_{i}",
                                       kernel_regularizer=L2(l2=reg_penalty), optimizer_name=optimizer_name, dropout=dropout, learning_rate=learning_rate)
                model.fit(df.iloc[train_split].to_numpy(dtype='float32'), labels.iloc[train_split].to_numpy(dtype='float32'), class_weight=None,
                                    sample_weight=None, batch_size=batchsize, epochs=epochs, verbose=2)
                logging.debug(f"saving {model_i}")
                model.save(model_i)
                del model
                tf.keras.backend.clear_session()
                # Save order of columns, scaling factors, all arguments.
                with open(os.path.join(model_i, "config.yaml"), "w") as yfile:
                    yaml.dump(
                        dict(columns=df.columns.to_list(),
                             mean=sv.loc["mean"].reindex(df.columns).to_list(),
                             std=sv.loc["std"].reindex(df.columns).to_list(),
                             labels=labels.columns.to_list(),
                             args=args,
                             ), yfile) # , Dumper=Dumper)


if __name__ == "__main__":
    main()
