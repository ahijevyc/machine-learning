import datetime
import logging
import os
import pdb
import re
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.regularizers import L2
import xarray
from ml_functions import get_argparser, get_features, get_optimizer, load_df, rptdist2bool, get_savedmodel_path
import visualizecv  # custom script by ahijevyc modified from sklearn web page


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
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=None, metrics=[
        MeanSquaredError(), AUC(), "accuracy"])

    return model


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
    kfold = args.kfold
    label_cols = args.labels
    learning_rate = args.learning_rate
    neurons = args.neurons
    nfit = args.nfits
    optimizer_name = args.optimizer
    reg_penalty = args.reg_penalty  # L2
    seed = args.seed
    testend = args.testend
    teststart = args.teststart
    trainend = args.trainend
    trainstart = args.trainstart
    suite = args.suite

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    if seed:
        # If seed == -1 use the fit index as the seed
        if seed == -1:
            seed = fits[0]
        logging.info(f"set random seed {seed}")
        tf.keras.utils.set_random_seed(seed)

    overlap = min([trainend, testend]) - max([trainstart, teststart])
    if overlap > datetime.timedelta(hours=0):
        logging.warning(f"training and testing periods overlap [{trainstart},{trainend}) [{teststart},{testend}]")
        if kfold > 1:
            logging.warning(f"but overlap is okay. kfold={kfold} cross-validation separates train and test cases")
        else:
            # Exit if requested training and test period overlap and kfold == 1.
            sys.exit(1)

    ### saved model name ###
    savedmodel = get_savedmodel_path(args)
    logging.info(f"savedmodel={savedmodel}")

    ##################################


    df = load_df(args)

    # Convert distance to closest storm report to True/False based on distance and time thresholds
    # And convert flash count to True/False based on distance, time, and flash thresholds
    df = rptdist2bool(df, args)

    before_filtering = len(df)
    logging.info(
        f"Use initialization times in range [{trainstart}, {trainend}) for training")
    idx = (trainstart <= df.initialization_time) & (df.initialization_time < trainend)
    df = df[idx]
    setattr(args, 'trainstart', df.initialization_time.min())
    setattr(args, 'trainend', df.initialization_time.max())
    logging.info(
        f"After trimming, trainstart={args.trainstart} trainend={args.trainend}")
    logging.info(f"keep {len(df)}/{before_filtering} cases for training")

    feature_list = get_features(args)

    before_filtering = len(df)
    logging.info(f"Retain rows with requested forecast hours {fhr}")
    df = df.loc[df["forecast_hour"].isin(fhr)]
    logging.info(
        f"kept {len(df)}/{before_filtering} rows with requested forecast hours")

    logging.info(f"Split {len(label_cols)} requested labels away from predictors")
    labels = df[label_cols]  # labels converted to Boolean above

    # info method prints summary of DataFrame and returns None.
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

    logging.info('checking for nans in DataFrame')
    if df.isna().any().any():
        logging.error(f"nan(s) in {df.columns[df.mean().isna()]}")

    if kfold > 1:
        cv = KFold(n_splits=kfold)
        # labels has no effect for KFold, but for GroupKFold, it, along with groups argument would affect folds.
        cvsplit = cv.split(df, labels)
        plot_splits = False
        if plot_splits:
            fig, ax = plt.subplots()
            y = labels["any_40km_2hr"]
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
                                       kernel_regularizer=L2(l2=reg_penalty), optimizer_name=optimizer_name, dropout=dropout, 
                                       learning_rate=learning_rate)
                model.fit(df.iloc[train_split].to_numpy(dtype='float32'), labels.iloc[train_split].to_numpy(dtype='float32'), 
                        class_weight=None, sample_weight=None, batch_size=batchsize, epochs=epochs, verbose=2)
                logging.info(f"saving {model_i}")
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
                             ), yfile)


if __name__ == "__main__":
    main()
