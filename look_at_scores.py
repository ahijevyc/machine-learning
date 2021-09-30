#!/usr/bin/env python


def brier_score(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score(obs, preds):
    bs = brier_score(obs, preds)
    obs_climo = K.mean(obs, axis=0) # use each observed class frequency instead of 1/nclasses. Only matters if obs is multiclass.
    bs_climo = K.mean((obs - obs_climo) ** 2)
    bss = 1.0 - (bs/bs_climo+K.epsilon())
    return bss

def baseline_model(input_dim=None, numclasses=None, num_neurons=30, optimizer='adam', dropout=0):

    # Discard any pre-existing version of the model.
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=input_dim, activation='relu', name="convective_mode"))
    model.add(Dropout(rate=dropout))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(rate=dropout))
    if numclasses > 1:
        model.add(Dense(numclasses, activation='softmax', name="predictions")) # class probabilities add to 1
    else:
        model.add(Dense(numclasses, activation='sigmoid')) 

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss="binary_crossentropy"
    if numclasses > 1:
        loss="categorical_crossentropy"
    model.compile(loss=loss, optimizer=optimizer, metrics=[brier_score, brier_skill_score, AUC(), "accuracy"])

    return model

def get_features():
    features = ['UP_HELI_MAX_mean',
           'UP_HELI_MAX_max', 'GRPL_MAX_mean', 'GRPL_MAX_max',
           'WSPD10MAX_mean', 'WSPD10MAX_max', 
           'W_UP_MAX_mean', 'W_UP_MAX_max', 'W_DN_MAX_mean',
           'W_DN_MAX_max', 'W_DN_MAX_min', 'RVORT1_MAX_mean', 'RVORT1_MAX_max',
           'RVORT5_MAX_mean', 'RVORT5_MAX_max', 
           'UP_HELI_MAX03_mean', 'UP_HELI_MAX03_max', 
           'UP_HELI_MAX01_mean', 'UP_HELI_MAX01_max', 
           'UP_HELI_MIN_mean', 'UP_HELI_MIN_max', 
           'REFL_COM_mean', 'REFL_COM_max', 'REFL_1KM_AGL_mean',
           'REFL_1KM_AGL_max', 'REFD_MAX_mean', 'REFD_MAX_max',
           'PSFC_mean', 'PSFC_max', 'PSFC_min', 'T2_mean',
           'T2_max', 'T2_min', 'Q2_mean', 'Q2_max', 'Q2_min', 'TD2_mean',
           'TD2_max', 'TD2_min', 'SBLCL-potential_mean', 'SBLCL-potential_max',
           'SBLCL-potential_min', 'MLLCL-potential_mean', 'MLLCL-potential_max',
           'MLLCL-potential_min', 'SBCAPE-potential_mean', 'SBCAPE-potential_max',
           'SBCAPE-potential_min', 'MLCAPE-potential_mean', 'MLCAPE-potential_max',
           'MLCAPE-potential_min', 'MUCAPE-potential_mean', 'MUCAPE-potential_max',
           'MUCAPE-potential_min', 'SBCINH-potential_mean', 'SBCINH-potential_max',
           'SBCINH-potential_min', 'MLCINH-potential_mean', 'MLCINH-potential_max',
           'MLCINH-potential_min', 'SRH03-potential_mean', 'SRH03-potential_max',
           'SRH03-potential_min', 'SRH01-potential_mean', 'SRH01-potential_max',
           'SRH01-potential_min', 'PSFC-potential_mean', 'PSFC-potential_max',
           'PSFC-potential_min', 'T2-potential_mean', 'T2-potential_max',
           'T2-potential_min', 'Q2-potential_mean', 'Q2-potential_max',
           'Q2-potential_min', 'TD2-potential_mean', 'TD2-potential_max',
           'TD2-potential_min', '10-potential_mean', '10-potential_max',
           '10-potential_min', 'area', 'eccentricity', 'major_axis_length',
           'minor_axis_length', 'orientation', # 'Max_Hail_Size', # Max_Hail_Size always 0.0, hence, normalized value is NaN (divide by zero std)
           '10_min', '10_mean', '10_max', 'SHR1-potential_min',
           'SHR1-potential_mean', 'SHR1-potential_max', 'SHR6-potential_min',
           'SHR6-potential_mean', 'SHR6-potential_max']

    features = [ 'UP_HELI_MAX_max', 'major_axis_length', 'UP_HELI_MAX03_max', 'area', 'minor_axis_length',
           'eccentricity', 'UP_HELI_MAX_mean', 'UP_HELI_MAX01_max', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean', 
           'REFD_MAX_max', 'RVORT1_MAX_max', 'RVORT5_MAX_max', 'REFL_1KM_AGL_max', '10_max', 
           '10-potential_max', 'SHR6-potential_max', 'SHR6-potential_mean', 'WSPD10MAX_max', 'UP_HELI_MIN_mean']
    return features


def label_longname_dict():
    d = {"Q1":"QLCS\nQ1+Q2+S2", "S1":"Supercell\nS1+S3", "D1":"Disorganized\nD1+D2"}
    d = {"Q1":"QLCS\nQ1+Q2+S2", "S1":"Supercell\nS1+S3", "D1":"Cell\nD1", "D2":"Cluster\nD2"}
    return d

import argparse
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import plot_keras_history
import sys, os, pickle, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold 
from sklearn.preprocessing import LabelEncoder

def main():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "train/predict neural network",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', nargs="+", default=[10], type=int, help="nn training batch size")
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("--confmin", type=int, default=3, help="minimum confidence-throw out hand labels with lower confidence")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0], help='fraction of neurons to drop in each hidden layer (0-1)')
    parser.add_argument('--epochs', nargs="+", default=[35], type=int, help="number of training epochs")
    parser.add_argument('--freezetime', type=lambda s: pd.to_datetime(s), default=pd.to_datetime("2021-07-08 18:00"), help="ignore labels created at this time and later")
    parser.add_argument('--labelpick', type=str, choices=["first", "last", "all"], default="all", help="how to handle multiple labels on same storm")
    parser.add_argument('--model_fname', type=str, help="filename of machine learning model")
    parser.add_argument('--neurons', nargs="+", default=[12], type=int, help="number of neurons in each nn layer")
    parser.add_argument('--segmentations', nargs="*", type=str, default=["hyst"], help="storm segmentation approaches")
    parser.add_argument('--split', type=lambda s: pd.to_datetime(s), default=pd.to_datetime("2013-06-25"), help="train with storms before this time; test this time and after")
    parser.add_argument('--suite', type=str, default='basic', help="name for group of features")


    # Assign arguments to simple-named variables
    args = parser.parse_args()
    batch_sizes           = args.batchsize
    clobber               = args.clobber
    confmin               = args.confmin
    debug                 = args.debug
    dropouts              = args.dropouts
    epochs                = args.epochs
    freezetime            = args.freezetime
    labelpick             = args.labelpick
    savedmodel            = args.model_fname
    num_neurons           = args.neurons
    segmentations         = args.segmentations
    train_test_split_time = args.split
    suite                 = args.suite

    print(args)

    ### NEURAL NETWORK PARAMETERS ###

    nn_params = { 'num_neurons': num_neurons }
    dataset = 'NSC'
    trained_models_dir = '/glade/work/ahijevyc/NSC_objects'
    if savedmodel:
        pass
    else:
        savedmodel = ".".join(label_longname_dict().keys())+f"{labelpick}label.dropout{dropouts[0]}.h5"

    ##################################

    mcd = os.getenv("TMPDIR", "/glade/ahijevyc/scratch/temp") + "/HWT_mode_output/atts_and_labels_"+".".join(segmentations)+".csv"
    print(f'Reading {mcd}')
    if os.path.exists(mcd):
        df = pd.read_csv(mcd, parse_dates=["Run_Date","Valid_Date","labeltime"])
        df = df.sort_values("labeltime") # so you can pick first (or last) label of each storm
    else:
        print(f"Use ~ahijevyc/bin/hagelslag_obj_pdf.py to make {mcd}")
        sys.exit(1)

    df.info() # tried adding show_counts=True but got TypeError: info() got an unexpected keyword argument 'show_counts'
    features = get_features()

    # Confidence filter
    confidence = df['conf']
    nlowconf = (df.conf < confmin).sum()
    print(f"dropping {nlowconf}/{len(df)} hand labels with confidence < {confmin}")
    df = df[df.conf >= confmin]

    # Labeltime filter
    nlate = (df.labeltime >= freezetime).sum()
    print(f"dropping {nlate}/{len(df)} labeltimes {freezetime} or later")
    df = df[df.labeltime < freezetime]

    # Multiple labels filter
    if labelpick == "first":
        df = df.groupby("Step_ID").first()
    elif labelpick == "last":
        df = df.groupby("Step_ID").last()
    elif labelpick == "all":
        pass

    # Split labels into training and testing sets (training < train_test_split_time <= testing).
    train_indices = df["Valid_Date"] < train_test_split_time
    test_indices = ~train_indices
    df["split"] = "train"
    df.loc[test_indices,"split"] = "test"
    print(f"{train_indices.sum()} ({100.*train_indices.sum()/len(df):.0f}%) training cases < {train_test_split_time}")
    print(f"{test_indices.sum()} ({100.*test_indices.sum()/len(df):.0f}%) test cases >= {train_test_split_time}")

    labels = df['label'].astype("category")
    labels = labels.cat.reorder_categories(["Q1", "Q2", "S1", "S2", "S3", "D1", "D2"], ordered=True)
    # If you change these, also change label_longname_dict() to have the correct categories.
    # labels[labels =="D2"] = "D1" # D1/D2 
    labels[labels =="S3"] = "S1" # S1/S3
    labels[labels =="Q2"] = "Q1" 
    labels[labels =="S2"] = "Q1" # Q1/Q2/S2

    labels = labels.cat.rename_categories(label_longname_dict())

    labels = labels.cat.remove_unused_categories()
    numclasses =labels.nunique()
    encoder = LabelEncoder()
    encoder.fit(labels)
    assert all(encoder.classes_ == sorted(labels.cat.categories)) # normalize_and_topn.ipynb assumes labels were encoded alphabetically
    # Unfortunately, LabelEncoder encodes alphabetically and does not honor the Pandas category order.
    encoded_labels = encoder.transform(labels)
    onehotlabels = to_categorical(encoded_labels)
    print(f"{numclasses} classes")

    df = df[features]
    
    # normalize data 
    scaler = pickle.load(open('/glade/work/ahijevyc/NSC_objects/scaler.pkl', 'rb'))
    scaler = scaler[features] 
    # We'll want to normalize the whole thing.
    norm_in_data = (df - scaler.loc["mean"] ) / scaler.loc["std"]
    print('done normalizing')

    y_pred = []
    tbles=[]
    for s in [f"test{i}.D.h5" for i in range(5)]:
        custom_objects = {"brier_score":brier_score, "brier_skill_score":brier_skill_score}
        model = load_model(s, custom_objects=custom_objects)
        y_pred.append(model.predict(norm_in_data[test_indices]))
    y_pred = np.mean(y_pred, axis=0)
    for i, label in enumerate(labels.cat.categories):
        bss =  brier_skill_score(K.constant(onehotlabels[test_indices][:,i]), K.constant(y_pred[:,i]))
        row = {"label":label, "BSS": K.get_value(bss), "AUC":roc_auc_score(onehotlabels[test_indices][:,i], y_pred[:,i])}
        tbles.append(row)

    xx = pd.DataFrame(tbles)
    print(xx)
    pdb.set_trace()


    tbles = []
    for s in [f"test{i}.D.h5" for i in range(5)]:
        custom_objects = {"brier_score":brier_score, "brier_skill_score":brier_skill_score}
        model = load_model(s, custom_objects=custom_objects)
        y_pred = model.predict(norm_in_data[test_indices])
        for i, label in enumerate(labels.cat.categories):
            bss =  brier_skill_score(K.constant(onehotlabels[test_indices][:,i]), K.constant(y_pred[:,i]))
            row = {"label":label, "BSS": K.get_value(bss), "AUC":roc_auc_score(onehotlabels[test_indices][:,i], y_pred[:,i]), "s":s}
            tbles.append(row)

    #pd.options.display.float_format = '{:.3f}'.format
    xx = pd.DataFrame(tbles).groupby("label").mean()
    print(xx)
    xx = pd.DataFrame(tbles).groupby("label").std()
    print(xx)
if __name__ == "__main__":
    main()
