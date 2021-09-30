import HWT_mode_train
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import scalar2vector
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.utils import to_categorical 
from xgboost import XGBRegressor


neurons =50 
models = [LinearRegression(), LogisticRegression(), DecisionTreeRegressor(), 
RandomForestRegressor(), XGBRegressor(), KNeighborsRegressor(), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
models = [RandomForestRegressor(n_estimators=200), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
models = [MLPClassifier(hidden_layer_sizes=(neurons,neurons))]

fig, axes = plt.subplots(nrows=len(models), figsize=(12, len(models)*4.), sharex=True)
#plt.subplots_adjust(left=0.03,right=0.98,top=0.80)

suite = "long"
print(f"suite = {suite}")
features = HWT_mode_train.get_features(suite)
segmentations = ["hyst"]
breast_cancer=False
class_startswith = None
if breast_cancer:
    ifile = "/Users/ahijevyc/Downloads/WDBC.csv"
    X = df.iloc[:,2:]
    y = df.iloc[:,1] == "M" # malignant is True
else:
    getstatic = True # Read merged dataset that was already created by hagelslag_obj_pdf.py
    if getstatic:
        ifile = f"/glade/scratch/ahijevyc/temp/HWT_mode_output/{suite}_atts_and_expertlabels_"+".".join(segmentations)+".csv"
        df = pd.read_csv(ifile, parse_dates=["labeltime", "Run_Date", "Valid_Date"])
    else:
        ifile = "/Users/ahijevyc/Downloads/atts_and_labels.csv"
        df = pd.read_csv(ifile,header=0)
    df["seconds to classify"] = df["seconds to classify"].fillna(value=600)
    df.loc[df["seconds to classify"] > 600,"seconds to classify"] = 600
    X = df
    if "orientation" in X:
        X = scalar2vector.decompose_circ_feature(X, "orientation", scale2rad=2., drop=True)
    if "Valid_Hour_UTC" in X:
        X.loc[:,"Local_Solar_Hour"] = X["Valid_Hour_UTC"] + X["Centroid_Lon"]/15.
        X = scalar2vector.decompose_circ_feature(X, "Local_Solar_Hour", scale2rad=2.*np.pi/24., drop=True)
    X = X[features]
    #class_startswith = "D2"
    y = df["label"].str.startswith(class_startswith).astype(float) # MLPClassifier can't handle Boolean y
    y = LabelEncoder().fit_transform(df["label"])
    y = to_categorical(y) # onehot vector

scaler = StandardScaler()
standardized = scaler.fit_transform(X)

if len(models) == 1:
    axes = np.array(axes) # avoid AttributeError: 'AxesSubplot' object has no attribute 'flatten'
for ax, model in zip(axes.flatten(), models):
    print(type(model))
    scaledict = {"raw":X,"standardized": standardized}
    del(scaledict["raw"])
    for m in scaledict:
        standardized = scaledict[m]
        model.fit(standardized, y)
        std=None
        if type(model).__name__ in ["KNeighborsRegressor","MLPClassifier"]:
            scoring = 'neg_mean_squared_error'
            scoring = 'accuracy'
            n_repeats = 100 
            pi = permutation_importance(model, standardized, y, scoring=scoring, n_repeats=n_repeats)
            print(pd.DataFrame([pi.importances_mean, pi.importances_std], columns=features, index=["importances_mean","importances_std"]).T)
            pi =pd.DataFrame([pi.importances_mean, pi.importances_std], columns=features, index=["importances_mean","importances_std"]).T.sort_values(by="importances_mean", ascending=False)
            importance = pi.importances_mean
            std = pi.importances_std
            ax.set_ylabel(f"permutation importance\nn_repeats={n_repeats} neurons={neurons}\nscoring={scoring}")
        for attr in ["coef_","feature_importances_"]:
            if hasattr(model, attr):
                importance = getattr(model,attr)
                ax.set_ylabel(attr[0:-1])
        if hasattr(model, "estimators_"): # e.g. RandomForestClassifier
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        if isinstance(model,sklearn.linear_model._logistic.LogisticRegression):
            importance = importance[0]
        print(pi.reset_index())
   
        pi.plot.bar(ax=ax,y="importances_mean", yerr="importances_std", label=m, alpha=0.5)
        

    ax.set_xlabel(suite+" feature suite")
    ax.grid()
    
    if len(scaledict) > 1:
        ax.legend()
    ax.set_title(type(model).__name__)


plt.tight_layout()
if class_startswith:
    plt.suptitle(f"class starts with {class_startswith}")
pdb.set_trace()
plt.savefig("importance_barplot.png")

