from collections import defaultdict
import datetime
import glob
import visualizecv
import HWT_mode_train
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import scalar2vector
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import sys
from tensorflow.keras.utils import to_categorical 
import xarray
#from xgboost import XGBRegressor

#np.random.seed(14)

neurons = 12 
batch_size = 10 # Default is min(200, n_samples) if batch_size is around 25 or larger, ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
#models = [LinearRegression(), LogisticRegression(), DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor(), KNeighborsRegressor(), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
#models = [RandomForestRegressor(n_estimators=150), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
models = [] # if you just want matrix and dendrogram
models = [MLPClassifier(hidden_layer_sizes=(neurons,neurons),batch_size=batch_size)]
scoring = 'accuracy'

dist_thresholds = [1]
dist_thresholds = [2.5, 2, 1.5, 1, 0.5, .25, 0]


def corr_dendro_plot(X, suite=None, dist_thresholds=dist_thresholds, importances=None, figh=14):
    features = X.columns
    corr = spearmanr(X).correlation # just training data

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    for d in dist_thresholds:
        fig = plt.figure(figsize=(18, figh))
        ax1 = plt.subplot2grid((1,3),(0,0), colspan=2)
        ax2 = plt.subplot2grid((1,3),(0,2), colspan=1)
        ax2.grid(axis="x")
        ax2.axvline(x=d, linestyle="dashed", linewidth=2)
        ax2.set_xlabel("distance")
        dendro = hierarchy.dendrogram(
            dist_linkage, labels=X.columns, ax=ax2, orientation='right', color_threshold=d
        )
        ax2.invert_yaxis() # so order of labels in dendrogram matches correlation matrix
        yticklabels = ax2.get_ymajorticklabels()
        dendro_idx = np.arange(0, len(dendro["ivl"]))

        ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap=sns.color_palette("vlag", as_cmap=True))
        ax1.set_title("Spearman rank correlation between features in training data")
        ax1.set_xticks(dendro_idx)
        ax1.set_yticks(dendro_idx)
        ax1.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax1.set_yticklabels(dendro["ivl"])

        cluster_ids = hierarchy.fcluster(dist_linkage, d, criterion="distance")

        if d>0 and importances is not None:
            # merge importance and cluster_id columns into single DataFrame
            pm = pd.concat([importances, pd.Series(cluster_ids, index=features, name="cluster_id")], axis=1)
            selected_features = pm.groupby("cluster_id")["importance"].idxmax() # get feature with max importance
            print(len(features), "to", len(selected_features), "features", selected_features.to_list())
            fontsize=yticklabels[0].get_fontsize() + d*8 # starting with current ticklabel fontsize get bigger to the right.
            for c, fs in pm.groupby("cluster_id"):
                # label one member of each cluster
                # Find the indices corresponding to this cluster and take the average tickvalue
                ii = [dendro["ivl"].index(f) for f in fs.index]
                y0 = np.mean(ax2.get_yticks()[ii])
                ax2.text(d, y0, selected_features[c], va="center", ha="center", fontsize=fontsize) 


        fig.tight_layout()
        ofile = os.path.realpath(f"{suite}_{d}_dendro.png")
        plt.savefig(ofile)
        print(ofile)
        plt.close()
    return dist_linkage


suite = "long2"
#print(f"suite = {suite}")
figh=14


def main():
    features = HWT_mode_train.feature_dict[suite]
    segmentations = ["hyst"]
    class_startswith = None
    labelpick = "first"
    getstatic = True # Read merged dataset that was already created by hagelslag_obj_pdf.py
    if getstatic:
        ifile = f"/glade/scratch/ahijevyc/temp/HWT_mode_output/atts_and_expertlabels_"+".".join(segmentations)+".csv"
        df = pd.read_csv(ifile, parse_dates=["labeltime", "Run_Date", "Valid_Date"])
    else:
        ifile = "/Users/ahijevyc/Downloads/atts_and_labels.csv"
        df = pd.read_csv(ifile,header=0)

    # Which labels to pick?
    if labelpick == "first":
        df = df.groupby("Step_ID").first()
    elif labelpick == "last":
        df = df.groupby("Step_ID").last()
    elif labelpick == "all":
        pass

    df["seconds to classify"] = df["seconds to classify"].fillna(value=600)
    df.loc[df["seconds to classify"] > 600,"seconds to classify"] = 600

    X = df
    if "orientation" in X:
        X = scalar2vector.decompose_circ_feature(X, "orientation", period=np.pi)
    if "Valid_Hour_UTC" in X:
        X.loc[:,"Local_Solar_Hour"] = X["Valid_Hour_UTC"] + X["Centroid_Lon"]/15.
        X = scalar2vector.decompose_circ_feature(X, "Local_Solar_Hour", period=24)
    X = X[features]

    # split into train and test.
    train_idx = df.Valid_Date < pd.to_datetime("20130625")

    # standard scale features based on training set
    # assign numpy array returned by StandardScalar to original DataFrame. This preserves columns
    X[X.columns] = StandardScaler().fit(X[train_idx]).transform(X)

    # Without providing a list of importances, this won't label the best member from each cluster.
    dist_linkage = corr_dendro_plot(X[train_idx], suite=suite, dist_thresholds=dist_thresholds, figh=figh)

    seven_to_three_categories = True
    if seven_to_three_categories:
        df.loc[df["label"] =="D2","label"] = "D1" # D1/D2 
        df.loc[df["label"] =="S3","label"] = "S1" # S1/S3
        df.loc[df["label"] =="Q2","label"] = "Q1" 
        df.loc[df["label"] =="S2","label"] = "Q1" # Q1/Q2/S2


    if class_startswith:
        y = df["label"].str.startswith(class_startswith).astype(float) # MLPClassifier can't handle Boolean y
    else:
        y = LabelEncoder().fit_transform(df["label"])
        y = df["label"]
        #y = to_categorical(y) # onehot vector


    n_splits = 5 
    cv = GroupKFold(n_splits=n_splits)
    # Also tried StratifiedGroupKFold. GroupKFold tries to keep same number of groups in each fold; class percentanges are a little uneven.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold says
    # GroupKFold attempts to create balanced folds so # of distinct groups is approximately the same in each fold, whereas StratifiedGroupKFold attempts to 
    # create folds that preserve percentage of samples for each class as much as possible given the constraint of non-overlapping groups between splits.

    group = LabelEncoder().fit_transform(df.Run_Date)
    plot_splits= False
    if plot_splits:
        fig, ax = plt.subplots()
        visualizecv.plot_cv_indices(cv, X[train_idx], y[train_idx], group[train_idx], ax, n_splits)
        plt.show()

    n_repeats = 50
    explabel = f"neurons={neurons} {labelpick} label\nbatch size={batch_size} scoring={scoring}"
    explabel += f"\npermutation importance\n(mean drop in {scoring} from permuting feature)\nrepeated {n_repeats} times * {n_splits} cv splits"

    if not models:
        sys.exit(0)

    # HACK for just 1st model. get importances so we can choose the predictor with highest importance in each cluster 
    model = models[0]
    ofile = os.path.realpath(f"{suite}_{neurons}neurons_{labelpick}label_{class_startswith}_bs{batch_size}_{n_splits}x{n_repeats}_imp.csv")
    if os.path.exists(ofile):
        feature_df = pd.read_csv(ofile, index_col=0)
    else:
        print(f"permutation importances n={n_repeats}")
        pis = np.zeros((len(features), n_repeats*n_splits)) # empty numpy array, a row for each features, and a column for each importance
        for i, (train,test) in enumerate(cv.split(X[train_idx],y[train_idx],group[train_idx])):
            X_split = X[train_idx].iloc[train]
            y_split = y[train_idx][train]
            fittedestimator = model.fit(X_split, y_split)
            pi = permutation_importance(estimator=fittedestimator, X=X_split, y=y_split, scoring=scoring, n_repeats=n_repeats).importances
            pis[:,i*n_repeats:(i+1)*n_repeats] = pi
            print(f"cross validation split {i}/{n_splits}")
        feature_df = pd.DataFrame(pis, index=features)
        feature_df.to_csv(ofile)
    print(ofile)
    ax = feature_df.T.boxplot(figsize=(18, 5),rot=90)
    ax.set_ylabel(explabel)
    ax.set_title(type(model).__name__)
    plt.tight_layout()
    ofile = os.path.realpath(f"{suite}_{neurons}neurons_{labelpick}label_{class_startswith}_bs{batch_size}_imp_barplot.png")
    plt.savefig(ofile)
    print(ofile)
    plt.close()

    feature_df["importance"] = feature_df.mean(axis="columns")

    #Redo dendrograms with labels of best importance
    dist_linkage = corr_dendro_plot(X[train_idx], suite=suite, dist_thresholds=dist_thresholds, figh=figh, importances=feature_df["importance"])

    for dist_threshold in dist_thresholds:
        cluster_ids = hierarchy.fcluster(dist_linkage, dist_threshold, criterion="distance")
        feature_df["cluster_id"] = cluster_ids
        selected_features = feature_df.groupby("cluster_id")["importance"].idxmax()
        print(len(features), "to", len(selected_features), "features", selected_features.to_list())

        X_train_sel = X[train_idx][selected_features]

        if models:
            fig, axes = plt.subplots(ncols=len(models), figsize=(len(models)*5, figh))
            if len(models) == 1: axes = np.array(axes) # avoid AttributeError: 'AxesSubplot' object has no attribute 'flatten'
            for ax, model in zip(axes.flatten(),models):
                scores = cross_val_score(estimator=model, X=X_train_sel, y=y[train_idx], groups=group[train_idx], cv=cv)
                print(f"dist_threshold={dist_threshold} {type(model).__name__} {scores} mean: {np.array(scores).mean()} std: {np.array(scores).std()}")
                if type(model).__name__ in ["KNeighborsRegressor","MLPClassifier"]:
                    pis = np.zeros((len(selected_features), n_repeats*n_splits)) # empty numpy array, a row for each features, and a column for each importance
                    for i, (train,test) in enumerate(cv.split(X_train_sel,y[train_idx],group[train_idx])):
                        X_split = X_train_sel.iloc[train]
                        y_split = y[train_idx][train]
                        fittedestimator = model.fit(X_split, y_split)
                        pi = permutation_importance(estimator=fittedestimator, X=X_split, y=y_split, scoring=scoring, n_repeats=n_repeats).importances
                        pis[:,i*n_repeats:(i+1)*n_repeats] = pi
                    pis = pd.DataFrame(pis, index=selected_features)
                for attr in ["coef_","feature_importances_"]:
                    if hasattr(model, attr):
                        importance = getattr(model,attr)
                        ax.set_xlabel(attr[0:-1])
                if hasattr(model, "estimators_"): # e.g. RandomForestClassifier
                    pi = np.array([tree.feature_importances_ for tree in model.estimators_])
                    perm_sorted_idx = importance.argsort()
                    importances_sorted = pi[:,perm_sorted_idx]
                    ax.set_xlabel(f"tree based feature importance\n{model.n_estimators} trees")
                if isinstance(model,sklearn.linear_model._logistic.LogisticRegression):
                    importance = importance[0]
                pis.T.boxplot(ax=ax, vert=False)
                ax.set_title(type(model).__name__)
            fig.tight_layout()
            if class_startswith:
                plt.suptitle(f"class starts with {class_startswith}")
            ofile = os.path.realpath(f"{suite}_{dist_threshold}_{neurons}neurons_{labelpick}label_{class_startswith}_bs{batch_size}_imp_barplot.png")
            plt.savefig(ofile)
            print(ofile)
            plt.close()

if __name__ == "__main__":
    if False:
        static_file = os.path.join('/glade/scratch/ahijevyc/temp/t.par')
        if os.path.exists(static_file):
            df = pd.read_parquet(static_file)
        else:
            ifiles = glob.glob(f'/glade/work/sobash/NSC_objects/HRRR/grid_data/grid_data_HRRR_d01_2020052400-0000.par')
            df = pd.concat(pd.read_parquet(ifile) for ifile in ifiles)
            df["time"] = pd.to_datetime(df["Date"]) + df["fhr"] * datetime.timedelta(hours=1)
            idate = df.Date.astype('datetime64[ns]')
            df = df.drop(columns="Date")
            
            df["Local_Solar_Hour"] = df["time"].dt.hour + df["lon"]/15
            df = df.rename(columns=dict(xind="projection_y_coordinate",yind="projection_x_coordinate"))
            df = df.set_index(["time","projection_y_coordinate","projection_x_coordinate"])
            glm = xarray.open_dataset("/glade/scratch/ahijevyc/temp/GLM_all.nc")
            assert (glm.time_coverage_start[1] - glm.time_coverage_start[0]) == np.timedelta64(3600,'s'), 'glm.time_coverage_start interval not 1h'
            # Add flashes from previous 2 times and next time to current time. 4-hour centered time window 
            glm = glm + glm.shift(time_coverage_start=2) + glm.shift(time_coverage_start=1) + glm.shift(time_coverage_start=-1)
            print("Merge flashes with df")
            df = df.merge(glm.to_dataframe(), left_on=["time","projection_y_coordinate","projection_x_coordinate"], right_index=True)

            # speed things up without multiindex
            df = df.reset_index(drop=True)
            # Local solar time, sin and cos components
            df = scalar2vector.decompose_circ_feature(df, "Local_Solar_Hour", period=24)

            rptdist = 40
            # Convert severe report distance to boolean (0-rptdist = True)
            for r in ["sighail", "sigwind", "hailone", "wind", "torn"]:
                for h in ["0","1","2"]:
                    rh = f"{r}_rptdist_{h}hr"
                    df[rh] = (df[rh] >= 0) & (df[rh] < rptdist)
            df.to_parquet(static_file, compression=None)
        print(static_file)
        y = df["wind_rptdist_2hr"] 
        X = df.drop(columns=[x for x in df.columns if x.endswith("hr")])
        # normalize data 
        mean = X.mean()
        std  = X.std()
        X = (X - mean) / std
        print('done normalizing')

        model = models[0]
        pi = permutation_importance(model.fit(X, y), X, y, scoring=scoring, n_repeats=1)
        dist_linkage = corr_dendro_plot(X, dist_threshold=[1.5], importances=pi.importances_mean, figh=figh)
    main()
