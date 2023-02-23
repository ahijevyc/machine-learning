import datetime
import glob
import visualizecv
import logging
import matplotlib.pyplot as plt
from ml_functions import get_argparser, get_features, get_savedmodel_path, load_df, rptdist2bool
import numpy as np
import os
import pandas as pd
import pdb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import sys
from tensorflow.keras.utils import to_categorical 
import xarray
#from xgboost import XGBRegressor

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = get_argparser()
args = parser.parse_args()
logging.info(args)


batchsize = args.batchsize # Default is min(200, n_samples) if batch_size is around 25 or larger, ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
neurons = args.neurons 
seed = args.seed
#models = [LinearRegression(), LogisticRegression(), DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor(), KNeighborsRegressor(), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
#models = [RandomForestRegressor(n_estimators=150), MLPClassifier(hidden_layer_sizes=(neurons,neurons))]
models = [] # if you just want matrix and dendrogram
models = [MLPClassifier(hidden_layer_sizes=neurons,batch_size=batchsize)]
scoring = 'balanced_accuracy'

dist_thresholds = [2.5, 2, 1.5, 1, 0.5, .25, 0]
dist_thresholds = [1]

np.random.seed(seed)

def corr_dendro_plot(X, suite=None, dist_thresholds=dist_thresholds, importances=None, figh=14):
    features = X.columns
    logging.info(f"corr_dendro_plot: {len(features)} features suite={suite} dist_thresholds={dist_thresholds} importances={importances} figh={figh}")
    corr = spearmanr(X).correlation # just training data

    logging.info("Ensure the correlation matrix is symmetric")
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    for d in dist_thresholds:
        logging.info(f"distance thresh={d}")
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
            logging.info(f"{len(features)} to {len(selected_features)} features {selected_features.to_list()}")
            fontsize=yticklabels[0].get_fontsize() + d*8 # starting with current ticklabel fontsize get bigger to the right.
            for c, fs in pm.groupby("cluster_id"):
                # label one member of each cluster
                # Find the indices corresponding to this cluster and take the average tickvalue
                ii = [dendro["ivl"].index(f) for f in fs.index]
                y0 = np.mean(ax2.get_yticks()[ii])
                ax2.text(d, y0, selected_features[c], va="center", ha="center", fontsize=fontsize) 


        fig.tight_layout()
        ofile = os.path.join(os.getenv("TMPDIR"), f"{suite}_{d:4.2f}_dendro.png")
        ofile = os.path.realpath(ofile)
        plt.savefig(ofile)
        logging.info(f"created {ofile}")
        plt.close()
    return dist_linkage



def main():
    clobber = args.clobber
    fhr = args.fhr
    figh=14
    kfold = args.kfold
    rptdist = args.rptdist
    savedmodel = get_savedmodel_path(args)
    suite = args.suite

    df = load_df(args)
    df, label_cols = rptdist2bool(df, args)

    # Make initialization_time a MultiIndex level
    df = df.set_index("initialization_time", append=True)

    validtimes = df.index.get_level_values(level="valid_time")
    logging.info(f"range of valid times: {validtimes.min()} - {validtimes.max()}")

    # Used to test all columns for NA, but we only care about the feature subset being complete.
    # For example, mode probs are not available for fhr=2 but we don't need to drop fhr=2 if
    # the other features are complete.
    feature_list = get_features(args)
    logging.info(
        f"Retain rows where all {len(feature_list)} requested features are present")
    beforedropna = len(df)
    df = df.dropna(axis="index", subset=feature_list)
    logging.info(f"kept {len(df)}/{beforedropna} cases with no NA features")

    before_filtering = len(df)
    logging.info(f"Retain rows with requested forecast hours {fhr}")
    df = df.loc[df["forecast_hour"].isin(fhr)]
    logging.info(
        f"kept {len(df)}/{before_filtering} rows with requested forecast hours")

    logging.info(f"Split {len(label_cols)} labels away from predictors")
    labels = df[label_cols]  # labels converted to Boolean above

    print(labels.sum())

    columns_before_filtering = df.columns
    df = df[feature_list]
    logging.info(
        f"dropped {set(columns_before_filtering) - set(df.columns)}")
    logging.info(
        f"kept {len(df.columns)}/{len(columns_before_filtering)} features")

    # Without providing a list of importances, this won't label the best member from each cluster.
    dist_linkage = corr_dendro_plot(df, suite=suite, dist_thresholds=dist_thresholds, figh=figh)


    logging.info("normalize predictors")
    mean = df.mean()
    std  = df.std()
    df = (df - mean) / std
    logging.info('done normalizing')

    X_train, X_val, y_train, y_val = train_test_split(df, labels, random_state=seed)

    #label_cols = ["flashes_40km_2hr"]
    for labelpick in label_cols:
        n_repeats = 25
        explabel = f"neurons={neurons} {labelpick}\nbatch size={batchsize} scoring={scoring}"
        explabel += f"\npermutation importance\n(mean drop in {scoring} from permuting feature)\nrepeated {n_repeats}x"

        assert models

        # HACK for just 1st model and all folds. get importances so we can choose the predictor with highest importance in each cluster 
        model = models[0]
        ofile = os.path.join(os.getenv("TMPDIR"), f"{savedmodel}_{n_repeats}repeats.{labelpick}.csv")
        ofile = os.path.realpath(ofile)
        if os.path.exists(ofile):
            logging.info(f"read {ofile}")
            imp_df = pd.read_csv(ofile, index_col=0)
        else:
            logging.info(f"train to predict {labelpick} with {len(X_train)} cases")
            fittedestimator = model.fit(X_train, y_train[labelpick])
            logging.info(f"score {len(X_val)} cases")
            score = sklearn.metrics.accuracy_score(y_val[labelpick], fittedestimator.predict(X_val))
            logging.info(f"accuracy {score}")
            score = sklearn.metrics.balanced_accuracy_score(y_val[labelpick], fittedestimator.predict(X_val))
            logging.info(f"{labelpick} balanced accuracy {score}")
            logging.info(f"calculate permutation importance on {len(X_val)} cases. n_repeats={n_repeats}")
            history = permutation_importance(fittedestimator, X_val, y_val[labelpick], scoring=scoring, n_repeats=n_repeats, n_jobs=-1)
            imp_df = pd.DataFrame(history.importances, index=fittedestimator.feature_names_in_)
            imp_df.to_csv(ofile)
            logging.info(f"created {ofile}")
        ax = imp_df.T.boxplot(figsize=(18, 5),rot=90)
        ax.set_ylabel(explabel)
        ax.set_title(type(model).__name__)
        plt.tight_layout()
        ofile = os.path.join(os.getenv("TMPDIR"), f"{savedmodel}_{labelpick}_bar.png")
        ofile = os.path.realpath(ofile)
        plt.savefig(ofile)
        logging.info(f"created {ofile}")
        plt.close()

        continue

        imp_df["importance"] = imp_df.mean(axis="columns")

        #Redo dendrograms with labels of best importance
        dist_linkage = corr_dendro_plot(df, suite=suite, dist_thresholds=dist_thresholds, figh=figh, importances=imp_df["importance"])

        for dist_threshold in dist_thresholds:
            cluster_ids = hierarchy.fcluster(dist_linkage, dist_threshold, criterion="distance")
            imp_df["cluster_id"] = cluster_ids
            selected_features = imp_df.groupby("cluster_id")["importance"].idxmax()
            logging.info(f"{len(feature_list)} to {len(selected_features)} features {selected_features.to_list()}")

            fig, axes = plt.subplots(ncols=len(models), figsize=(len(models)*5, figh))
            if len(models) == 1: axes = np.array(axes) # avoid AttributeError: 'AxesSubplot' object has no attribute 'flatten'
            for ax, model in zip(axes.flatten(),models):
                logging.info(f"dist_threshold={dist_threshold} {type(model).__name__} {scores} mean: {np.array(scores).mean()} std: {np.array(scores).std()}")
                if type(model).__name__ in ["KNeighborsRegressor","MLPClassifier"]:
                    fittedestimator = model.fit(X_train[selected_features], y_train[labelpick])
                    history = permutation_importance(fittedestimator, X_val[selected_features], y_val[labelpick], 
                            scoring=scoring, n_repeats=n_repeats, n_jobs=-1)
                    pis = pd.DataFrame(history.importances, index=selected_features)
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
            ofile = os.path.join(os.getenv("TMPDIR"), f"{savedmodel}_{dist_threshold}_{labelpick}_imp_barplot.png")
            ofile = os.path.realpath(ofile)
            plt.savefig(ofile)
            print(ofile)
            plt.close()

if __name__ == "__main__":
    if False:
        df = load_df(args)
        y = df["wind_40km_2hr"] 
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
