import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn import metrics
#from scikitplot.metrics import plot_roc
from tqdm import tqdm

# Use scikitplot.metrics.plot_roc - nice because has adds other blended ROC curves in it.


def plot_roc(y_true, y_probas, **kwargs):
    # Reorder y_probas columns to match order of categories in y_true
    # plot_roc() expects alphabetical, but labels are probably ordered categories in a different order
    new_column_order = np.argsort(y_true.cat.categories)
    y_probas_new = y_probas[:, new_column_order]
    return plot_roc(y_true, y_probas_new, **kwargs)


def bss(obs, fcst):
    bs = np.mean((fcst - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - bs / climo


def reliability_diagram(ax, obs, fcst, thresh, n_bins=10, plabel=True, **kwargs):
    no_lines_yet = len(ax.get_lines()) == 0
    if no_lines_yet:
        ax.plot([0, 1], [0, 1], "k", alpha=0.5)

    for o_thresh in thresh:
        # calibration curve
        true_prob, fcst_prob = calibration_curve(obs >= o_thresh, fcst, n_bins=n_bins)
        bss_val = bss(obs >= o_thresh, fcst)
        print(f"{bss_val:.3f}")
        base_rate = (obs >= o_thresh).mean()  # base rate
        (s,) = ax.plot(
            fcst_prob,
            true_prob,
            "s-",
            label=f"{fcst.name} bss:{bss_val:1.3f}",
            **kwargs,
        )
        if plabel:
            for x, f in zip(fcst_prob, true_prob):
                if np.isnan(f):
                    continue  # avoid TypeError: ufunc 'isnan' not supported...
                # label reliability points
                ax.annotate(
                    "%1.3f" % f,
                    xy=(x, f),
                    xytext=(0, 1),
                    textcoords="offset points",
                    va="bottom",
                    ha="center",
                    fontsize="xx-small",
                )

        noskill_line = ax.plot(
            [0, 1],
            [base_rate / 2, (1 + base_rate) / 2],
            linewidth=0.3,
            alpha=0.7,
            label="",
            color=s.get_color(),
        )
        baserateline = ax.axhline(
            y=base_rate,
            label=f"base rate {base_rate:.3f}",
            linewidth=0.5,
            linestyle="dashed",
            dashes=(9, 9),
            color=s.get_color(),
        )
        baserateline_vertical = ax.axvline(
            x=base_rate,
            linewidth=0.5,
            linestyle="dashed",
            dashes=(9, 9),
            color=s.get_color(),
        )

    ax.set_xlabel(f"forecast prob. of {o_thresh}+ {obs.name}")
    ax.set_ylabel(f"obs. fraction of {o_thresh}+ {obs.name}")
    ax.set_title("(a) Reliability", loc="left")
    ax.legend(loc="upper left", fontsize="xx-small")
    ax.set_xlim((0, 1))

    return ax


def ROC_curve(ax, obs, fcst, label="", sep=0.1, plabel=True, fill=False):
    """
    Generate a ROC curve from the contingency table by calculating the probability of detection (TP/(TP+FN)) and the
    probability of false detection (FP/(FP+TN)).
    """

    no_lines_yet = len(ax.get_lines()) == 0
    if no_lines_yet:
        ax.plot([0, 1], [0, 1], "k", alpha=0.5)

    auc = None
    if obs.all() or (obs == False).all():
        logging.info("obs are all True or all False. ROC AUC score not defined")
        r = ax.plot([0], [0], marker="+", linestyle="solid", label=label)
    elif obs is None or fcst is None:
        # placeholders
        r = ax.plot([0], [0], marker="+", linestyle="solid", label=label)
    else:
        # ROC auc with threshold labels separated by sep
        auc = metrics.roc_auc_score(obs, fcst)
        logging.debug(f"auc {auc}")
        pofd, pody, thresholds = metrics.roc_curve(obs, fcst)
        r = ax.plot(
            pofd,
            pody,
            marker="+",
            markersize=1 / np.log10(len(pofd)),
            linestyle="solid",
            label=f"{fcst.name} auc:{auc:1.3f}",
        )
        if fill:
            auc = ax.fill_between(pofd, pody, alpha=0.2)
        if plabel:
            old_x, old_y = 0.0, 0.0
            for x, y, s in zip(pofd, pody, thresholds):
                if ((x - old_x) ** 2 + (y - old_y) ** 2.0) ** 0.5 > sep:
                    # label thresholds on ROC curve
                    ax.annotate(
                        "%1.3f" % s,
                        xy=(x, y),
                        xytext=(0, 1),
                        textcoords="offset points",
                        va="baseline",
                        ha="left",
                        fontsize="xx-small",
                    )
                    old_x, old_y = x, y
                else:
                    logging.debug(
                        f"statisticplot.ROC_curve: toss {x},{y},{s} annotation. Too close to last label."
                    )
    ax.set_title("(d) Receiver operating characteristic", loc="left")

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("probability of false detection")
    ax.set_ylabel("probability of detection")
    ax.legend(loc="lower right", fontsize="x-small")
    return r, auc


map_crs = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25, 25))


def make_map(
    figsize=(9,6),
    bbox=[-121, -72, 22, 50],
    projection=map_crs,
    gridlines:bool=False,
    draw_labels:bool=True,
    scale=1
) -> tuple[plt.Figure, plt.Axes]:
    """cartopy map of CONUS with coastlines, states, etc."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    ax = ax_features(ax, scale=scale)
    if gridlines:
        gl = ax.gridlines(
            draw_labels=draw_labels, x_inline=False
        )
        gl.top_labels = False
    return fig, ax

def ax_features(ax, scale=1):
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25 * scale)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.25 * scale)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.05 * scale)
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        edgecolor="k",
        linewidth=0.25 * scale,
        facecolor="k",
        alpha=0.05,
    )
    return ax



def far(obs, fcst):
    """
    false alarm ratio
    fp / (tp+fp)
    """
    tp = (obs & fcst).sum()
    fp = (~obs & fcst).sum()
    return fp / (tp + fp) if tp + fp else np.nan


def pod(obs, fcst):
    """
    probability of detection
    tp / (tp+fn)
    """
    tp = (obs & fcst).sum()
    fn = (obs & ~fcst).sum()
    return tp / (tp + fn) if tp + fn else np.nan


def count_histogram(ax, fcst, n_bins=10, count_label=True):
    """
    histogram of forecast probability
    """
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("(c) Histogram", loc="left")
    if fcst is None:
        return None
    # Histogram of counts
    counts, bins, patches = ax.hist(
        fcst, bins=n_bins, histtype="step", lw=2, label=fcst.name
    )
    logging.debug(f"counts={counts}")
    logging.debug(bins)
    logging.debug(patches)
    if count_label:
        for count, x in zip(counts, bins):
            # label counts
            ax.annotate(
                str(int(count)),
                xy=(x, count),
                xytext=(0, -1),
                textcoords="offset points",
                va="top",
                ha="left",
                fontsize="xx-small",
            )
    ax.set_xlim((0, 1))
    return ax


def performance_diagram(ax: plt.axes, obs, fcst, thresh, pthresh):
    """
    performace diagram
    xaxis = 1-far
    yaxis = prob of detection
    where far = fp / (tp+fp)

    Parameters
    ----------

    """
    bias_lines = [0.2, 0.5, 0.8, 1, 1.3, 2, 5]
    csi_lines = np.arange(0.1, 1.0, 0.1)
    alpha = 0.8
    lw = 1
    color="0.8"
    bias_pts = [
        [sr * b for sr in [0, 1.0]] for b in bias_lines
    ]  # compute pod values for each bias line
    csi_pts = np.array(
        [
            [csi / (csi - (csi / sr) + 1) for sr in np.arange(0.011, 1.01, 0.005)]
            for csi in csi_lines
        ]
    )  # compute pod values for each csi line
    csi_pts = np.ma.masked_array(csi_pts, mask=(csi_pts < 0.05))

    # add bias and CSI lines to performance diagram
    for r in bias_pts:
        ax.plot([0, 1], r, color=color, linestyle="dashed", lw=lw, alpha=alpha)
    for r in csi_pts:
        ax.plot(
            np.arange(0.01, 1.01, 0.005),
            r,
            color=color,
            alpha=alpha,
            linestyle="solid",
            linewidth=lw,
        )
    for x in [b for b in bias_lines if b <= 1]:
        ax.text(1.002, x, x, va="center", ha="left", fontsize="x-small", color="0.5")
    for x in [b for b in bias_lines if b > 1]:
        ax.text(1 / x, 1, x, va="bottom", ha="center", fontsize="xx-small", color="0.5")

    # axes limits, labels
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_title("(b) Performance", loc="left")
    ax.set_xlabel("1 - false alarm ratio")
    ax.set_ylabel("probability of detection")
    for o in thresh:
        ostr = ""
        if len(thresh) > 1:
            # Add observation threshold if there are more than one.
            ostr = f" {o}+"
        x = [1 - far(obs >= o, fcst >= p) for p in pthresh]
        y = [pod(obs >= o, fcst >= p) for p in pthresh]
        ax.plot(x, y, marker='o', label=f"{fcst.name}{ostr}")
        for x, y, p in zip(x, y, pthresh):
            if ~np.isnan(x) and ~np.isnan(y):
                ax.text(x, y, p, fontsize="x-small")

    ax.legend(fontsize="x-small", loc="upper right")
    return ax


def stat_plots(
        obs: pd.Series,
        fcst: pd.Series,
        thresh: pd.Series = pd.Series(np.arange(1, 10), name=f"obs threshold"),
        pthresh: pd.Series = pd.Series(np.arange(0, 1.1, 0.2), name=f"prob threshold"),
        o_thresh_roc: float=10,
        sep: float=0.01,
        suptitle: str =None,
        fig: plt.figure =None,
        **kwargs,
        ):
    """
    Parameters
    ----------
    obs : pd.Series
        truth
    fcst : pd.Series
        forecast 
        forecast.name is used as label
    """

    if fig is None:
        ncols, nrows = 2, 2
        fig, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(ncols * 4, nrows * 4)
        )
        axes = iter(axes.flatten())
    else:
        logging.info(f"use old figure {fig} with {fig.get_axes()}")
        axes = iter(fig.get_axes())

    reliability_diagram(next(axes), obs, fcst, thresh, plabel=False, **kwargs)

    performance_diagram(next(axes), obs, fcst, thresh, pthresh=pthresh)

    """
    logging.info("pod")
    ax = next(axes)
    df=pd.DataFrame([[pod(obs>=o,fcst>=p) for p in pthresh] for o in tqdm(thresh)],index=thresh, columns=pthresh )
    df.plot(ax=ax, title="probability of detection")
    ax.set_xscale("log")
    ax.set_ylim((0,1))
    """

    logging.info("count histogram")
    count_histogram(next(axes), fcst, count_label=False, **kwargs)

    logging.info("ROC curve")
    ROC_curve(next(axes), obs >= o_thresh_roc, fcst, sep=sep, plabel=False)

    """
    logging.info("brier skill score")
    ax = next(axes)
    df = pd.Series([bss(obs > o, fcst) for o in tqdm(thresh)], index=thresh)
    df.plot(ax=ax, title="brier skill score", label=label)
    ax.set_xscale("log")
    ax.set_ylim((-0.3, 0.6))
    """

    # commented in deference to seaborn set_style
    # [a.grid(visible=True, lw=0.5, linestyle="dotted") for a in fig.get_axes()]
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig
