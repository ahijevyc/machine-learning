import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pdb
import seaborn as sns
import sys

#sns.set_theme()  # pretty axes background and gridlines
sns.set_style("whitegrid")

def nns(ifile:str) -> str:
    # abbreviate input file name for legend
    ifile = ifile.replace(".scores.txt", "")
    return ifile

def fhr(s:str):
    """ given a numeric str return it
    given a range, return mean
    """
    if s == "all":
        return s
    try:
        s=int(s)
        return s
    except Exception as e:
        logging.info(e)
        start, end = s.lstrip("[").rstrip(")").split(",")
        return np.mean([float(start), float(end)])


def parse_args():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description="plot NN verification scores written by test_stormrpts_dnn.py, often in /glade/work/ahijevyc/NSC_objects/nn/.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ifiles', nargs="+", type=str, help="input file(s)")
    parser.add_argument("--classes", nargs="+", default=["cg.ic"],
                        choices=["flashes", "ic", "cg", "cg.ic", "any"], help="classes")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="turn on debug mode")
    parser.add_argument("--dpi", type=int, default=120,
                        help="output resolution")
    parser.add_argument("-v", "--variable", type=str,
                        default="bss", help="variable to plot")
    parser.add_argument("--lonbin", nargs="+", help="lon bin")
    parser.add_argument("--twin", nargs="+",
                        default=["1hr", "2hr", "4hr"], choices=["1hr", "2hr", "4hr"], help="time window")
    parser.add_argument("--rptdist", nargs="+",
                        default=["20km"], choices=["20km", "40km"], help="distance to report or event")
    parser.add_argument("--thresh", type=int, nargs="+",
                        default=[1], help="flash threshold")
    parser.add_argument("--ymax", type=float, default=0.64,
                        help="maximum on y-axis")
    parser.add_argument("--ymin", type=float, default=0.04,
                        help="minimum on y-axis")
    args = parser.parse_args()
    return args


def double(s):
    assert s.isin(["0hr", "1hr", "2hr"]).all()
    # Kludge for severe storm reports (twin is 0,1,2 instead of 1, 2, 4)
    s[s == "2hr"] = "4hr"
    s[s == "1hr"] = "2hr"
    s[s == "0hr"] = "1hr"
    return s

def main():
    args = parse_args()
    ifiles = args.ifiles
    debug = args.debug
    variable = args.variable

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)

    # Figure dimensions
    fig = plt.figure(figsize=(8, 6))

    logging.info(
        f"{len(ifiles)} input files into Pandas DataFrame, with new nn column equal to input filename")
    dfs = pd.concat([pd.read_csv(ifile, header=0).assign(nn=nns(ifile))
                    for ifile in ifiles], ignore_index=True)  # ignore index or get duplicate indices

    logging.info(f"read {len(dfs)} lines")

    # Append fold to fit and drop fold column.
    dfs["fit"] = dfs["fit"] + "." + dfs["fold"]
    dfs = dfs.drop(columns="fold")

    # extract flash count threshold (all digits between word boundary and plus sign)
    t = dfs["nn"].str.extract(r"\b(\d+)\+.*", expand=True).astype(int)
    dfs["flash threshold"] = t

    # common prefix for nn
    prefix = os.path.commonprefix(dfs["nn"].tolist())
    # remove common prefix from nn (make it shorter)
    if len(ifiles) > 1:  # otherwise nn is empty and doesn't trigger legend
        dfs["nn"] = dfs["nn"].str.replace(prefix, "", regex=False)
        # labels with leading underscore not shown in legend
        dfs["nn"] = dfs["nn"].str.lstrip("_")

    c = dfs["class"].str.split("_", expand=True)
    c = c.rename(columns={0: "class", 1: "rptdist", 2: "time window"})
    dfs = dfs.drop(columns="class")
    dfs = pd.concat([dfs, c], axis="columns")

    # Kludge for severe storm reports (twin is 0,1,2 instead of 1, 2, 4)
    iseverestorm = dfs["class"].isin(
        ["sighail", "sigwind", "hailone", "wind", "torn", "any"])
    dfs.loc[iseverestorm, "time window"] = double(
        dfs.loc[iseverestorm, "time window"])
    dfs = dfs.rename(
            columns={
                "forecast_hour": "forecast hour",
                "base_rate": "base rate",
                "bss": "Brier Skill Score",
                }
            )
    if variable == "bss":
        variable = "Brier Skill Score"

    topax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    topax.xaxis.set_major_locator(ticker.MultipleLocator(6))
    # Secondary x axis on the top with UTC instead of forecast hour
    topax2 = topax.secondary_xaxis("top")
    topax2.xaxis.set_major_locator(ticker.MultipleLocator(12))
    topax2.xaxis.set_major_formatter(lambda x, pos:f"{x%24:.0f}UTC")
    botax = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=topax)

    
    logging.warning("tossing forecast hour cuts (time ranges) TODO: handle these")
    dfs = dfs[~dfs["forecast hour"].str.startswith("[")]

    logging.debug("keep numeric forecast hours, drop individual fits")
    dfs["forecast hour"] = dfs["forecast hour"].apply(fhr)
    dfs = dfs.loc[
        dfs["forecast hour"].ne("all") &
        (dfs.fit == "ensmean.all") &
        (dfs["class"].isin(args.classes))
    ]
    
    logging.info(
        f"kept {len(dfs)} numeric forecast hour, ensmean.all, class={args.classes} lines")
    dfs["forecast hour"] = dfs["forecast hour"].astype(int)
    dfs = dfs.loc[dfs["time window"].isin(args.twin)]
    logging.info(f"kept {len(dfs)} twin={args.twin} lines")
    dfs = dfs.loc[dfs["flash threshold"].isin(args.thresh)]
    logging.info(f"kept {len(dfs)} thresh={args.thresh} lines")
    if args.lonbin is not None:
        dfs = dfs.loc[dfs["lon_bin"].isin(args.lonbin)]
        logging.info(f"kept {len(dfs)} lonbin={args.lonbin} lines")
    dfs = dfs.loc[dfs["rptdist"].isin(args.rptdist)]
    logging.info(f"kept {len(dfs)} rptdist={args.rptdist} lines")
    dfs["class"] = dfs["class"].str.replace("any", "any severe")
    dfs["class"] = dfs["class"].str.replace(
        "cg.ic", "ENTLN CG+IC", regex=False)
    dfs["class"] = dfs["class"].str.replace("cg", "ENTLN CG")
    dfs["class"] = dfs["class"].str.replace("ic", "ENTLN IC")
    dfs["class"] = dfs["class"].str.replace("flashes", "GLM flash")
    # Only put in hue_order elements that are in "class" column
    hue = "class"
    hue_order = ordered_intersection(
        ["any severe", "ENTLN CG", "ENTLN IC", "ENTLN CG+IC", "GLM flash"], dfs[hue])
    # hue = "rptdist"
    # hue_order = ordered_intersection(["40km", "20km"], dfs[hue])
    #hue = "lon_bin"
    #hue_order = None

    
    size = "time window"
    size_order = ordered_intersection(["4hr", "2hr", "1hr"], dfs[size])

    style = "flash threshold"
    style_order = ordered_intersection([1, 50], dfs[style])
    style=None
    style_order=None

    if len(size_order) == 1:
        sizes = [4]
    else:
        sizes = [4, 2]


    sns.lineplot(data=dfs, x="forecast hour", y=variable, ax=topax, marker="o",
                 hue=hue, style=style, sizes=sizes, size_order=size_order, size=size,
                 markersize=1.9, markerfacecolor="white",
                 style_order=style_order, hue_order=hue_order)
    topax.legend(loc="upper right", bbox_to_anchor=(0.6, 0.995))
    sns.lineplot(data=dfs, x="forecast hour", y="base rate", ax=botax, marker="o",
                 hue=hue, style=style, sizes=sizes, size_order=size_order, size=size,
                 markersize=1.9, markerfacecolor="white",
                 style_order=style_order, hue_order=hue_order, legend=False)


    if args.ymin is not None:
        topax.set_ylim(bottom=args.ymin)
    if args.ymax is not None:
        topax.set_ylim(top=args.ymax)
    topax.set_xlim((0,48))

    # don't show bottom panel if top panel already has base rate
    if variable == "base rate":
        topax.set_xlim((1,25))
        botax.set_visible(False)

    for x in range(0,48,24):
        topax.axvspan(x, x+12, alpha=0.05, facecolor="black")
        botax.axvspan(x, x+12, alpha=0.05, facecolor="black")
    #botax.set_ylim(bottom=0, top=0.09)
    ofile = f"{prefix}{variable}.{'.'.join([str(x) for x in args.thresh])}.{'.'.join(args.classes)}.{'.'.join(args.twin)}.png"
    plt.tight_layout()
    fig.savefig(ofile, dpi=args.dpi)
    print(os.path.realpath(ofile))


def ordered_intersection(arr, col):
    r = [e for e in arr if e in col.values]
    return r


if __name__ == "__main__":
    main()
