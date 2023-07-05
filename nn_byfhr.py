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

sns.set_theme() # pretty axes background and gridlines

def nns(ifile):
    # abbreviate input file name for legend 
    ifile = ifile.replace(".scores.txt","")
    return ifile

def parse_args():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "plot NN verification scores written by test_stormrpts_dnn.py, often in /glade/work/ahijevyc/NSC_objects/nn/.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ifiles', nargs="+", type=str, help="input file(s)")
    parser.add_argument("--classes", nargs="+", default=["flashes", "ic", "cg"], help="classes")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
    parser.add_argument("--dpi", type=int, default=120, help="output resolution")
    parser.add_argument("-v","--variable", type=str, default="bss", help="variable to plot")
    parser.add_argument("--twin", nargs="+", default=["1hr","2hr","4hr"], help="time window")
    parser.add_argument("--ymax",type=float, default=0.54, help="maximum on y-axis")
    parser.add_argument("--ymin",type=float, default=0.08, help="minimum on y-axis")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ifiles      = args.ifiles
    debug       = args.debug
    variable    = args.variable

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)


    # Figure dimensions
    fig = plt.figure(figsize=(10,7))

    logging.info(f"Read {len(ifiles)} input files into Pandas DataFrame, with new nn column equal to input filename")
    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile)) for ifile in ifiles], ignore_index=True) # ignore index or get duplicate indices

    # Append fold to fit and drop fold column.
    dfs["fit"] = dfs["fit"] + "." + dfs["fold"]
    dfs = dfs.drop(columns="fold")


    # extract flash count threshold
    t = dfs["nn"].str.extract(r"\b(\d\d)flash.*", expand=True).astype(int)
    dfs["flash threshold"] = t

    # common prefix for nn
    prefix = os.path.commonprefix(dfs["nn"].tolist())
    # remove common prefix from nn (make it shorter) 
    if len(ifiles) > 1: # otherwise nn is empty and doesn't trigger legend
        dfs["nn"] = dfs["nn"].str.replace(prefix,"",regex=False)
        dfs["nn"] = dfs["nn"].str.lstrip("_") # labels with leading underscore not shown in legend

    c = dfs["class"].str.split("_", expand=True)
    c = c.rename(columns={0:"class",1:"rptdist",2:"time window"})
    dfs = dfs.drop(columns="class")
    dfs = pd.concat([dfs,c], axis="columns")

    topax = plt.subplot2grid((3,1),(0,0), rowspan=2)
    botax = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=topax)
    botax.xaxis.set_major_locator(ticker.MultipleLocator(3))

    logging.info("drop all-fhr averages")
    dfs = dfs.loc[(dfs.forecast_hour != "all") & (dfs.fit == "ensmean.all") & 
            (dfs["class"].isin(args.classes))]
    dfs = dfs.loc[dfs["time window"].isin(args.twin)]
    dfs["forecast_hour"] = dfs["forecast_hour"].astype(int)
    dfs["class"] = dfs["class"].str.replace("cg","WeatherBug CG")
    dfs["class"] = dfs["class"].str.replace("ic","WeatherBug IC")
    dfs["class"] = dfs["class"].str.replace("flashes","GLM flash")
    hue_order = ["GLM flash", "WeatherBug IC", "WeatherBug CG"]
    style_order = ["4hr", "2hr", "1hr"]

    sns.lineplot(data=dfs, x="forecast_hour", y=variable,    ax=topax, hue="class", style="time window", size="flash threshold", style_order=style_order, hue_order=hue_order)
    sns.lineplot(data=dfs, x="forecast_hour", y="base_rate", ax=botax, hue="class", style="time window", size="flash threshold", style_order=style_order, hue_order=hue_order, legend=False)
    if args.ymin:
        topax.set_ylim(bottom=args.ymin)
    if args.ymax:
        topax.set_ylim(top=args.ymax)
    botax.set_ylim(bottom=0.005, top=0.09)
    ofile = f"{prefix}{variable}.{'.'.join(args.classes)}.{'.'.join(args.twin)}.png"
    plt.tight_layout()
    fig.savefig(ofile, dpi=args.dpi)
    logging.info(os.path.realpath(ofile))

if __name__ == "__main__":
    main()
