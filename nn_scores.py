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
    parser = argparse.ArgumentParser(description = "plot NN verification scores written by test_stormrpts_dnn.py, often in /glade/work/ahijevyc/NSC_objects/nn/.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ifiles', nargs="+", type=argparse.FileType("r"), help="input file(s)")
    parser.add_argument("--ci", type=int, default=95, help="confidence interval. Show individual lines if ci=0")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
    parser.add_argument("--ensmean", action="store_true", help="ensemble mean")
    parser.add_argument("--mask", type=str, nargs="+", help="only show this(these) mask value(s)")
    parser.add_argument("--nofineprint", action="store_true", help="no fine print (ci, time created, etc)")
    parser.add_argument("--nomem", action="store_true", help="no members")
    parser.add_argument("--noplot", action="store_true", help="no plot, just print all-forecast hour means)")
    parser.add_argument("-v","--variable", type=str, default="bss", help="variable to plot")
    parser.add_argument("--ymax",type=float, help="maximum on y-axis")
    parser.add_argument("--ymin",type=float, help="minimum on y-axis")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ifiles      = args.ifiles
    ci          = args.ci # default ci is 95
    debug       = args.debug
    ensmean     = args.ensmean
    mask        = args.mask
    nofineprint = args.nofineprint
    nomem       = args.nomem
    noplot      = args.noplot
    variable    = args.variable
    ymax        = args.ymax
    ymin        = args.ymin

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)


    # Figure dimensions, line thicknesses, text alignment
    fig = plt.figure(figsize=(15,11))
    text_kw = dict(fontsize=10, ha="left", va="center", clip_on=True) # clip_on in case variable is so low it squishes botax
    # If ci is zero, don't plot confidence band; plot individual lines for all members    
    if ci == 0:
        line_kw = dict(units="mem", estimator=None)
    else:
        # If ci is not zero plot ci% confidence interval
        line_kw = dict(errorbar=('ci',95))

    line_kw.update(dict(hue="nn", style="nn"))

    logging.info(f"Read {len(ifiles)} input files into Pandas DataFrame, with new nn column equal to input filename")
    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile.name)) for ifile in ifiles], ignore_index=True) # ignore index or get duplicate indices

    # If DataFrame has a "mask" column, use it to signify linewidth.
    if "mask" in dfs:
        line_kw.update(dict(size="mask"))
        if mask is not None:
            dfs = dfs[dfs["mask"].isin(mask)]

    # Append fold to mem and drop fold
    dfs["mem"] = dfs["mem"] + "." + dfs["fold"]
    dfs = dfs.drop(columns="fold")

    # common prefix for nn
    prefix = os.path.commonprefix(dfs["nn"].tolist())
    # remove common prefix from nn (make it shorter) 
    if len(ifiles) > 1: # otherwise nn is empty and doesn't trigger legend
        dfs["nn"] = dfs["nn"].str.replace(prefix,"",regex=False)

    # Loop thru types of event (torn, wind, hail, lightning)
    for cl, df in dfs.groupby("class"):
        print(f"\n{cl}    prefix={prefix}")
        # Separate fhr=all
        df_all = df[df.fhr=="all"]
        df = df.copy()[df.fhr != "all"]
        df["fhr"] = df["fhr"].astype(int)
        if not noplot:
            topax = plt.subplot2grid((3,1),(0,0), rowspan=2)
            botax = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=topax)
            # Empty fineprint_string placeholder for fine print in lower left corner of image.
            fineprint = plt.annotate(text="", xy=(0,-55), xycoords=('axes fraction','axes pixels'), va="top", fontsize=6, wrap=True)
            if not nofineprint:
                fineprint.set_text(f"{ci}% confidence interval\ncreated {datetime.datetime.now()}")

            iens = df["mem"] == "ensmean.all"
            if not nomem:
                sns.lineplot(data=df[~iens], x="fhr", y=variable,  ax=topax, **line_kw)
            if ensmean:
                logging.info("ensemble mean")
                sns.lineplot(data=df[iens], x="fhr", y=variable,  ax=topax, **line_kw, legend=nomem)
                for i,row in df[iens & (df.fhr == df.fhr.max())].iterrows():
                    topax.text(df.fhr.max(), row[variable], "ens. mean", **text_kw)
                topax.set_xlim(topax.get_xlim()[0], topax.get_xlim()[1]+1.2) # add space for "ensmean" label

            # Base rate
            base_rate_ax = botax
            sns.lineplot(data=df, x="fhr", y="base rate", ax=base_rate_ax, legend=False, **line_kw) # ignores color arg
            base_rate_ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

            if variable == "bss":
                ylim = (-0.03,0.35)
                if cl == "flashes": ylim = (0,0.75)
                if cl.startswith("torn") or cl.startswith("sig"): ylim = (-0.03, 0.12)
                if cl.startswith("hailone"): ylim = (-0.02, 0.18)
                topax.set_ylim(ylim)

            if ymax is not None:
                topax.set_ylim(top=ymax)
            if ymin is not None:
                topax.set_ylim(bottom=ymin)

            handles, labels = topax.get_legend_handles_labels()
            if len(handles) > 8:
                topax.legend(handles, labels, ncol=2, fontsize=7, labelspacing=0.45, columnspacing=1, title=prefix,
                        handlelength=3, title_fontsize=8) #default handlelength=2. to see entire cycle of long patterns
            ofile = f"{os.path.join(os.path.dirname(prefix),cl+os.path.basename(prefix))}.png"
            plt.tight_layout()
            fig.savefig(ofile)
            logging.info(os.path.realpath(ofile))
            plt.clf()

        # Look at the aggregrate scores for "all" forecast hours. Not what was plotted.
        df_all = df_all[df_all.mem == "ensmean.all"].set_index("nn")
        df_all = df_all.sort_values(variable,ascending=False)
        if "mask" in dfs:
            print(df_all[["mask","bss","base rate","auc","aps"]])
        else:
            print(df_all[["bss","base rate","auc","aps"]])

if __name__ == "__main__":
    main()
