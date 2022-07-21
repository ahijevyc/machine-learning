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
    ifile = os.path.realpath(ifile) # absolute path
    ifile = ifile[ifile.index("/nn/")+4:] # everything after nn/ directory
    ifile = ifile.replace(".scores.txt","")
    return ifile

def parse_args():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "plot NN verification scores written by test_stormrpts_dnn.py, often in /glade/work/ahijevyc/NSC_objects/nn/.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ifiles', nargs="+", type=argparse.FileType("r"), help="input file(s)")
    parser.add_argument("--ci", type=int, default=95, help="confidence interval. Show individual lines if ci=0")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
    parser.add_argument("--auc", action="store_true", help="area under ROC curve")
    parser.add_argument("--ensmean", action="store_true", help="ensemble mean")
    parser.add_argument("--nofineprint", action="store_true", help="no fine print (ci, time created, etc)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ifiles      = args.ifiles
    ci          = args.ci # default ci is 95
    debug       = args.debug
    plotauc     = args.auc
    ensmean     = args.ensmean
    nofineprint = args.nofineprint

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)


    # Figure dimensions, line thicknesses, text alignment
    fig = plt.figure(figsize=(11,8))
    lw = 2  # line width for mean of members
    text_kw = dict(fontsize=10, ha="left", va="center")
    # If ci is zero, don't plot confidence band; plot individual lines for all members    
    if ci == 0:
        line_kw = dict(units="mem", estimator=None)
    else:
        # If ci is not zero plot ci% confidence interval
        line_kw = dict(ci=ci)



    # Read input files into Pandas DataFrame, with nn column = filename
    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile.name)) for ifile in ifiles])


    # Loop thru types of event (torn, wind, hail, lightning)
    for cl, df in dfs.groupby("class"):
        logging.info(f"class={cl}")
        topax = plt.subplot2grid((3,1),(0,0), rowspan=2)
        botax = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=topax)
        # Empty fineprint_string placeholder for fine print in lower left corner of image.
        fineprint = plt.annotate(text="", xy=(0,-55), xycoords=('axes fraction','axes pixels'), va="top", fontsize=6, wrap=True)
        if not nofineprint:
            fineprint.set_text(f"{ci}% confidence interval\ncreated {datetime.datetime.now()}")
        df = df.reset_index(drop=True) # multiple csv files makes duplicates

        iens = df["mem"] == "ensmean"
        imem = df["mem"] != "ensmean"
        sns.lineplot(data=df[imem], x="fhr", y="bss",  ax=topax, style="nn", linewidth=lw, **line_kw)
        if ensmean:
            sns.lineplot(data=df[iens], x="fhr", y="bss",  ax=topax, style="nn", color="black", lw=lw*0.25, legend=False)
            topax.text(df.fhr.max(), df[iens]["bss"].iloc[-1], "ens. mean", **text_kw)
            topax.set_xlim(topax.get_xlim()[0], topax.get_xlim()[1]+2) # add space for "ensmean" label
        topax.text(df.fhr.max(), df[imem].groupby("fhr")["bss"].mean().iloc[-1], " bss", **text_kw) # looks better preceded by space

        if len(ifiles) == 2: # Plot difference between 2 input files on bottom axes
            ddf = df.set_index(["class","mem","fhr","nn"])
            nn0, nn1 = ddf.index.unique(level="nn")
            logging.info(f"difference plot ( {nn0} - {nn1} )")
            diff = ddf.xs(nn0, level="nn") - ddf.xs(nn1, level="nn")
            imem_diff = diff.index.get_level_values(level="mem") != "ensmean" # can't reuse imem and iens variables; they are used later
            iens_diff = diff.index.get_level_values(level="mem") == "ensmean"
            sns.lineplot(data=diff[imem_diff], x="fhr", y="bss",  ax=botax, linewidth=lw, **line_kw)
            botax.set_ylabel(f"bss difference")
            if plotauc:
                # AUC difference on secondary axis (right side)
                baxr=botax.twinx()
                sns.lineplot(data=diff[imem_diff], x="fhr", y="auc",  ax=baxr, linewidth=lw, color="gold", **line_kw)
                baxr.set_ylabel(f"auc difference")
                baxr.set_ylim(np.array(botax.get_ylim())/10) # make auc y-limits 1/10th bss y-limits
                baxr.grid(False) #needs to be after lineplot
            botax.set_title(f"{nn0} - {nn1}", fontsize="small")
            if ensmean:
                sns.lineplot(data=diff[iens_diff], x="fhr", y="bss",  ax=botax, color="black", linewidth=lw*0.25, legend=False)
                botax.text(df.fhr.max(), diff[iens_diff]["bss"].iloc[-1], " ens. mean bss diff", fontsize=7, ha="left", va="center")


        # Base rate
        sns.lineplot(data=df, x="fhr", y="base rate", ax=topax, style="nn", linewidth=lw, color="red", legend=False)
        topax.set_ylabel(topax.get_ylabel()+" and base rate")
        topax.text(df.fhr.max(), df["base rate"].iloc[-1], "base rate", **text_kw)
        topax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ylim = (0,0.225)
        if cl == "flashes": ylim = (0,0.6)
        if cl.startswith("torn") or cl.startswith("sighail"): ylim = (-0.03, 0.1)
        topax.set_ylim(ylim)

        if plotauc:
            # AUC on secondary axis (right side)
            axr=topax.twinx()
            sns.lineplot(data=df[imem], x="fhr", y="auc", ax=axr, style="nn", linewidth=lw, color="gold", legend=False, **line_kw)
            if ensmean:
                sns.lineplot(data=df[iens], x="fhr", y="auc", ax=axr, style="nn", color="black", lw=lw*0.25, legend=False)
                axr.text(df.fhr.max(), df[iens]["auc"].iloc[-1], "ens. mean", **text_kw)
            axr.text(df.fhr.max(), df[imem].groupby("fhr")["auc"].mean().iloc[-1], "auc", **text_kw)
            axr.grid(False) #needs to be after lineplot
            axr.set_ylim((0.9,1))
            axr.set_ylabel("auc")

        plt.setp(topax.get_legend().get_texts(), fontsize='8') # for legend text
        ofile = f"{cl}.png"
        plt.tight_layout()
        fig.savefig(ofile)
        logging.info(os.path.realpath(ofile))
        plt.clf()

if __name__ == "__main__":
    main()
