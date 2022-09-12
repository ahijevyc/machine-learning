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
    parser.add_argument("--auc", action="store_true", help="area under ROC curve")
    parser.add_argument("--ensmean", action="store_true", help="ensemble mean")
    parser.add_argument("--nofineprint", action="store_true", help="no fine print (ci, time created, etc)")
    parser.add_argument("--noplot", action="store_true", help="no plot, just print all-forecast hour means)")
    parser.add_argument("-v","--variable", type=str, default="bss", help="variable to plot")
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
    noplot      = args.noplot
    variable    = args.variable

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)


    # Figure dimensions, line thicknesses, text alignment
    fig = plt.figure(figsize=(15,11))
    lw = 2  # line width for mean of members
    text_kw = dict(fontsize=10, ha="left", va="center", clip_on=True) # clip_on in case variable is so low it squishes botax
    # If ci is zero, don't plot confidence band; plot individual lines for all members    
    if ci == 0:
        line_kw = dict(units="mem", estimator=None)
    else:
        # If ci is not zero plot ci% confidence interval
        line_kw = dict(ci=ci)
    if len(ifiles) > 2 and not plotauc:
        line_kw.update(dict(hue="nn"))


    difference_plot = len(ifiles) == 2 # Plot difference between 2 input files on bottom axes

    logging.info(f"Read {len(ifiles)} input files into Pandas DataFrame, with nn column = filename")
    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile.name)) for ifile in ifiles], ignore_index=True) # ignore index or get duplicate indices



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
            imem = df["mem"] != "ensmean.all"
            sns.lineplot(data=df[imem], x="fhr", y=variable,  ax=topax, style="nn", linewidth=lw, **line_kw)
            if ensmean:
                logging.info("ensemble mean")
                sns.lineplot(data=df[iens], x="fhr", y=variable,  ax=topax, style="nn", color="black", lw=lw*0.25, legend=False)
                topax.text(df.fhr.max(), df[iens][variable].iloc[-1], "ens. mean", **text_kw)
                topax.set_xlim(topax.get_xlim()[0], topax.get_xlim()[1]+2) # add space for "ensmean" label
            topax.text(df.fhr.max(), df[imem].groupby("fhr")[variable].mean().iloc[-1], f" {variable}", **text_kw) # looks better preceded by space

            if difference_plot: # Plot difference between 2 input files on bottom axes
                ddf = df.set_index(["class","mem","fhr","nn"])
                nn0, nn1 = ddf.index.unique(level="nn")
                logging.info(f"difference plot ( {nn0} - {nn1} )")
                diff = ddf.xs(nn0, level="nn") - ddf.xs(nn1, level="nn")
                imem_diff = diff.index.get_level_values(level="mem") != "ensmean.all" # can't reuse imem and iens variables; they are used later
                iens_diff = diff.index.get_level_values(level="mem") == "ensmean.all"
                sns.lineplot(data=diff[imem_diff], x="fhr", y=variable,  ax=botax, linewidth=lw, **line_kw)
                botax.set_ylabel(f"{variable} difference")
                if plotauc:
                    # AUC difference on secondary axis (right side)
                    baxr=botax.twinx()
                    sns.lineplot(data=diff[imem_diff], x="fhr", y="auc",  ax=baxr, linewidth=lw, color="gold", **line_kw)
                    baxr.set_ylabel(f"auc difference")
                    baxr.set_ylim(np.array(botax.get_ylim())/10) # make auc y-limits 1/10th variable y-limits
                    baxr.grid(False) #needs to be after lineplot
                botax.set_title(f"{nn0} - {nn1}", fontsize="small")
                if ensmean:
                    sns.lineplot(data=diff[iens_diff], x="fhr", y=variable,  ax=botax, color="black", linewidth=lw*0.25, legend=False)
                    botax.text(df.fhr.max(), diff[iens_diff][variable].iloc[-1], f" ens. mean {variable} diff", fontsize=7, ha="left", va="center")


            # Base rate
            base_rate_ax = botax
            sns.lineplot(data=df, x="fhr", y="base rate", ax=base_rate_ax, linewidth=lw, legend=False, **line_kw) # ignores color arg
            base_rate_ax.text(df.fhr.max(), df["base rate"].iloc[-1], "base rate", **text_kw)
            base_rate_ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

            if variable == "bss":
                ylim = (-0.03,0.35)
                if cl == "flashes": ylim = (0,0.75)
                if cl.startswith("torn") or cl.startswith("sig"): ylim = (-0.03, 0.1)
                if cl.startswith("hailone"): ylim = (-0.02, 0.18)
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

            if len(ifiles) >= 8:
                topax.legend(*topax.get_legend_handles_labels(), ncol=2, fontsize=7, labelspacing=0.45, columnspacing=1, title=prefix, title_fontsize=8) 
            ofile = f"{os.path.join(os.path.dirname(prefix),cl+os.path.basename(prefix))}.png"
            plt.tight_layout()
            fig.savefig(ofile)
            logging.info(os.path.realpath(ofile))
            plt.clf()

        # Look at the aggregrate scores for "all" forecast hours. Not what was plotted.
        df_all = df_all[df_all.mem == "ensmean.all"].set_index("nn")
        df_all = df_all.sort_values(variable,ascending=False)
        print(df_all[[variable,"base rate","auc","aps"]])

if __name__ == "__main__":
    main()
