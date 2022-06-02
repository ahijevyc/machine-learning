import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import pdb
import seaborn as sns
import sys

sns.set_theme()

def nns(ifile):
    # abbreviate input file name for legend 
    base, ext = os.path.splitext(ifile)
    words = base.split('.')
    words = words[1:-1] # ignore 1st and last words
    return '.'.join(words)

def main():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "plot NN verification scores written by GLM_HRRR_test.py, often in /glade/work/ahijevyc/NSC_objects/nn/.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ifiles', nargs="+", type=argparse.FileType("r"), help="input file(s)")
    parser.add_argument("--ci", type=int, default=95, help="confidence interval")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
    parser.add_argument("--noensmean", action="store_true", help="no ensemble mean")
    parser.add_argument("--noauc", action="store_true", help="no ROC curve")

    args = parser.parse_args()
    ifiles  = args.ifiles
    debug   = args.debug
    ensmean = not args.noensmean
    plotauc = not args.noauc
    ci      = args.ci # default ci is 95

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)

    fig = plt.figure(figsize=(10,7))
    lw = 2  # mean of members
    text_kw = dict(fontsize=10, ha="left", va="center")

    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile.name)) for ifile in ifiles])

    for cl, df in dfs.groupby("class"):
        topax = plt.subplot2grid((3,1),(0,0), rowspan=2)
        botax = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=topax)
        logging.info(f"class={cl}")
        df = df.reset_index(drop=True) # multiple csv files makes duplicates

        iens = df["mem"] == "ensmean"
        imem = df["mem"] != "ensmean"
        sns.lineplot(data=df[imem], x="fhr", y="bss",  ax=topax, style="nn", linewidth=lw, ci=ci)
        if ensmean:
            sns.lineplot(data=df[iens], x="fhr", y="bss",  ax=topax, style="nn", color="black", lw=lw*0.25, legend=False)
            topax.text(df.fhr.max(), df[iens]["bss"].iloc[-1], "ensmean", **text_kw)
            topax.set_xlim(topax.get_xlim()[0], topax.get_xlim()[1]+2) # add space for "ensmean" label
        topax.text(df.fhr.max(), df[imem].groupby("fhr")["bss"].mean().iloc[-1], "bss", **text_kw)

        if len(ifiles) == 2:
            logging.info("difference plot")
            ddf = df[imem].set_index(["class","mem","fhr","nn"])
            nn0, nn1 = ddf.index.unique(level="nn")
            diff = ddf.xs(nn0, level="nn") - ddf.xs(nn1, level="nn")
            sns.lineplot(data=diff, x="fhr", y="bss",  ax=botax, linewidth=lw, ci=ci)
            sns.lineplot(data=diff, x="fhr", y="auc",  ax=botax, linewidth=lw, color="gold", ci=ci)
            botax.set_ylabel("diff")


        sns.lineplot(data=df, x="fhr", y="base rate", ax=topax, style="nn", linewidth=lw, color="red", legend=False)
        topax.set_ylabel(topax.get_ylabel()+" and base rate")
        topax.text(df.fhr.max(), df["base rate"].iloc[-1], "base\nrate", **text_kw)
        topax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ylim = (0,0.225)
        if cl == "flashes": ylim = (0,0.6)
        if cl == "torn_rptdist_2hr": ylim = (-0.1, 0.1)
        topax.set_ylim(ylim)

        if plotauc:
            axr=topax.twinx()
            sns.lineplot(data=df[imem], x="fhr", y="auc", ax=axr, style="nn", linewidth=lw, color="gold", legend=False, ci=ci)
            if ensmean:
                sns.lineplot(data=df[iens], x="fhr", y="auc", ax=axr, style="nn", color="black", lw=lw*0.25, legend=False)
                axr.text(df.fhr.max(), df[iens]["auc"].iloc[-1], "ensmean", **text_kw)
            axr.text(df.fhr.max(), df[imem].groupby("fhr")["auc"].mean().iloc[-1], "auc", **text_kw)
            axr.grid(False) #needs to be after lineplot
            axr.set_ylim((0.8,1))
            axr.set_ylabel("auc")

        plt.setp(topax.get_legend().get_texts(), fontsize='8') # for legend text
        ofile = f"{cl}.png"
        plt.tight_layout()
        fig.savefig(ofile)
        logging.info(os.path.realpath(ofile))
        plt.clf()

if __name__ == "__main__":
    main()
