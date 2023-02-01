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
    parser.add_argument('ifiles', nargs="+", type=str, help="input file(s)")
    parser.add_argument("--ci", type=int, default=95, help="confidence interval. Show individual lines if ci=0")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
    parser.add_argument("--dpi", type=int, default=120, help="output resolution")
    parser.add_argument("--ensmean", action="store_true", help="ensemble mean")
    parser.add_argument("--lw", type=float, default=2, help="line width")
    parser.add_argument("--mask", type=str, nargs="+", help="only show this(these) mask value(s)")
    parser.add_argument("--n_boot", type=int, default=1000, help="number of bootstrap sammples")
    parser.add_argument("--nofineprint", action="store_true", help="no fine print (ci, time created, etc)")
    parser.add_argument("--nofits", action="store_true", help="don't show individual fits")
    parser.add_argument("--noplot", action="store_true", help="no plot, just print all-forecast hour means)")
    parser.add_argument("-v","--variable", type=str, default="bss", help="variable to plot")
    parser.add_argument("--xmax",type=float, help="maximum on x-axis")
    parser.add_argument("--xmin",type=float, help="minimum on x-axis")
    parser.add_argument("--ymax",type=float, help="maximum on y-axis")
    parser.add_argument("--ymin",type=float, help="minimum on y-axis")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ifiles      = args.ifiles
    ci          = args.ci # default ci is 95
    debug       = args.debug
    dpi         = args.dpi
    ensmean     = args.ensmean
    lw          = args.lw
    mask        = args.mask
    n_boot      = args.n_boot
    nofineprint = args.nofineprint
    nofit       = args.nofits
    noplot      = args.noplot
    variable    = args.variable
    xmax        = args.xmax
    xmin        = args.xmin
    ymax        = args.ymax
    ymin        = args.ymin

    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)
    logging.debug(args)


    # Figure dimensions, line thicknesses, text alignment
    fig = plt.figure(figsize=(11,8.5))
    text_kw = dict(fontsize=10, ha="left", va="center", clip_on=True) # clip_on in case variable is so low it squishes botax
    # If ci is zero, don't plot confidence band; plot individual lines for all fits
    if ci == 0:
        line_kw = dict(units="fit", estimator=None)
    else:
        # If ci is not zero plot ci% confidence interval
        line_kw = dict(errorbar=('ci',95), n_boot=n_boot)

    line_kw.update(dict(hue="nn", style="nn", lw=lw))

    logging.info(f"Read {len(ifiles)} input files into Pandas DataFrame, with new nn column equal to input filename")
    dfs = pd.concat([pd.read_csv(ifile,header=0).assign(nn=nns(ifile)) for ifile in ifiles], ignore_index=True) # ignore index or get duplicate indices

    # If DataFrame has a "mask" column, use it to signify linewidth.
    if "mask" in dfs:
        line_kw.update(dict(size="mask"))
        if mask is not None:
            dfs = dfs[dfs["mask"].isin(mask)]

    # Append fold to fit and drop fold column.
    dfs["fit"] = dfs["fit"] + "." + dfs["fold"]
    dfs = dfs.drop(columns="fold")

    # common prefix for nn
    prefix = os.path.commonprefix(dfs["nn"].tolist())
    # remove common prefix from nn (make it shorter) 
    if len(ifiles) > 1: # otherwise nn is empty and doesn't trigger legend
        dfs["nn"] = dfs["nn"].str.replace(prefix,"",regex=False)

    # Loop thru types of event (torn, wind, hail, lightning)
    for cl, df in dfs.groupby("class"):
        print(f"\n{cl}    prefix={prefix}")
        # Separate forecast_hour=all
        df_all = df[df.forecast_hour=="all"]
        df = df.copy()[df.forecast_hour != "all"]
        df["forecast_hour"] = df["forecast_hour"].astype(int)
        if not noplot:
            topax = plt.subplot2grid((3,1),(0,0), rowspan=2)
            botax = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=topax)
            # Empty fineprint_string placeholder for fine print in lower left corner of image.
            # counterintuitively, larger y offset makes more space for fineprint at bottom (cause of tight_layout)
            fineprint = plt.annotate(text="", xy=(0,-62), xycoords=('axes fraction','axes pixels'), va="top", fontsize=6, wrap=True)
            if not nofineprint:
                txt = args.__dict__.copy()
                # ifiles list is too long for fineprint. Can get them from legend and title anyway.
                del(txt["ifiles"])
                fineprint.set_text(f"{txt}\ncreated {datetime.datetime.now()}")

            iens = df["fit"] == "ensmean.all"
            if not nofit:
                sns.lineplot(data=df[~iens], x="forecast_hour", y=variable,  ax=topax, **line_kw)
            if ensmean:
                logging.info("ensemble mean")
                lp = sns.lineplot(data=df[iens], x="forecast_hour", y=variable,  ax=topax, **line_kw, legend=nofit)
                # Used to label with "ens. mean". but it keeps labeling fits mean too. (the one within the CI band)

            # Base rate
            base_rate_ax = botax
            sns.lineplot(data=df, x="forecast_hour", y="base_rate", ax=base_rate_ax, legend=False, **line_kw) # ignores color arg
            base_rate_ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

            if variable == "bss":
                ylim = (-0.03,0.35)
                if cl.startswith("flashes") or cl.startswith("cg_") or cl.startswith("ic_") : ylim = (0,0.75)
                if cl.startswith("torn") or cl.startswith("sig"): ylim = (-0.03, 0.12)
                #if cl.startswith("hailone"): ylim = (-0.02, 0.18)
                topax.set_ylim(ylim)

            if xmax is not None:
                topax.set_xlim(right=xmax)
            if xmin is not None:
                topax.set_xlim(left=xmin)
            if ymax is not None:
                topax.set_ylim(top=ymax)
            if ymin is not None:
                topax.set_ylim(bottom=ymin)

            handles, labels = topax.get_legend_handles_labels()
            fontsize = 7
            if len(prefix) > 100:
                fontsize=6
            if len(handles) > 8:
                topax.legend(handles, labels, ncol=2, fontsize=fontsize, labelspacing=0.45, columnspacing=1, title=prefix,
                        handlelength=3, title_fontsize=fontsize*1.1) #default handlelength=2. to see entire cycle of long patterns
            else:
                topax.legend(handles, labels, fontsize=fontsize, title=prefix, title_fontsize=fontsize*1.1)
            ofile = f"{os.path.join(os.getenv('TMPDIR',os.path.dirname(prefix)),cl+'.'+os.path.basename(prefix))}.png"
            plt.tight_layout()
            fig.savefig(ofile, dpi=dpi)
            logging.info(os.path.realpath(ofile))
            plt.clf()

        # Look at the aggregrate scores for "all" forecast hours. Not what was plotted.
        df_all = df_all[df_all.fit == "ensmean.all"].set_index("nn")
        df_all = df_all.sort_values(variable,ascending=False)
        columns = ["bss","base_rate","auc","aps"]
        columns_added_later = ["mask", "n"]
        for col in columns_added_later:
            if col in dfs:
                columns.append(col)
        print(df_all[columns])

if __name__ == "__main__":
    main()
