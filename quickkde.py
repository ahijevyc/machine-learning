import argparse
import cartopy
import datetime
import G211
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
import pdb
import seaborn as sns

dpi = 170


# =============Arguments===================
parser = argparse.ArgumentParser(description = "kdeplot of labels", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", action="store_true", help="debug info")
parser.add_argument("ifile", type=argparse.FileType("r"), help="path to *.csv file") 
parser.add_argument("--odir", type=str, default="/glade/scratch/ahijevyc/temp/HWT_mode_output", help="save pngs here")

# Assign arguments to simple-named variables
args = parser.parse_args()
debug         = args.debug
ifile         = os.path.realpath(args.ifile.name)
odir          = args.odir

if debug: print(args)

print("Reading labels", ifile)
df = pd.read_csv(ifile)

#df = df[df.UP_HELI_MAX_max >= 75]


def add_fineprint(text=None, **kwargs):
    annotate_dict = {"xy":(3,2), "xycoords":'figure pixels', "fontsize":5}
    annotate_dict.update(kwargs)
    if not text:
        text = f"\ncreated {datetime.datetime.now(tz=None)}"
    fineprint_obj = plt.annotate(text=text, **annotate_dict)
    return fineprint_obj

fineprint = "\ncreated "+str(datetime.datetime.now(tz=None)).split('.')[0]

if debug:
    pdb.set_trace()

df["label"] = df.iloc[:,-3:].idxmax(axis="columns")

print(df["label"].value_counts())

ax = plt.axes(projection=G211.g211)
sns.kdeplot(
        ax = ax,
        common_norm=False,
        cut=0,
        data=df,
        x="Centroid_Lon",
        y="Centroid_Lat",
        hue="label",
        fill=False,
        thresh=0.25,
        levels=4,
        linestyles=["dotted","solid","solid"],
        linewidths=3,
        transform=cartopy.crs.PlateCarree()
        )
ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), linewidth=0.25)
ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linewidth=0.25)
ax.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.05)
ax.add_feature(cartopy.feature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25, facecolor='k', alpha=0.05)
fp = add_fineprint(fineprint)
ofile = f"{odir}/kdeplot.png"
ax.figure.savefig(ofile,dpi=dpi)
print(f"made {os.path.realpath(ofile)}")
plt.close()

