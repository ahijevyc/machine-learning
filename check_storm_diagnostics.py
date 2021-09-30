#!/usr/bin/env python

import numpy as np
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
import os
import pandas as pd

def read_csv_files(r):
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        csv_file = '/glade/work/sobash/NSC_objects/track_data_ncargrib_2019_csv/track_step_NCARGRIB_mem1_%s-0000_13.csv'%(yyyymmdd)
        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print 'Reading %s files'%(len(all_files))
    df = pd.concat((pd.read_csv(f) for f in all_files))

    return df, len(all_files)

sdate = dt.datetime(2019,4,19,0,0,0)
edate = dt.datetime(2019,6,24,0,0,0)
dateinc = dt.timedelta(days=1)

df, numfcsts = read_csv_files('3km')

#cols = df.columns
#for c in cols: print df[c].describe()

import seaborn as sns
sns.set_style("white", {"axes.linewidth":0.5})

xticks, yticks = [0,0.01,0.02,0.03,0.04,0.05], [0,0.01,0.02,0.03,0.04,0.05]
xticks, yticks = [0,0.01,0.02], [0,0.01,0.02]
xticks, yticks = np.arange(0,2,0.1), range(0,800,25)
xfieldname, yfieldname = 'eccentricity', 'area'

xmax, ymax = xticks[-1], yticks[-1]

xfield, yfield = df[xfieldname], df[yfieldname]

df['orientation'].plot.hist(bins=50)

plt.savefig('histogram.png', dpi=150)
