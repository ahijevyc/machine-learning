import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import os

datadirs = ["HRRR/grid_data","HRRR/grid_data_new","HRRR/grid_data"] 
datadirs = ["HRRR_new/grid_data/grid_data_HRRR_", "HRRR_new/grid_data/grid_data_HRRRX_"]

columns = ["W_UP_MAX", "W_DN_MAX","W_UP_MAX-N5T5","W_DN_MAX-N5T5"]

fig, axes = plt.subplots(ncols=len(datadirs), figsize=(len(datadirs)*4,len(columns)))

for ax, grid_data in zip(axes, datadirs): 
    search_str = f'/glade/work/sobash/NSC_objects/{grid_data}*.par'
    files = glob.glob(search_str)
    print(f"{len(files)} files {search_str}")
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files])#, index=files)
    print(df.describe())
    p = df.boxplot(ax=ax)
    p.set_title(grid_data, fontsize="small")

ofile = "W_DN_MAX_boxplot.png"
plt.savefig(ofile)
print("created", os.path.realpath(ofile))

