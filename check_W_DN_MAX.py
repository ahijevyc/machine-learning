import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import os

datadirs = ["grid_data","grid_data_new"] 
fig, axes = plt.subplots(ncols=len(datadirs), figsize=(len(datadirs)*4,4))

for ax, grid_data in zip(axes, datadirs): 
    search_str = f'/glade/work/sobash/NSC_objects/HRRR/{grid_data}/*_HRRRX_d01_20*par'
    files = glob.glob(search_str)
    print(f"{len(files)} files {search_str}")
    columns = ["W_UP_MAX", "W_DN_MAX","W_UP_MAX-N5T5","W_DN_MAX-N5T5"]
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files])#, index=files)
    print(df.describe())
    p = df.boxplot(ax=ax)
    p.set_title(search_str, fontsize="small")

ofile = "W_DN_MAX_boxplot.png"
plt.savefig(ofile)
print("created", os.path.realpath(ofile))

