import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

search_str = '/glade/work/sobash/NSC_objects/HRRR/grid_data/*_HRRR_d01_20*par'
files = glob.glob(search_str)
print(f"{len(files)} files")
columns = ["W_UP_MAX", "W_DN_MAX","W_UP_MAX-N5T5","W_DN_MAX-N5T5"]
df = pd.concat([pd.read_parquet(f, columns=columns) for f in files])#, index=files)
print(df.describe())
p = df.boxplot()
plt.suptitle(search_str)
plt.savefig("W_DN_MAX_boxplot.png")
pdb.set_trace()
