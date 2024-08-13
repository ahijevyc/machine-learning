from pathlib import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

"""
Compare scaling values of different training sets.
Mean, std, etc.
"""

sns.set_theme()
gs = ["1km", "3km-12sec", "15km"]
ifiles = ["data/scaling_values_NSC" + x + "_20160701_1200.pk" for x in gs]
df = pd.concat([pd.read_pickle(f) for f in ifiles], axis=0, keys=gs, names=["gs","stat"])

nrows=3
ncols=4
n = nrows*ncols
fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,10), sharex=True)
for i,c in enumerate(df.columns):
    df[c].unstack().drop(columns="count").plot.bar(ax=axes.flatten()[i%n],legend=i%n == 0, title=c)
    # final axes in figure
    if i % n == n-1:
        ofile = Path(os.getenv("TMPDIR")) / f"NSC1-3-15_scaling_values{i/n:02.0f}.png"
        plt.savefig(ofile)
        print(f"created {ofile}")
        [a.clear() for a in axes.flatten()]
