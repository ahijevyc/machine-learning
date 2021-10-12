import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd
import pdb
import pickle


def plot_qq(ds):

    # Input 
    # ds: xarray DataSet
    ax = plt.gca()
    #percs = list(range(0,100))
    percs =np.append(  np.arange(90,99.99,0.01), np.arange(99.99,99.99999,0.0001) )
    print(percs)

    # Loop through DataArrays
    # The first one, plot 1:1 line and remember percentiles so you can compare others to them.
    # For others plot their qq line.
    compare_to = None
    for name, da in ds.data_vars.items():
        qn = np.nanpercentile(da, percs)
        print(name,qn)
        if compare_to is None:
            compare_to = name
            qn_a = qn
            x = np.linspace(ds.to_array().min(), ds.to_array().max())
            ax.plot(x,x, color="k", ls="--", label=name)
        else:
            ax.plot(qn_a, qn, label=name)

    l = ax.legend()
    plt.savefig('qq.png')

fields = ['255_0mb_above_ground/CAPE', 'surface/CAPE', '90_0mb_above_ground/CAPE']

mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb')).reshape([65,93])

search_string = "/glade/work/ahijevyc/NSC_objects/HRRR/*HRRR-ZARR*.par"
ifiles = glob.glob(search_string)
print(f"found {len(ifiles)} files matching {search_string}")
itimes = [dt.datetime.strptime(os.path.basename(i),'%Y%m%d%H_HRRR-ZARR_upscaled.par') for i in ifiles]
dfs = [pd.read_parquet(i) for i in ifiles]
ds = pd.concat(dfs).set_index("forecast_reference_time", append=True, verify_integrity=True).to_xarray()

# Sanity check
assert (ds.forecast_reference_time == np.array(itimes).astype(np.datetime64)).all(), "file names do not match DataFrame forecast_reference_times"

ds = ds.transpose("forecast_reference_time", "forecast_period", "projection_y_coordinate", "projection_x_coordinate")
ds = ds.mean(dim="forecast_period").mean(dim="forecast_reference_time")
ds.coords['mask'] = (("projection_y_coordinate", "projection_x_coordinate"), mask)
p = ds[fields].to_array().where(ds.mask, drop=True).plot(col="variable", robust=True)

plt.savefig('test.png')

plt.clf()
plot_qq(ds[fields])
