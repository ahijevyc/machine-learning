import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import xarray


debug=False

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
            ax.set_xlabel(da.attrs["units"])
        else:
            ax.plot(qn_a, qn, label=name)
            units = da.attrs["units"]
            assert units == ax.get_xlabel()
            ax.set_ylabel(units)
    l = ax.legend()

fields = ['255_0mb_above_ground-CAPE', 'surface-CAPE', '90_0mb_above_ground-CAPE']

search_string = "/glade/work/ahijevyc/NSC_objects/HRRR/202*HRRR-ZARR*.nc"
ifiles = sorted(glob.glob(search_string))
print(f"found {len(ifiles)} files matching {search_string}")
itimes = [datetime.datetime.strptime(os.path.basename(i),'%Y%m%d%H_HRRR-ZARR_upscaled.nc') for i in ifiles]
if debug:
    for ifile in ifiles:
        print("opening", ifile)
        ds = xarray.open_mfdataset(ifile)[fields]
    
ds = xarray.open_mfdataset(ifiles)

# Sanity check
assert (ds.forecast_reference_time == np.array(itimes).astype(np.datetime64)).all(), "file names do not match DataFrame forecast_reference_times"

ds = ds.transpose("forecast_reference_time", "forecast_period", "pts")
print("grabbing",fields)
ds = ds[fields]
ds = ds.mean(dim="forecast_period", keep_attrs=True).mean(dim="forecast_reference_time", keep_attrs=True)
plot_qq(ds)
plt.suptitle(search_string)
plt.savefig('qq.png')
