import matplotlib.pyplot as plt
import sys
import time
import xarray

yyyymm = sys.argv[1]  # "202101" Only one month at a time is reasonable
t = time.perf_counter()
ds = xarray.open_mfdataset(f'/glade/work/ahijevyc/NSC_objects/HRRR/{yyyymm}????_HRRR-ZARR_upscaled.nc', compat="override", coords="minimal", join="exact", preprocess=lambda x:x.isel(forecast_period=2).mean(dim="pts"))
print(time.perf_counter()-t)

ts = ds.to_array()
xx = ts.plot(col="variable", sharey=False, col_wrap=6).set_ylabels(label="").set_titles(template='{value}',size="xx-small")
plt.savefig(f"{yyyymm}.png")


