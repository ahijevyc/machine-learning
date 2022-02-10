import s3fs
import xarray as xr
from datetime import datetime, timedelta
import numpy as np


# Test whether forecast is a repeat of the previous hour. 
# If you find one add it to the list in ./HRRR/upscale_HRRR-ZARR.py

var_name = "CAPE"
var_level = "surface"

fs = s3fs.S3FileSystem(anon=True)
dss = []

reference_times = [datetime(2021, 1, 28, 18)]
reference_times.append(reference_times[0] + timedelta(hours=1))

for reference_time in reference_times:

    root_url = reference_time.strftime("s3://hrrrzarr/sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr")

    s3_lookups = [s3fs.S3Map(url, s3=fs) for url in [f"{root_url}/{var_level}/{var_name}",
                                                     f"{root_url}/{var_level}/{var_name}/{var_level}"]]

    ds = xr.open_mfdataset(s3_lookups, engine="zarr")
    print(f"requested {reference_time}, got {ds.forecast_reference_time.values}.")

    dss.append(ds.sel(time=reference_times[1]))

print(dss[1].equals(dss[0]))

