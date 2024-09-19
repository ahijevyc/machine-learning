""" 
First got permission to read files from Wiebke and Dan.
Give credit to Earth Networks in publications/presentations.
Tarred and gzipped files in /gpfs/csfs1/ral/aap/hardt/LTG/.
Text file lines are either flashes or pulses. Pulses are components of a flash. 
Grep all flash lines (with the letter "f") to new txt file.
Store in /glade/campaign/mmm/parc/ahijevyc/wbug_lightning/

Convert text files to parquet with txt2par.py (one at a time).

This script concatenates those parquet files (listed on command line) that
hold individual unmapped flashes. It puts them in half-hourly bins and maps to grid.

Aggregate as many parquet files as practical before grouping by time to avoid edge effects.
The original txt files from which the parquet files were made had ragged temporal edges. 
Flashes from the same hour could be split between different files. 

python wbug_G211.py /glade/campaign/mmm/parc/ahijevyc/wbug_lightning/flash20*.par

This script puts flashes in 30 minute bins, and grids to G211 grid (40km) and half-grid spacing G211 grid (20km).
"""
import datetime
import dask.dataframe as dd
import G211
import glob
import logging
import numpy as np
import os
import pandas as pd
import pdb
from scipy.spatial import KDTree
import sys
import xarray


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def bintimespace(df, tree):
    logging.debug(f"{df.name} {len(df)} ungridded flashes")
    dd, ii = tree.query(np.c_[df.lon, df.lat], distance_upper_bound=0.5)
    bc = np.bincount(ii) # number of occurences of each ii value
    bc = np.pad(bc, (0, tree.n), constant_values=0) # pad right side with zeros
    bc = bc[0:tree.n] # trim at tree.n elements (eliminate out-of-bounds bin count)
    if bc.sum():
        logging.debug(f"{df.name} {bc.sum()} gridded flashes. max {bc.max()}")
    df = pd.Series(bc, name=df.name, dtype=np.uint32)
    return df

def binflashes(df, grid_lon, grid_lat):
    logging.info(f"KDTree for {len(df)} flashes.")
    tree = KDTree(np.c_[grid_lon.ravel(), grid_lat.ravel()])
    logging.info(f"bin flashes in space")
    meta_df = pd.DataFrame(columns=range(grid_lon.size)) # this is what output from bintimespace() looks like
    flashes = df.groupby(["time_coverage_start","incloud"]).apply(bintimespace, tree, meta=meta_df).compute()
    # TODO: Figure out how to reshape and add cg and ic to xarray together.
    cg = flashes.xs(0, level="incloud")
    logging.info(f"cloud-to-ground")
    data = cg.values.reshape((-1,*grid_lon.shape)) 
    cg = xarray.DataArray(
            data=data,
            name="cg",
            dims=["time_coverage_start","y","x"],
            coords=dict(
                time_coverage_start=(["time_coverage_start"], cg.index),
                lon=(["y","x"],grid_lon),
                lat=(["y","x"],grid_lat)
                )
            )
    # in-cloud
    ic = flashes.xs(1, level="incloud")
    logging.info(f"in-cloud")
    data = ic.values.reshape((-1,*grid_lon.shape)) 
    ic = xarray.DataArray(
            data=data,
            name="ic",
            dims=["time_coverage_start","y","x"],
            coords=dict(
                time_coverage_start=(["time_coverage_start"], ic.index),
                lon=(["y","x"],grid_lon),
                lat=(["y","x"],grid_lat)
                )
            )
    logging.info("merge cg and ic")
    ds = xarray.merge([cg,ic]).fillna(0) # replace missing values (nan) with zero
    ds.attrs.update(G211.g211.proj4_params)
    for array in ds:
        ds[array].encoding["zlib"] = True
    return ds

ifiles = sys.argv[1:]
clobber = False

# Read parquet files all together because the first flashes in each file belong to the previous file. 
logging.info(f"read {len(ifiles)} parquet files")
df = dd.read_parquet(ifiles)

# Sanity check. Should be okay because txt2par.py removed bad lines.
time0 = pd.to_datetime("20180101")
past = df["time"] < time0
future = df["time"] >= datetime.datetime.utcnow()
outofbounds = (df["lat"] < 11) | (df["lat"] > 62) | (df["lon"] < -150) | (df["lon"] > -50)

assert not any(past), f"{past.sum()} ({past.sum()/len(df):%}) flashes before {time0}"
assert not any(future), f"{future.sum()} ({future.sum()/len(df):%}) flashes in future"
assert not any(outofbounds), f"{outofbounds.sum()} ({outofbounds.sum()/len(df):%}) flashes out-of-bounds"
assert df["incloud"].isin([0,1]).all(), "incloud should be 0 or 1"

logging.info(f"truncate time to multiple of 30 minutes")
df["time_coverage_start"] = df["time"].dt.floor(freq="30min")

ofile = os.path.realpath(f"flash.40km_30min.nc")
if not os.path.exists(ofile) or clobber:
    grid_lon = G211.lon
    grid_lat = G211.lat
    binnedflashes = binflashes(df, grid_lon, grid_lat)
    binnedflashes.to_netcdf(ofile, unlimited_dims=["time_coverage_start"])
    logging.info(f"created {ofile}")
else:
    logging.warning(f"{ofile} already exists")

ofile = os.path.realpath(f"flash.20km_30min.nc")
if not os.path.exists(ofile) or clobber:
    halfgrid = G211.x2()
    grid_lon, grid_lat = halfgrid.lon, halfgrid.lat
    binnedflashes = binflashes(df, grid_lon, grid_lat)
    binnedflashes.to_netcdf(ofile, unlimited_dims=["time_coverage_start"])
    logging.info(f"created {ofile}")
else:
    logging.warning(f"{ofile} already exists")

