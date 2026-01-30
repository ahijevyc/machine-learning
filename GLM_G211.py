import argparse
import datetime
from ahijevyc import G211
from ahijevyc import glm as myglm
from glmtools.io.glm import GLMDataset
import logging
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pdb
from scipy.spatial import KDTree
import sys
from tqdm import tqdm
import xarray


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_argparser():
    parser = argparse.ArgumentParser(description = "Accumulate GLM flashes for one hour on G211 grid.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('center', help="center of accumation time window")
    parser.add_argument('twin', type=float, help="total width of accumulation time window in hours")
    parser.add_argument("--clobber", action='store_true', help="clobber existing file(s)")
    parser.add_argument("--maxbad", type=int, default=0, help= (
        "maximum number of corrupt or missing GLM 20-second files in time window to return a file"
        )
                        )
    parser.add_argument('--pool', type=int, default=18, help="workers in pool")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--odir", default="/glade/campaign/mmm/parc/ahijevyc/GLM", help="output path")
    parser.add_argument('-c', '--crd', action='store_true',  # syntax like ncks
                        help="Include lat/lon coordinates in the output file")
    return parser

def bincount(l2, tree, lon_range, lat_range, n):        
    logging.debug(f"load GLMDataset {os.path.basename(l2)}")
    try:
        glm = GLMDataset(l2)
    except Exception as error:
        logging.warning(f"GLMDataset ({l2}) {error}")
        return

    # Get flashes in lat/lon bounds, as flashes are scattered across the full disk. 
    flashes_subset = glm.subset_flashes(lon_range = lon_range, lat_range = lat_range)
    logging.debug(f"{flashes_subset.number_of_flashes.size} flashes")
    if flashes_subset.number_of_flashes.size == 0:
        return np.zeros(n)
    dd, ii = tree.query(np.c_[flashes_subset.flash_lon, flashes_subset.flash_lat], distance_upper_bound=0.5)
    bc = np.bincount(ii) # number of occurences of each ii value
    bc = np.pad(bc, (0, n)) # pad with n zeros on the right side
    bc = bc[0:n]
    return bc

def accum_on_grid(ifiles, lon, lat, maxbad=0, pool=18):
    # allow maxbad bad times (only 20s each)
    lon_range = (lon.min(), lon.max())
    lat_range = (lat.min(), lat.max())
    logging.info(f"lon_range {lon_range} lat_range {lat_range}")

    tree = KDTree(np.c_[lon.ravel(), lat.ravel()])

    logging.info(f"process {len(ifiles)} GLM Datasets (level 2)")
    if pool > 1:
        items = [(ifile, tree, lon_range, lat_range, lon.size) for ifile in ifiles]
        with Pool(pool) as p:
            result = p.starmap(bincount, tqdm(items, total=len(ifiles)))
    else:
        result = [bincount(ifile, tree, lon_range, lat_range, lon.size) for ifile in ifiles]

    # Filter out None's. Those are returned if GLMDataset is corrupt.
    result = [x for x in result if x is not None]

    nbad = len(ifiles) - len(result)
    assert nbad <= maxbad, f"too many bad files ({nbad}/{maxbad+1})."
    if nbad:
        logging.warning(f"{nbad} bad files (max {maxbad+1}).")

    flashes = np.array(result).sum(axis=0)
    flashes = flashes.reshape(lon.shape)
    #flashes = flashes.astype(np.int32) # didn't reduce filesize
    flashes = xarray.DataArray(data=flashes, name="flashes", dims=["y","x"], 
            coords=dict(lon=(["y","x"],lon), lat=(["y","x"],lat)))
    # specific unit strings CDO looks for
    flashes.lat.attrs['units'] = 'degrees_north'
    flashes.lon.attrs['units'] = 'degrees_east'

    # standard_name to identify the axis type
    flashes.lat.attrs['standard_name'] = 'latitude'
    flashes.lon.attrs['standard_name'] = 'longitude'
    return flashes


def main():
    """ 
    Download GLM for time range (twin).
    Accumulate on G211 grid (40km) and half-spacing grid (20km).

    Allow some missing data in the twin-hr window.
    For example, the maximum number of missing files (maxbad) 
    could be equal to the time window in hours. There are 180 files per hour, so 1/180 = 0.6% can be missing.
    If there are more bad files than maxbad, then output file is not created. 
    """

    parser = get_argparser()
    args = parser.parse_args()
    include_coords = args.crd
    clobber = args.clobber
    twin = args.twin
    center = pd.to_datetime(args.center)
    start = center - datetime.timedelta(hours=twin/2)
    end   = center + datetime.timedelta(hours=twin/2)
    odir = os.path.join(args.odir, center.strftime('%Y'))

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    bucket = "noaa-goes19"
    # goes16 stopped GRB broadcast 4/7/2025
    if start < pd.Timestamp("20250407"):
        bucket = "noaa-goes16"

    logging.info(f"download data [{start},{end}]")
    list_of_level2_files = myglm.download(start, end, bucket=bucket, clobber=clobber)
    assert list_of_level2_files is not None, f"glm download {start} {end} {bucket} failed"
    if len(list_of_level2_files) == 0:
        logging.error(f"no level2 files found")
        sys.exit(1)

    # make year directory
    if not os.path.isdir(odir):
        logging.info(f"making new directory {odir}")
        os.makedirs(odir)

    # Global attributes of output netCDF file.
    attrs = {"projection": G211.G211.proj4_init}
    attrs.update(
        dict(
            time_coverage_start=start.isoformat(), 
            time_coverage_center=center.isoformat(),
            time_coverage_end=end.isoformat(), 
            bucket=bucket,
            maxbad=args.maxbad,
            twin=twin,
         )
    )


    ofile = os.path.join(odir, center.strftime("%Y%m%d_%H%M") + f".glm_40km_{twin:.0f}hr.nc")
    if os.path.exists(ofile) and not clobber:
        logging.warning(f"found {ofile} skipping.")
    else:
        grid = G211.GridManager(factor=1)
        flashes = accum_on_grid(list_of_level2_files, grid.lon, grid.lat, maxbad=args.maxbad, pool=args.pool)
        saveflashes(flashes, center, attrs, ofile, include_coords=include_coords)

    # Now do half-distance grid (half the 40km half-grid spacing of G211)
    ofile = os.path.join(odir, center.strftime("%Y%m%d_%H%M") + f".glm_20km_{twin:.0f}hr.nc")
    if os.path.exists(ofile) and not clobber:
        logging.warning(f"found {ofile} skipping.")
    else:
        grid = G211.GridManager(factor=2)
        flashes = accum_on_grid(list_of_level2_files, grid.lon, grid.lat, maxbad=args.maxbad, pool=args.pool)
        saveflashes(flashes, center, attrs, ofile, include_coords=include_coords)

def saveflashes(flashes, center, attrs, ofile, include_coords=False):
    flashes = flashes.expand_dims(time=[center])
    flashes.attrs.update(attrs)

    if not include_coords:
        logging.info("Dropping lat/lon coords from output.")
        flashes = flashes.drop_vars(['lat', 'lon'])
    logging.info(f"{flashes.sum().values} flashes over domain. max {flashes.values.max()} in one cell")
    # Set encoding to minutes or hours since... or else it will be an integer number of days with no fraction. 
    flashes.time.encoding["units"] = "minutes since "+center.isoformat()
    flashes.encoding["zlib"] = True
    # got about 20% compression with complevel=9 or default of 4.
    #flashes.encoding["complevel"] = 9
    flashes.to_netcdf(ofile,unlimited_dims=["time"])
    logging.info(f"created {ofile}")


if __name__ == "__main__":
    main()
