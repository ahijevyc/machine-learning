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
    
    # 1. Convert to Dataset so we can add the projection variable
    ds = flashes.to_dataset()

    # 2. Create CF-compliant grid mapping variable (G211 Lambert Conformal)
    ds['projection'] = xarray.DataArray(np.array('', dtype='S1'), dims=())
    ds['projection'].attrs = {
        'grid_mapping_name': 'lambert_conformal_conic',
        'standard_parallel': [25.0, 25.0],
        'longitude_of_central_meridian': -95.0,
        'latitude_of_projection_origin': 39.0,
        'false_easting': 0.0,
        'false_northing': 0.0,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
        'proj4_params': attrs.get("projection", G211.G211.proj4_init)
    }

    # 3. Coordinate Attributes (Strict CF Requirements)
    # Ensure x and y have standard names and axis types
    if 'x' in ds.coords:
        ds.x.attrs.update({'standard_name': 'projection_x_coordinate', 'axis': 'X', 'units': 'm'})
    if 'y' in ds.coords:
        ds.y.attrs.update({'standard_name': 'projection_y_coordinate', 'axis': 'Y', 'units': 'm'})
    
    ds.time.attrs.update({'standard_name': 'time', 'axis': 'T'})

    ds['flashes'].attrs.update({
        'grid_mapping': 'projection',
        'coordinates': 'y x', # Space-separated list of coordinate names
        'units': '1',         # '1' is the CF standard for 'count' or 'dimensionless'
        'standard_name': 'lightning_flash_count' # A recognized (though custom here) standard name
    })
    
    # 4. Cleanup and Global Attributes
    ds.attrs.update(attrs)
    ds.attrs['Conventions'] = 'CF-1.8'

    if not include_coords:
        logging.info("Dropping lat/lon coords from output.")
        ds = ds.drop_vars(['lat', 'lon'], errors='ignore')

    logging.info(f"{ds.flashes.sum().values} flashes over domain. max {ds.flashes.values.max()} in one cell")

    # 5. Strict Encoding (Forces int32 for both Flashes AND Time)
    encoding = {
        "flashes": {
            "dtype": "int32",
            "zlib": True,
            "complevel": 4,
            "_FillValue": -1
        },
        "time": {
            "dtype": "int32",  # Forcing time to int32 as requested
            "units": "minutes since " + center.strftime('%Y-%m-%d %H:%M:%S'),
            "calendar": "proleptic_gregorian",
        },
    }

    ds.to_netcdf(ofile, encoding=encoding, unlimited_dims=["time"])
    logging.info(f"created {ofile}")


if __name__ == "__main__":
    main()
