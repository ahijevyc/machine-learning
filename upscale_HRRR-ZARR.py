#!/usr/bin/env python

import argparse
import cartopy.crs
from datetime import *
import matplotlib.pyplot as plt
import metpy 
import metpy.calc as mcalc
from metpy.units import units
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import pdb
import pickle
import s3fs
import scipy.ndimage.filters
from scipy import spatial
import sys
import xarray


### THIS CODE EVOLVED FROM CODE WITHIN /glade/u/home/sobash/NSC_scripts
### TO UPSCALE 3-KM CAM DATA TO AN 80-KM GRID

def get_closest_gridbox():
    ### find closest 3-km or 1-km grid point to each 80-km grid point
    gpfname = f'{odir}/NSC_objects/nngridpts_80km_{model}.pk'
    if os.path.exists(gpfname):
        nngridpts = pickle.load(open(gpfname, 'rb'), encoding='bytes')
    else:
        print('finding closest grid points')
        # INTERPOLATE TO 80KM GRID
        awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
        grid81 = awips.makegrid(93, 65, returnxy=True)
        x81, y81 = awips(grid81[0], grid81[1])
        #x81 = (x[1:,1:] + x[:-1,:-1])/2.0
        #y81 = (y[1:,1:] + y[:-1,:-1])/2.0

        f = xarray.open_dataset('/glade/work/ahijevyc/share/HRRR.nc')
        lats = f['gridlat_0']
        lons = f['gridlon_0']
        f.close()
        xy = awips(lons.values.ravel(), lats.values.ravel())
        tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
        nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
        pickle.dump(nngridpts, open(gpfname, 'wb'))

    return nngridpts

def scipyfilt(da):
    this_field = da.name
    print(f"upscaling {this_field}")
    da = da.astype(np.float32) # avoid *** RuntimeError: array type dtype('float16') not supported from scipy.ndimage
    # use maximum for certain fields, mean for others 
    if this_field in ['MAXUVV_1hr_max_fcst', 'MAXREF_1hr_max_fcst', 'WIND_1hr_max_fcst', 'MXUPHL_1hr_max_fcst', 'HAIL_1hr_max_fcst']: 
        field = scipy.ndimage.filters.maximum_filter(da, size=(1,27,27), mode='reflect') # default mode='reflect' (d c b a | a b c d | d c b a) The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric
    elif this_field in ['MAXDVV_1hr_max_fcst', 'MNUPHL_1hr_min_fcst']:
        field = scipy.ndimage.filters.minimum_filter(da, size=(1,27,27))
    else:
        field = scipy.ndimage.filters.uniform_filter(da, size=(1,27,27))
    field_interp = np.empty((da.time.size,65,93))
    for t,_ in enumerate(field):
        field_interp[t] = field[t].flatten()[nngridpts[1]].reshape((65,93))
    ds = xarray.Dataset(data_vars={this_field:(da.dims,field_interp)})
    ds[this_field].attrs.update(da.attrs)
    return ds

def rename_upscale(ds): # how to handle multiple levels with same variable name
    # rename dataarray so it includes the level. 
    # Otherwise, ValueError: Could not find any dimension coordinates to use to order the datasets for concatenation.
    # If you have two CAPEs from different levels. 
    # use long_name attribute instead. It has the level and name.
    # for example: CAPE -> 0_3000m_above_ground/CAPE or CAPE -> surface/CAPE
    # Don't try to rename forecast time variables
    assert 'forecast_period' not in ds.data_vars
    for da in ds:
        long_name = ds[da].attrs['long_name']
        ds = ds.rename({da:long_name})
        return scipyfilt(ds[long_name])

def upscale_forecast(upscaled_field_list,nngridpts,debug=False):

    # Open HRRR-ZARR forecasts and return xarray Dataset.
    # Ignores analysis fhr=0 (_anl.zarr).

    fs = s3fs.S3FileSystem(anon=True)


    level, variable = upscaled_field_list[0]
    # url without final level subdirectory has time, projection_x_coordinate, forecast_period, and forecast_reference_time
    coord_url = os.path.join('s3://hrrrzarr/sfc', sdate.strftime("%Y%m%d/%Y%m%d_%Hz_fcst.zarr"), level, variable)
    coord_ds = xarray.open_dataset(s3fs.S3Map(coord_url, s3=fs), engine='zarr')
    coord_ds = coord_ds.drop(labels=["projection_x_coordinate", "projection_y_coordinate"]) # projection coordinates will be different after upscaling
    urls = [os.path.join('s3://hrrrzarr/sfc', sdate.strftime("%Y%m%d/%Y%m%d_%Hz_fcst.zarr"), level, variable, level) for (level, variable) in upscaled_field_list]
    if debug:
        # One at a time to find cause of zarr.errors.GroupNotFoundError: group not found at path ''
        # usually a typo in url
        for url in urls:
            print(f"getting {url}")
            ds = xarray.open_dataset(s3fs.S3Map(url, s3=fs), engine='zarr') 
    print(f"opening {len(urls)} {model} urls, {len(coord_ds.time)} forecast times")
    # if parallel=True and nvars>19 on casper, RuntimeError: can't start new thread. 
    # casperexec run times with 50 vars
    # ncpus   runtime 
    #   1       3:00
    #   2       2:00
    #   3       1:45
    #   4       1:43
    #   5       1:40
    #   6       2:45
    #  10       6:00
    #  20     >10:00
    ds = xarray.open_mfdataset([s3fs.S3Map(url, s3=fs) for url in urls], engine='zarr', preprocess=rename_upscale, parallel=True)

    # Swap forecast_period with time coordinate so output files may be concatenated along forecast_reference_time and aligned along forecast_period.
    coord_ds = coord_ds.swap_dims({"time": "forecast_period"})
    ds = ds.rename({"time":"forecast_period"}).merge(coord_ds) # Rename "time" "forecast_period" and merge with coord_ds.
   
    if debug:
        da = ds["surface/CAPE"]
        fig, ax = plt.subplots(ncols=2)
        itime = 18
        vmin, vmax = da.load().quantile([0.02, 0.98])
        ax[0].imshow(da.isel(forecast_period=itime), vmin=vmin, vmax=vmax)
        ax[1].imshow(field_interp[itime], vmin=vmin, vmax=vmax)
        plt.show()
        #plt.savefig(f"{this_field}.png")
        plt.close(fig=fig)

    return ds

# =============Arguments===================
parser = argparse.ArgumentParser(description = "Read HRRR-ZARR, upscale, save", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("sdate", type=str, help='start date YYYYMMDDHH format')
parser.add_argument("--clobber", action="store_true", help='clobber existing output')
parser.add_argument("--debug", action="store_true", help='debug mode')
parser.add_argument("--npz", action="store_true", help='save compressed numpy file')
parser.add_argument("--parquet", action="store_true", help='save parquet file')
args = parser.parse_args()

clobber = args.clobber
debug   = args.debug
npz     = args.npz
parquet = args.parquet

sdate   = datetime.strptime(args.sdate, '%Y%m%d%H')
model   = 'HRRR-ZARR'
odir    = "/glade/work/" + os.getenv("USER") 
ofile   = f'{odir}/NSC_objects/HRRR/%s_{model}_upscaled.nc'%(sdate.strftime('%Y%m%d%H'))
if os.path.exists(ofile) and not clobber:
    print(ofile, "already exists. Exiting. Use --clobber option to override")
    sys.exit(1)


# get closest grid boxes
nngridpts = get_closest_gridbox()

upscaled_field_list = [
        ("0_1000m_above_ground", "VUCSH"),
        ("0_1000m_above_ground", "VVCSH"),
        #("0_3000m_above_ground", "CAPE"), # no 0_3000m_above_ground/CIN. CAPE truncated at 3000m AGL, or low-level CAPE?
        ("0_6000m_above_ground", "VUCSH"),
        ("0_6000m_above_ground", "VVCSH"),
        ("0C_isotherm", "HGT"),
        ("100_1000mb_above_ground", "MAXDVV_1hr_max_fcst"),
        ("100_1000mb_above_ground", "MAXUVV_1hr_max_fcst"),
        ("1000_0m_above_ground", "HLCY"),
        ("1000_0m_above_ground", "RELV_1hr_max_fcst"),
        ("1000m_above_ground", "MAXREF_1hr_max_fcst"),
        ("1000m_above_ground", "REFD"),
        ("10m_above_ground", "MAXUW_1hr_max_fcst"),
        ("10m_above_ground", "MAXVW_1hr_max_fcst"),
        ("10m_above_ground", "WIND_1hr_max_fcst"),
        #("180_0mb_above_ground", "CAPE"), # Associated with Best(4-layer) Lifted Index. not mucape or mlcape.
        #("180_0mb_above_ground", "CIN"),  # Associated with Best(4-layer) Lifted Index. not mucape or mlcape.
        ("255_0mb_above_ground", "CAPE"), # mucape
        ("255_0mb_above_ground", "CIN"), # mucin
        ("2m_above_ground", "DPT"),
        ("2m_above_ground", "SPFH"),
        ("2m_above_ground", "TMP"),
        ("3000_0m_above_ground", "HLCY"),
        ("3000_0m_above_ground", "MNUPHL_1hr_min_fcst"),
        ("3000_0m_above_ground", "MXUPHL_1hr_max_fcst"),
        ("4000m_above_ground", "REFD"),
        ("500mb", "DPT"),
        ("500mb", "HGT"),
        ("500mb", "TMP"),
        ("500mb", "UGRD"),
        ("500mb", "VGRD"),
        ("5000_2000m_above_ground", "MXUPHL_1hr_max_fcst"),
        ("5000_2000m_above_ground", "MNUPHL_1hr_min_fcst"),
        ("700mb", "DPT"),
        ("700mb", "HGT"),
        ("700mb", "TMP"),
        ("700mb", "UGRD"),
        ("700mb", "VGRD"),
        ("850mb", "DPT"),
        ("850mb", "TMP"),
        ("850mb", "UGRD"),
        ("850mb", "VGRD"),
        ("90_0mb_above_ground", "CAPE"), # mlcape
        ("90_0mb_above_ground", "CIN"), # mlcinh
        ("925mb", "DPT"),
        ("925mb", "TMP"),
        ("925mb", "UGRD"),
        ("925mb", "VGRD"),
        ("entire_atmosphere", "HAIL_1hr_max_fcst"),
        ("entire_atmosphere", "REFC"),
        ("entire_atmosphere_single_layer", "TCOLG_1hr_max_fcst"),
        ("level_of_adiabatic_condensation_from_sfc", "HGT"),
        ("surface", "APCP_1hr_acc_fcst"),
        ("surface", "CAPE"),
        ("surface", "CIN"),
        ("surface", "HAIL_1hr_max_fcst"),
        ("surface", "PRES"),
        ("surface", "PRATE")
        ]

fields_are_unique = len(set(upscaled_field_list)) == len(upscaled_field_list)
assert fields_are_unique, set([x for x in upscaled_field_list if upscaled_field_list.count(x) > 1])

upscaled_fields = upscale_forecast(upscaled_field_list,nngridpts,debug=debug)

derive_fields = len(upscaled_fields) > 10 # may have incomplete short list for debugging
if derive_fields:
    upscaled_fields = upscaled_fields.metpy.quantify() 
    print("Derive fields")
    upscaled_fields["0_1000m_above_ground/VSH"] = (upscaled_fields["0_1000m_above_ground/VUCSH"]**2 + upscaled_fields["0_1000m_above_ground/VVCSH"]**2)**0.5 * units["m/s"] # warned mesowest about VUCSH and VVCSH not having units 
    upscaled_fields["0_6000m_above_ground/VSH"] = (upscaled_fields["0_6000m_above_ground/VUCSH"]**2 + upscaled_fields["0_6000m_above_ground/VVCSH"]**2)**0.5 * units["m/s"] # warned mesowest about VUCSH and VVCSH not having units 
    # TODO: fix level_of_adiabatic_condensation_from_sfc/HGT. They are all NaN. I emailed James Powell at atmos-mesowest@lists.utah.edu
    print(upscaled_fields["level_of_adiabatic_condensation_from_sfc/HGT"].to_dataframe().describe())
    upscaled_fields["STP"] = mcalc.significant_tornado(upscaled_fields["surface/CAPE"], upscaled_fields["level_of_adiabatic_condensation_from_sfc/HGT"],
            upscaled_fields["1000_0m_above_ground/HLCY"], upscaled_fields["0_6000m_above_ground/VSH"])
    upscaled_fields["LR75"] = ( upscaled_fields["500mb/TMP"] - upscaled_fields["700mb/TMP"] ) / ( upscaled_fields["500mb/HGT"] - upscaled_fields["700mb/HGT"] )
    upscaled_fields["CAPESHEAR"] = upscaled_fields["0_6000m_above_ground/VSH"] * upscaled_fields["90_0mb_above_ground/CAPE"] #mlcape
    upscaled_fields = upscaled_fields.metpy.dequantify()




# Save upscaled data to file. There are tradeoffs to each format.
# .npz is fast and efficient but no names or labels.
# parquet has variable names and coordinate names and can write half-precision (2-byte) floats, like HRRR-ZARR, but no long_name or units.
# netCDF can't handle np.float16, or slash characters in variable names, but it handles np.float32 (twice the filesize of parquet).
# netCDF has named variables, coordinates, long_names and units.
# Use netcdf but strip masked pts for efficiency.

# Caveat: forecast_reference_time must remain 64-bit, so set aside before converting Dataset to np.float32.
forecast_reference_time = upscaled_fields["forecast_reference_time"]
upscaled_fields = upscaled_fields.astype(np.float32) # less disk space. HRRR-ZARR was even less precise, with np.float16, but to_netcdf() needs np.float32. 
upscaled_fields["forecast_reference_time"] = forecast_reference_time

# Stack x and y into 1-D pts coordinate
upscaled_fields = upscaled_fields.stack(pts=("projection_y_coordinate","projection_x_coordinate"))

# Drop masked pts before saving--reduces file size by 75%
mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk','rb'))
upscaled_fields.coords["mask"] = (("pts"), mask)
# reset multi-index level "pts" or NotImplementedError: isna is not defined for MultiIndex from .to_dataframe().to_parquet(). Also, to_netcdf() can't save MultiIndex.
upscaled_fields = upscaled_fields.where(upscaled_fields.mask, drop=True).reset_index("pts")

root, ext = os.path.splitext(ofile)
if parquet:
    ds = upscaled_fields.astype(np.float16)
    ds["forecast_reference_time"] = forecast_reference_time
    ds.to_dataframe().to_parquet(root+".par")
if npz:
    ds = upscaled_fields.astype(np.float16)
    ds["forecast_reference_time"] = forecast_reference_time
    np.savez_compressed(root+".npz", a=ds.to_dict())

# Change slash to hyphen for netcdf variable names
upscaled_fields = upscaled_fields.rename_vars({x:x.replace("/","-") for x in upscaled_fields.data_vars})
upscaled_fields.to_netcdf(ofile)
print("saved", f"{os.path.realpath(ofile)}")
