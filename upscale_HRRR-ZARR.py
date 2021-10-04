#!/usr/bin/env python

import pdb
from netCDF4 import Dataset
import numpy as np
from datetime import *
import time, os, sys
import scipy.ndimage as ndimage
import os
import pickle as pickle
import multiprocessing
from mpl_toolkits.basemap import *
from scipy import spatial
import s3fs
import scipy.ndimage.filters
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray


### THIS CODE EVOLVED FROM CODE WITHIN /glade/u/home/sobash/NSC_scripts
### TO UPSCALE 3-KM CAM DATA TO AN 80-KM GRID


def get_closest_gridbox():
    ### find closest 3-km or 1-km grid point to each 80-km grid point
    gpfname = f'{odir}/NSC_objects/nngridpts_80km_{model}.pk'
    if os.path.exists(gpfname):
        nngridpts = pickle.load(open(gpfname, 'rb'), encoding='bytes')
    else:
        # INTERPOLATE NARR TO 80KM GRID
        #fig, axes, m  = pickle.load(open('/glade/u/home/wrfrt/rt_ensemble_2018wwe/python_scripts/rt2015_CONUS.pk', 'r'))
        awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
        grid81 = awips.makegrid(93, 65, returnxy=True)
        x81, y81 = awips(grid81[0], grid81[1])
        #x81 = (x[1:,1:] + x[:-1,:-1])/2.0
        #y81 = (y[1:,1:] + y[:-1,:-1])/2.0

        if model.startswith('HRRR'):
            f = Dataset('/glade/work/ahijevyc/share/HRRR.nc')
            lats = f.variables['gridlat_0'][:]
            lons = f.variables['gridlon_0'][:]
            f.close()
        else:
            if model == 'NSC1km': f = Dataset('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
            if model == 'NSC3km-12sec': f = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
            if model == 'GEFS': f = Dataset('/glade/scratch/sobash/ncar_ens/gefs_ics/2017042500/wrf_rundir/ens_1/diags_d02.2017-04-25_00:00:00.nc', 'r')
            if model == 'RT2020': f = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
            lats = f.variables['XLAT'][0,:]
            lons = f.variables['XLONG'][0,:]
            f.close()
        xy = awips(lons.ravel(), lats.ravel())
        tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
        nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
        pickle.dump(nngridpts, open(gpfname, 'wb'))

    return nngridpts

def upscale_forecast(upscaled_field_list):
    if False: # coordinates not needed now
        level, variable = upscaled_field_list[0]
        # first url without the last level subdirectory has time and projection_x_coordinate and forecast_period and forecast_reference_time
        urls = [os.path.join('s3://hrrrzarr/sfc', tdate.strftime("%Y%m%d/%Y%m%d_%Hz_fcst.zarr"), level, variable)]
    fs = s3fs.S3FileSystem(anon=True)
    urls=[]
    for (level, variable) in upscaled_field_list:
        urls.append( os.path.join('s3://hrrrzarr/sfc', tdate.strftime("%Y%m%d/%Y%m%d_%Hz_fcst.zarr"), level, variable, level) )
    ds = xarray.open_mfdataset([s3fs.S3Map(url, s3=fs) for url in urls], engine='zarr') # Tried parallel=True, but not significantly faster
    print(ds)
    upscaled_fields = {}
    for this_field, da in ds.data_vars.items():
        da = da.astype(float) # avoid *** RuntimeError: array type dtype('float16') not supported from scipy.ndimage
        # use maximum for certain fields, mean for others 
        if this_field in ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'HAILCAST_DIAM_MAX']:
            field = scipy.ndimage.filters.maximum_filter(da, size=(1,27,27), mode='nearest')
        else:
            field = scipy.ndimage.filters.uniform_filter(da, size=(1,27,27), mode='nearest')
        field_interp = np.empty((ds.time.size,65,93))
        for t,_ in enumerate(field):
            field_interp[t] = field[t].flatten()[nngridpts[1]].reshape((65,93))
        upscaled_fields[this_field] = field_interp
    ds.close()

    return upscaled_fields

sdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
tdate = sdate
model = sys.argv[2]

odir = "/glade/work/" + os.getenv("USER") 

upscaled_fields = { 'UP_HELI_MAX':[], 'UP_HELI_MAX03':[], 'UP_HELI_MAX01':[], 'W_UP_MAX':[], 'W_DN_MAX':[], 'WSPD10MAX':[], 'STP':[], 'LR75':[], 'CAPESHEAR':[],
           'MUCAPE':[], 'SBCAPE':[], 'SBCINH':[], 'MLCINH':[], 'MLLCL':[], 'SHR06': [], 'SHR01':[], 'SRH01':[], 'SRH03':[], 'T2':[], 'TD2':[], 'PSFC':[], 'PREC_ACC_NC':[], \
           'HAILCAST_DIAM_MAX':[], \
           'T925':[], 'T850':[], 'T700':[], 'T500':[], 'TD925':[], 'TD850':[], 'TD700':[], 'TD500':[], 'U925':[], 'U850':[], 'U700':[], 'U500':[], 'V925':[], 'V850':[], 'V700':[], 'V500':[], \
           'UP_HELI_MAX80':[], 'UP_HELI_MAX120':[], 'UP_HELI_MAX01-120':[] }
press_levels = [1000,925,850,700,600,500,400,300,250,200,150,100]

upscaled_fields = { 'UP_HELI_MAX01-120':[] }

# get closest grid boxes
print('finding closest grid points')
nngridpts = get_closest_gridbox()

upscaled_fields = [
        ("surface","HAIL_1hr_max_fcst"),
        ("surface", "PRATE"),
        ("3000_0m_above_ground", "HLCY"),
        ("5000_2000m_above_ground", "MXUPHL_1hr_max_fcst"),
        ("5000_2000m_above_ground", "MNUPHL_1hr_min_fcst"),
        ]

upscaled_fields = upscale_forecast(upscaled_fields)

ofile = f'{odir}/NSC/%s_%s_upscaled'%(sdate.strftime('%Y%m%d%H'),model)
np.savez_compressed(ofile, a=upscaled_fields)
print("saved", f"{os.path.realpath(ofile)}.npz")
