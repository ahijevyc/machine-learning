#!/usr/bin/env python

import numpy as np
from datetime import *
import time as t
import os, sys
from get_osr_gridded_by_day_hr import *
import pickle
import scipy.ndimage.filters
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
from matplotlib.colors import BoundaryNorm

def readSevereClimo(fname, day_of_year, hr):
    from scipy.interpolate import RectBivariateSpline
    data = np.load(fname)
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    grid81 = awips.makegrid(93, 65, returnxy=True)
    x, y = awips(data['lons'], data['lats'])
    
    spline = RectBivariateSpline(x[0,:], y[:,0], data['severe'][day_of_year-1,hr,:].T, kx=3, ky=3)
    interp_data = spline.ev(grid81[2].ravel(), grid81[3].ravel())
    return np.reshape(interp_data, (65,93))

def computeClimo():
    gmt2cst = timedelta(hours=6)
    
    m = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    grid81 = m.makegrid(93, 65, returnxy=True)

    mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
    mask = np.logical_not(mask)
    mask = mask.reshape((65,93))

    osr81_sum_by_year = []
    for year in range(1986,2016):
    #for year in range(2001,2016):
        #times in database are in CST, so if we want 00z-00z, subtract 6 hrs
        obs_start, obs_end = datetime(year,1,1,0,0,0) - gmt2cst, datetime(year,12,31,23,59,0) - gmt2cst
    
        if year < 2010:
            #osr81, osr81_count = get_osr_gridded(obs_start, obs_end, 93, 65, ['wind'])
            #osr81, osr81_count = get_osr_gridded_by_day(obs_start, obs_end, 93, 65, ['wind'])
            #osr81, osr81_count = get_osr_gridded_by_day_hr(obs_start, obs_end, 93, 65, ['wind','hailone','torn'])
            osr81, osr81_count = get_osr_gridded_by_day_hr(obs_start, obs_end, 93, 65, ['sighail'])
        else:
            #osr81, osr81_count = get_osr_gridded(obs_start, obs_end, 93, 65, ['wind'])
            #osr81, osr81_count = get_osr_gridded_by_day(obs_start, obs_end, 93, 65, ['wind'])
            #osr81, osr81_count = get_osr_gridded_by_day_hr(obs_start, obs_end, 93, 65, ['wind','hailone','torn'])
            osr81, osr81_count = get_osr_gridded_by_day_hr(obs_start, obs_end, 93, 65, ['sighail'])
    
        osr81[:,:,mask] = 0.0
        osr81_count[:,:,mask] = 0.0
        osr81_sum_by_year.append(osr81)

        print(year, osr81.sum(), osr81_count.sum())

    osr81_sum_by_year = np.array(osr81_sum_by_year)

    data = []
    for sig in [40, 120]: 
        #determine if report occurred within 2-hr and X-km of central grid pt
        if sig == 40:  osr81_sum_by_year = scipy.ndimage.filters.maximum_filter(osr81_sum_by_year, footprint=np.ones((1,1,5,1,1)), mode='wrap')
        if sig == 120: osr81_sum_by_year = scipy.ndimage.filters.maximum_filter(osr81_sum_by_year, footprint=np.ones((1,1,5,3,3)), mode='wrap')

        frequency = osr81_sum_by_year.mean(axis=0)
        frequency = scipy.ndimage.filters.gaussian_filter(frequency, sigma=[15,1.5,1.5,1.5], mode='wrap')
        print(frequency.shape)

        #for i in range(0,101,10): print(i, np.percentile(frequency, i))
        data.append( frequency )   

    ds = xr.Dataset(data_vars={
                                'climo': ( ['window', 'day', 'hr', 'y', 'x'], np.array(data).astype('float32') ),
                             },
                            coords={'window': [40, 120],
                                    'day': range(1,367),
                                    'hr': range(0,24),
                                    'lon': (('y', 'x'), grid81[0].astype('float32')),
                                    'lat': (('y', 'x'), grid81[1].astype('float32')),
                             },
                            attrs={ 'output time':datetime.utcnow().strftime('%Y-%m-%d %H:%m:%s UTC') },
                            )
    ds.to_netcdf('climo_severe_2hr_sighail.nc')
 
    #np.savez('climo_severe_120km_2hr_torn.npz', lats=grid81[1], lons=grid81[0], severe=frequency.astype('float32'))
    #np.savez('climo_severe_40km_2hr_torn.npz', lats=grid81[1], lons=grid81[0], severe=frequency.astype('float32'))

def plot_climo():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # setup color table
    levels = np.arange(0,0.05,0.005) 
    #levels = np.arange(0,0.1,0.01)
    cmap = plt.get_cmap('Reds')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(9,9))

    # old basemap plotting code
    m = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    grid81 = m.makegrid(93, 65, returnxy=True)
    xorig, yorig = m(grid81[0], grid81[1])
    x = (xorig[1:,1:] + xorig[:-1,:-1])/2.0
    y = (yorig[1:,1:] + yorig[:-1,:-1])/2.0 
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    #a = m.pcolormesh(x, y, readSevereClimo('../severe.npz', 121)[1:,1:], cmap=cmap, norm=norm)
    #a = plt.pcolormesh(grid81[0], grid81[1],readSevereClimo('climo_torn_15yr.npz', 121)[1:,1:], cmap=cmap, norm=norm, transform=ccrs.LambertConformal())
   
    # cartopy code 
    #m = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    #grid81 = m.makegrid(93, 65, returnxy=True)

    #ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=38.33643, central_longitude=-97.53348, standard_parallels=(32,46)))
    #ax.set_extent([-122, -70, 24, 50], ccrs.PlateCarree()) 
    
    #states = cfeature.NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_lakes_shp')
    #ax.add_feature(states, linewidth=0.25, color='gray')
    #ax.coastlines('50m', linewidth=0.25, color='gray')
    #a = plt.pcolormesh(lons, lats, np.ma.masked_less(climo_to_plot[1:,1:], 0.0025), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1000)

    lons = (grid81[0][1:,1:] + grid81[0][:-1,:-1])/2.0
    lats = (grid81[1][1:,1:] + grid81[1][:-1,:-1])/2.0 
    a = m.pcolormesh(x, y, np.ma.masked_less(climo_to_plot[1:,1:], 0.0025), cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(a, shrink=0.95, pad=0, orientation='horizontal')
    plt.savefig('test%02d.png'%f, dpi=200, bbox_inches='tight')


computeClimo()
 
#data = np.load('climo_severe_40km_2hr.npz')
#climo = data['severe']
#for f in range(0,24):
#    climo_to_plot = climo[:,f,:].mean(axis=0)
#    #climo_to_plot = readSevereClimo('climo_severe_120km_2hr.npz', 181, f)
#    print(climo_to_plot.max(), climo_to_plot.min(), climo_to_plot.shape)                                                          
#    plot_climo()
