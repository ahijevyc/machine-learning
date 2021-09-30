#!/usr/bin/env python

from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
import pickle

def saveNewMap(domstr='CONUS', wrfout=None):
    fh = Dataset(wrfout, 'r')
    lats = fh.variables['XLAT'][0,:]
    lons = fh.variables['XLONG'][0,:]
    ll_lat, ll_lon, ur_lat, ur_lon = lats[0,0], lons[0,0], lats[-1,-1], lons[-1,-1]
    lat_1, lat_2, lon_0 = fh.TRUELAT1, fh.TRUELAT2, fh.STAND_LON
    fig_width = 1080
    fh.close()

    dpi = 90
    fig = plt.figure(dpi=dpi)
    ll_lat, ll_lon, ur_lat, ur_lon = [27,-96,37,-75] 
    m = Basemap(projection='lcc', resolution='i', llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat, \
                lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, area_thresh=1000)

    # compute height based on figure width, map aspect ratio, then add some vertical space for labels/colorbar
    fig_width  = fig_width/float(dpi)
    fig_height = fig_width*m.aspect + 0.93
    #fig_height = fig_width*m.aspect + 1.25
    figsize = (fig_width, fig_height)
    fig.set_size_inches(figsize)

    # place map 0.7" from bottom of figure, leave rest of 0.93" at top for title (needs to be in figure-relative coords)
    #x,y,w,h = 0.01, 0.8/float(fig_height), 0.98, 0.98*fig_width*m.aspect/float(fig_height) #too much padding at top
    x,y,w,h = 0.01, 0.7/float(fig_height), 0.98, 0.98*fig_width*m.aspect/float(fig_height)
    ax = fig.add_axes([x,y,w,h])
    #for i in ax.spines.itervalues(): i.set_linewidth(0.5)

    m.drawcoastlines(linewidth=0.5, ax=ax)
    m.drawstates(linewidth=0.25, ax=ax)
    m.drawcountries(ax=ax)
    #m.drawcounties(linewidth=0.1, color='gray', ax=ax)

    #pickle.dump((fig,ax,m), open('rt2015_ch_%s.pk'%domstr, 'wb'))
    pickle.dump((fig,ax,m), open('rt2015_ch_SE.pk', 'wb'))

saveNewMap('CONUS', wrfout='/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011042700/diags_d01_2011-04-28_00_00_00.nc')
