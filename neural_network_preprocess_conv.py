#!/usr/bin/env python

import pandas as pd
import numpy as np
import time, sys, os
from datetime import *
import sqlite3, cartopy, pickle
from mpl_toolkits.basemap import *
from matplotlib.path import Path
from get_osr_gridded_new import *
from cartopy.geodesic import Geodesic
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.ndimage.filters

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    import os
    rgb, appending = [], False
    rgb_dir_ys = '/glade/apps/opt/ncl/6.2.0/intel/12.1.5/lib/ncarg/colormaps'
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    if os.path.isdir(rgb_dir_ys): fh = open('%s/%s.rgb'%(rgb_dir_ys,name), 'r')
    else: fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def writeOutput():
    outnc = Dataset('/glade/work/sobash/NSC/gridded_windows_conv_%s.nc'%thisdate.strftime('%Y%m%d%H'), 'w')
    
    outnc.history = 'Created '+ time.ctime(time.time())

    outnc.createDimension('fhrs', 36)
    outnc.createDimension('windows', total_points)
    outnc.createDimension('window_size', window_size)

    for k in upscaled_fields_conv.keys():
        var = outnc.createVariable(k, 'f4', ('fhrs','windows','window_size','window_size'), zlib=True)
        var[:] = upscaled_fields_conv[k][:]

    for r in report_types:
        var = outnc.createVariable('reportdist%s'%r, 'i4', ('fhrs','windows'), zlib=True)
        var[:] = all_dist[r]

    outnc.close()

def get_closest_report_distances(grid_lats, grid_lons, grid_times): 
    # read storm reports from database
    sdate, edate = thisdate+timedelta(hours=0) - gmt2cst, thisdate+timedelta(hours=36) - gmt2cst
  
    conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_all.db')
    c = conn.cursor()

    dists = {}
    for type in report_types:
        if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
        elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 1.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
        elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))
        rpts = c.fetchall()
        #print len(rpts), type, 'reports'

        if len(rpts) > 0:
            report_lats, report_lons, report_times = zip(*rpts)
            report_times = [ int((datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - thisdate).total_seconds()/3600.0 - 0.000001) + 6 + 1 for t in report_times ] #convert to UTC, then add one so 00z-01z reports are compared with 1z forecast

        # loop over each storm and find the reports within time and distance tolerances
        all_distances = []
        for i in range(len(grid_lats)):
            #print 'gpt %d/%d'%(i+1,len(grid_lats))
            if len(rpts) > 0:
                #find all reports w/in 1 hour of this grid point
                report_mask = ( report_times >= grid_times[i]-time_tolerance ) & ( report_times <= grid_times[i]+time_tolerance ) #add 1 here so obs between 12-13Z are matched with proper storms? 
                report_mask = np.array(report_mask)
                these_report_lons, these_report_lats = np.array(report_lons)[report_mask], np.array(report_lats)[report_mask]

                reports = zip(these_report_lons, these_report_lats)
                pts  = (grid_lons[i], grid_lats[i])

                # see if any remain after filtering, if so compute distances from storm centroid
                if len(reports) > 0:
                    t = geo.inverse( pts , reports )
                    t = np.asarray(t)

                    distances_meters = t[:,0]
                    closest_report_distance = np.amin(distances_meters)
                else:
                    closest_report_distance = -9999
            else:
                closest_report_distance = -9999

            all_distances.append(int(closest_report_distance))

        dists[type] = all_distances
        #df['%s_report_closest_distance'%type] = all_distances

    return dists

def plotfield(plot_field):
    #plot_field = np.array(upscaled_fields['MUCAPE'])
    #print plot_field.shape
    #plot_field = np.amax(plot_field, axis=0)

    print plot_field.shape, plot_field.max(), plot_field.min()

    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    grid81 = awips.makegrid(93, 65, returnxy=True)
    x81, y81 = awips(grid81[0], grid81[1])

    levels = np.arange(270,310,5)
    #levels = np.arange(0,100,2.5)
    test = readNCLcm('MPL_Reds')[10:]
    cmap = colors.ListedColormap(test)
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    ##awips.pcolormesh(x81, y81, np.ma.masked_less(u_interp, 100.0), cmap=cmap, norm=norm)
    awips.pcolormesh(x81, y81, plot_field, cmap=cmap, norm=norm)
    #awips.pcolormesh(x81, y81, env['b'], cmap=cmap, norm=norm)
    awips.drawstates()
    awips.drawcountries()
    awips.drawcoastlines()
    plt.savefig('test.png')

gmt2cst = timedelta(hours=6)
report_types = ['hailone', 'wind', 'torn']

startdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
enddate = datetime.strptime(sys.argv[2], '%Y%m%d%H')
#startdate = datetime(2010,1,1,0,0,0)
#enddate = datetime(2017,12,31,0,0,0)
geo = Geodesic()
thisdate = startdate
forecasts_processed = 0
model = 'NSC3km-12sec'
time_tolerance = 2

# make sure these are 1s in the masked area if we want to pull out these values
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'r'))
#mask = np.logical_not(mask)
mask = mask.reshape((65,93)) 
print mask.sum()

awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
lons, lats = awips.makegrid(93, 65)
lons = np.repeat(lons[np.newaxis,:], 37, axis=0)
lats = np.repeat(lats[np.newaxis,:], 37, axis=0) #first index will be removed below

while thisdate <= enddate:
    yyyymmdd = thisdate.strftime('%Y%m%d')
    #thisdate = datetime.strptime(sys.argv[1], '%Y%m%d')
    print thisdate
  
    fname = '/glade/work/sobash/NSC/%s_%s_upscaled.npz'%(thisdate.strftime('%Y%m%d00'), model)

    if os.path.exists(fname):
        data = np.load('/glade/work/sobash/NSC/%s_%s_upscaled.npz'%(thisdate.strftime('%Y%m%d00'), model))
        upscaled_fields = data['a'].item() #have to use item since dictionary was stored
        
        # add lat/lon
        upscaled_fields['lat'] = lats[:,:]
        upscaled_fields['lon'] = lons[:,:]
        upscaled_fields['fhr'] = []
        upscaled_fields['doy'] = []
 
        # add storm reports 
        sdate, edate = thisdate+timedelta(hours=0) - gmt2cst, thisdate+timedelta(hours=36) - gmt2cst 
        osr81 = get_osr_gridded(sdate, edate, 93, 65, report_types)
        upscaled_fields['OSR'] = osr81[:,:]
        print upscaled_fields['OSR'].shape

        # window settings 
        stride_length, window_size = 5, 5 
        stride_index_x = range(0,93,stride_length)
        stride_index_y = range(0,65,stride_length)
        #total_points = len(stride_index_x) * len(stride_index_y)

        # use only 80km grid points where center point falls within US boundaries
        total_points = 0
        for n,i in enumerate(stride_index_x):
            for m,j in enumerate(stride_index_y):
                if mask[j,i]: total_points += 1
        #plotfield(np.array(upscaled_fields['T850'])[18,:])

        upscaled_fields_conv = {}
        for k in upscaled_fields:
            # extract out forecast hour 1 and beyond
            if k not in ['OSR','fhr','doy']: upscaled_fields[k] = np.array(upscaled_fields[k])[1:,:]

            print 'processing %s'%k
            upscaled_fields_conv[k] = np.zeros((36,total_points,window_size,window_size))
            # create array with "images" to put into convnet
            for s,t in enumerate(range(1,37)):
                if k == 'fhr':
                    upscaled_fields_conv['fhr'][s,:] = t
                    continue
                
                if k == 'doy':
                    upscaled_fields_conv['doy'][s,:] = thisdate.timetuple().tm_yday
                    continue

                win_idx = 0
                for n,i in enumerate(stride_index_x):
                    for m,j in enumerate(stride_index_y):
                        if not mask[j,i]: continue
                        if j+stride_length > 65 or i+stride_length > 93: continue
                        upscaled_fields_conv[k][s,win_idx,:] = upscaled_fields[k][s,j:j+window_size,i:i+window_size]
                        win_idx += 1
      
        # get closest report from center point of windows
        cidx = window_size/2
        all_dist = get_closest_report_distances(upscaled_fields_conv['lat'][:,:,cidx,cidx].flatten(), upscaled_fields_conv['lon'][:,:,cidx,cidx].flatten(), upscaled_fields_conv['fhr'][:,:,cidx,cidx].flatten())

        writeOutput()

        thisdate += timedelta(days=1)
        forecasts_processed += 1    
        print 'forecasts processed', forecasts_processed
  
    else:
        thisdate += timedelta(days=1)
    
