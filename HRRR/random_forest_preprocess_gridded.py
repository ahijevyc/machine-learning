#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys, os, math
from datetime import *
import sqlite3, cartopy, pickle
from mpl_toolkits.basemap import *
from matplotlib.path import Path
#from get_osr_gridded_new import *
from cartopy.geodesic import Geodesic
import scipy.ndimage.filters

def get_closest_report_distances(grid_lats, grid_lons, grid_times): 
    # read storm reports from database
    # get observations before forecast hour 0 and after forecast hour 36 for longer verification windows
    sdate, edate = thisdate-timedelta(hours=4) - gmt2cst, thisdate+timedelta(hours=nfhr+4) - gmt2cst
  
    conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_v20200626.db')
    c = conn.cursor()

    for type in report_types:
        if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 1.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))
        rpts = c.fetchall()
        #print len(rpts), type, 'reports'

        if len(rpts) > 0:
            report_lats, report_lons, report_times = list(zip(*rpts))
            report_times = [ int((datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - thisdate).total_seconds()/3600.0 - 0.000001) + 6 + 1 for t in report_times ] #convert to UTC, then add one so 00z-01z reports are compared with 1z forecast

        for time_tolerance in [0,1,2]:
            # loop over each storm and find the reports within time and distance tolerances
            all_distances = []
            for i in range(len(grid_lats)):
                #print 'gpt %d/%d'%(i+1,len(grid_lats))
                if len(rpts) > 0:
                    #find all reports w/in 1 hour of this grid point
                    # should second <= be a < so not overlapping? probably doesnt change much...
                    report_mask = ( report_times >= grid_times[i]-time_tolerance ) & ( report_times <= grid_times[i]+time_tolerance ) #add 1 here so obs between 12-13Z are matched with proper storms? 
                    report_mask = np.array(report_mask)
                    these_report_lons, these_report_lats = np.array(report_lons)[report_mask], np.array(report_lats)[report_mask]

                    reports = list(zip(these_report_lons, these_report_lats))
                    pts  = (grid_lons[i], grid_lats[i])

                    # see if any remain after filtering, if so compute distances from storm centroid
                    if len(reports) > 0:
                        t = geo.inverse( pts , reports )
                        t = np.asarray(t)

                        distances_meters = t[:,0]/1000.0
                        closest_report_distance = np.amin(distances_meters)
                    else:
                        closest_report_distance = -1
                else:
                    closest_report_distance = -1

                all_distances.append(int(closest_report_distance))

            df['%s_rptdist_%dhr'%(type,time_tolerance)] = all_distances

def maximum_filter_ignore_nan(data, footprint):
    nans = np.isnan(data)
    replaced = np.where(nans, -np.inf, data) #need to ignore nans, this *should* do it
    return scipy.ndimage.filters.maximum_filter(replaced, footprint=footprint)

gmt2cst = timedelta(hours=6)
report_types = ['hailone', 'wind', 'torn', 'sighail', 'sigwind']

startdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
enddate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
#startdate = datetime(2010,1,1,0,0,0)
#enddate = datetime(2017,12,31,0,0,0)
geo = Geodesic()
thisdate = startdate
forecasts_processed = 0
model = 'NSC3km-12sec'
model = 'HRRR'
#model = 'HRRRX'
#model = 'NSC1km'
#mem = int(sys.argv[3])
#time_tolerance = 2

# determine forecast length for v3/v4
if startdate >= datetime(2020,12,2,12,0,0) or model == 'HRRRX':
    if startdate.hour in [0,6,12,18]: nfhr = 48
    else: nfhr = 18
else:
    if startdate.hour in [0,6,12,18]: nfhr = 36
    else: nfhr = 18

# make sure these are 1s in the masked area if we want to pull out these values
#mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask  = pickle.load(open('/glade/work/sobash/NSC_objects/HRRR/usamask_mod.pk', 'rb'))
#mask = np.logical_not(mask)
mask = mask.reshape((65,93)) 

awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
lons, lats = awips.makegrid(93, 65)
lons = np.repeat(lons[np.newaxis,:], nfhr, axis=0)
lats = np.repeat(lats[np.newaxis,:], nfhr, axis=0)
x_ind, y_ind = np.indices((65,93))
x_ind = np.repeat(x_ind[np.newaxis,:], nfhr, axis=0)
y_ind = np.repeat(y_ind[np.newaxis,:], nfhr, axis=0)

while thisdate <= enddate:
    yyyymmdd = thisdate.strftime('%Y%m%d')
    yyyymmddhh = thisdate.strftime('%Y%m%d%H')
    #thisdate = datetime.strptime(sys.argv[1], '%Y%m%d')
    print(thisdate)
  
    if model == 'GEFS': fname = '/glade/work/sobash/NSC/%s_GEFS_mem%d_upscaled.npz'%(thisdate.strftime('%Y%m%d00'), mem)
    else: fname = '/glade/work/sobash/NSC/%s_%s_upscaled.npz'%(yyyymmddhh, model)

    if os.path.exists(fname):
        if model == 'GEFS': data = np.load('/glade/work/sobash/NSC/%s_%s_mem%d_upscaled.npz'%(thisdate.strftime('%Y%m%d00'), model,mem))
        else: data = np.load('/glade/work/sobash/NSC/%s_%s_upscaled.npz'%(yyyymmddhh, model), allow_pickle=True)
        upscaled_fields = data['a'].item() #have to use item since dictionary was stored 
   
        # convert to numpy arrays
        for k in upscaled_fields:
            upscaled_fields[k] = np.array(upscaled_fields[k])

        #instead of neighbor approach - lets just do a maximum within X km for surrogates, and grid point values for environmental fields
        #max_within_1box, max_within_2box, max_within_1hr, max_within_2hr, max_within_3hr
        simple_max_fields = ['UP_HELI_MAX', 'UP_HELI_MAX03', 'RVORT1', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'PREC_ACC_NC']
        simple_mean_fields = ['STP', 'CAPESHEAR', 'MUCAPE', 'SBCAPE', 'SBCINH', 'SBLCL', 'SHR06', 'SHR01', 'SRH03', 'SRH01', 'T2', 'TD2', 'PSFC']
        # simple fields
        for k in simple_max_fields:
            for x in [3,5]:
                for t in [1,3,5]:
                    #upscaled_fields[k+'-N%dT%d'%(x,t)] = scipy.ndimage.filters.maximum_filter(upscaled_fields[k], footprint=np.ones((t,x,x)))
                    upscaled_fields[k+'-N%dT%d'%(x,t)] = maximum_filter_ignore_nan(upscaled_fields[k], np.ones((t,x,x)))

        # simple fields
        for k in simple_mean_fields:
            for x in [3,5]:
                for t in [1,3,5]:
                    # modified for the possibility of a missing forecast hour -- need to convolve and ignore nans in time
                    # this produces files that are slightly larger than old way, not sure why
                    nanmask = np.isnan(upscaled_fields[k])
                    kernel = np.ones((t,x,x))
                    denom = scipy.ndimage.filters.convolve(np.logical_not(nanmask).astype(np.float32), kernel)
                    #upscaled_fields[k+'-N%dT%d'%(x,t)] = scipy.ndimage.filters.convolve(upscaled_fields[k], weights=kernel/float(kernel.size))
                    upscaled_fields[k+'-N%dT%d'%(x,t)] = scipy.ndimage.filters.convolve(np.where(nanmask, 0, upscaled_fields[k]), kernel) / denom


        #maximum_filter uses reflect along edges - important for forecast hour 36
        upscaled_fields['UP_HELI_MAX-N1T5'] = maximum_filter_ignore_nan(upscaled_fields['UP_HELI_MAX'], np.ones((5,1,1))) #max 4-hr UH within 40km
        upscaled_fields['UP_HELI_MAX80-N1T5'] = maximum_filter_ignore_nan(upscaled_fields['UP_HELI_MAX80'], np.ones((5,1,1))) #max 4-hr UH within 80km
        upscaled_fields['UP_HELI_MAX120-N1T5'] = maximum_filter_ignore_nan(upscaled_fields['UP_HELI_MAX120'], np.ones((5,1,1))) #max 4-hr UH within 120km
        
        print('masking')
        # only use grid points within mask, and remove first forecast hour (to match OSRs)
        for k in upscaled_fields:
            upscaled_fields[k] = upscaled_fields[k][1:,mask]

        print('obs')
        # read storm reports from database
        #sdate, edate = thisdate+timedelta(hours=0) - gmt2cst, thisdate+timedelta(hours=36) - gmt2cst 
        #osr81 = get_osr_gridded(sdate, edate, 93, 65, report_types)

        #upscaled_fields['OSR'] = osr81[:,mask]
        upscaled_fields['xind'] = x_ind[:,mask]
        upscaled_fields['yind'] = y_ind[:,mask]
        upscaled_fields['lat'] = lats[:,mask]
        upscaled_fields['lon'] = lons[:,mask]
 
        # create pandas dataframe for all fields
        names = ['fhr', 'pt']
        index = pd.MultiIndex.from_product([range(1,s+1) for s in upscaled_fields['UP_HELI_MAX'].shape], names=names)
        for k in upscaled_fields.keys():
            upscaled_fields[k] = np.array(upscaled_fields[k]).flatten()
        
        df = pd.DataFrame(upscaled_fields, index=index)
        df = df.reset_index(level=['fhr'])
        df['Date'] = thisdate.strftime('%Y-%m-%d %H:%M:%S')
        
        get_closest_report_distances(df['lat'].values, df['lon'].values, df['fhr'].values)

        #for c in df:
        #    print(c, df.iloc[1297*4][c])
        # get rid of rows where UH is NaN (indicating missing forecast hour) 
        df.dropna(subset=['UP_HELI_MAX'], inplace=True) 
        
        #df.to_csv('./grid_data_ncarstorm_3km_csv_preprocessed/grid_data_NCARSTORM_d01_%s-0000.csv.gz'%(yyyymmdd), float_format='%.2f', index=False, compression='gzip')
        #if model == 'GEFS': df.to_csv('grid_data_GEFS_mem%d_d01_%s-0000.csv'%(mem,yyyymmdd), float_format='%.2f', index=False)
        #elif model == 'HRRR': df.to_csv('grid_data_HRRR_d01_%s-0000.csv'%(yyyymmddhh), float_format='%.2f', index=False)
        #else: df.to_csv('grid_data_NCARSTORM_d01_%s-0000.csv'%(yyyymmdd), float_format='%.2f', index=False)
   
        df.to_parquet('./grid_data/grid_data_%s_d01_%s-0000.par'%(model,yyyymmddhh))
 
        thisdate += timedelta(days=1)
        forecasts_processed += 1    
        print('forecasts processed', forecasts_processed)
  
    else:
        thisdate += timedelta(days=1)
    
