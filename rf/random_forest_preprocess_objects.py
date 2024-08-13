#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys, os
from datetime import *
import sqlite3
import cartopy
from cartopy.geodesic import Geodesic
from mpl_toolkits.basemap import *
from matplotlib.path import Path

def usa_filter(df):
    # convert storm lat lons to map proj coords
    lats, lons = df['Centroid_Lat'].values, df['Centroid_Lon'].values
    x, y = awips_proj(lons, lats)
    storm_proj = np.array((x.flatten(), y.flatten())).T

    mask = usaPath.contains_points(storm_proj)
  
    return df[mask]

gmt2cst = timedelta(hours=6)
report_types = ['hail', 'wind', 'torn']

startdate = datetime(2010,1,1,0,0,0)
enddate = datetime(2017,12,31,0,0,0)
#startdate = datetime(2016,11,30,0,0,0)
#enddate = datetime(2017,12,31,0,0,0)
time_tolerance = 2
geo = Geodesic()

usecols = 'Step_ID,Track_ID,Ensemble_Name,Ensemble_Member,Run_Date,Valid_Date,Forecast_Hour,Valid_Hour_UTC,Duration,Centroid_Lon,Centroid_Lat,Centroid_X,Centroid_Y,Storm_Motion_U,Storm_Motion_V,UP_HELI_MAX_mean,UP_HELI_MAX_max,UP_HELI_MAX_min,GRPL_MAX_mean,GRPL_MAX_max,GRPL_MAX_min,WSPD10MAX_mean,WSPD10MAX_max,WSPD10MAX_min,W_UP_MAX_mean,W_UP_MAX_max,W_UP_MAX_min,W_DN_MAX_mean,W_DN_MAX_max,W_DN_MAX_min,RVORT1_MAX_mean,RVORT1_MAX_max,RVORT1_MAX_min,RVORT5_MAX_mean,RVORT5_MAX_max,RVORT5_MAX_min,UP_HELI_MAX03_mean,UP_HELI_MAX03_max,UP_HELI_MAX03_min,UP_HELI_MAX01_mean,UP_HELI_MAX01_max,UP_HELI_MAX01_min,UP_HELI_MIN_mean,UP_HELI_MIN_max,UP_HELI_MIN_min,REFL_COM_mean,REFL_COM_max,REFL_COM_min,REFL_1KM_AGL_mean,REFL_1KM_AGL_max,REFL_1KM_AGL_min,REFD_MAX_mean,REFD_MAX_max,REFD_MAX_min,PSFC_mean,PSFC_max,PSFC_min,T2_mean,T2_max,T2_min,Q2_mean,Q2_max,Q2_min,TD2_mean,TD2_max,TD2_min,U10_mean,U10_max,U10_min,V10_mean,V10_max,V10_min,SBLCL-potential_mean,SBLCL-potential_max,SBLCL-potential_min,MLLCL-potential_mean,MLLCL-potential_max,MLLCL-potential_min,SBCAPE-potential_mean,SBCAPE-potential_max,SBCAPE-potential_min,MLCAPE-potential_mean,MLCAPE-potential_max,MLCAPE-potential_min,MUCAPE-potential_mean,MUCAPE-potential_max,MUCAPE-potential_min,SBCINH-potential_mean,SBCINH-potential_max,SBCINH-potential_min,MLCINH-potential_mean,MLCINH-potential_max,MLCINH-potential_min,USHR1-potential_mean,USHR1-potential_max,USHR1-potential_min,VSHR1-potential_mean,VSHR1-potential_max,VSHR1-potential_min,USHR6-potential_mean,USHR6-potential_max,USHR6-potential_min,VSHR6-potential_mean,VSHR6-potential_max,VSHR6-potential_min,U_BUNK-potential_mean,U_BUNK-potential_max,U_BUNK-potential_min,V_BUNK-potential_mean,V_BUNK-potential_max,V_BUNK-potential_min,SRH03-potential_mean,SRH03-potential_max,SRH03-potential_min,SRH01-potential_mean,SRH01-potential_max,SRH01-potential_min,PSFC-potential_mean,PSFC-potential_max,PSFC-potential_min,T2-potential_mean,T2-potential_max,T2-potential_min,Q2-potential_mean,Q2-potential_max,Q2-potential_min,TD2-potential_mean,TD2-potential_max,TD2-potential_min,U10-potential_mean,U10-potential_max,U10-potential_min,V10-potential_mean,V10-potential_max,V10-potential_min,area,eccentricity,major_axis_length,minor_axis_length,orientation'

# for NCAR ensemble
#usecols = 'Step_ID,Track_ID,Date,Forecast_Hour,Valid_Hour_UTC,Duration,Centroid_Lon,Centroid_Lat,Storm_Motion_U,Storm_Motion_V,LTG3_MAX_mean,LTG3_MAX_max,LTG3_MAX_min,PWAT-potential_mean,PWAT-potential_max,PWAT-potential_min,HAIL_MAX2D_mean,HAIL_MAX2D_max,HAIL_MAX2D_min,HAIL_MAXK1_mean,HAIL_MAXK1_max,HAIL_MAXK1_min,UP_HELI_MAX_mean,UP_HELI_MAX_max,UP_HELI_MAX_min,GRPL_MAX_mean,GRPL_MAX_max,GRPL_MAX_min,WSPD10MAX_mean,WSPD10MAX_max,WSPD10MAX_min,W_UP_MAX_mean,W_UP_MAX_max,W_UP_MAX_min,W_DN_MAX_mean,W_DN_MAX_max,W_DN_MAX_min,RVORT1_MAX_mean,RVORT1_MAX_max,RVORT1_MAX_min,UP_HELI_MAX03_mean,UP_HELI_MAX03_max,UP_HELI_MAX03_min,UP_HELI_MIN_mean,UP_HELI_MIN_max,UP_HELI_MIN_min,REFD_MAX_mean,REFD_MAX_max,REFD_MAX_min,LCL_HEIGHT-potential_mean,LCL_HEIGHT-potential_max,LCL_HEIGHT-potential_min,CAPE_SFC-potential_mean,CAPE_SFC-potential_max,CAPE_SFC-potential_min,MUCAPE-potential_mean,MUCAPE-potential_max,MUCAPE-potential_min,CIN_SFC-potential_mean,CIN_SFC-potential_max,CIN_SFC-potential_min,UBSHR1-potential_mean,UBSHR1-potential_max,UBSHR1-potential_min,VBSHR1-potential_mean,VBSHR1-potential_max,VBSHR1-potential_min,UBSHR6-potential_mean,UBSHR6-potential_max,UBSHR6-potential_min,VBSHR6-potential_mean,VBSHR6-potential_max,VBSHR6-potential_min,SRH3-potential_mean,SRH3-potential_max,SRH3-potential_min,area,eccentricity,major_axis_length,minor_axis_length,orientation'

thisdate = startdate
forecasts_processed = 0
remove_ocean_storms = True

if remove_ocean_storms:
    # READ IN US LAT/LON FILE
    fh = open('/glade/u/home/sobash/2013RT/uspoints', 'r')
    pts = fh.readlines()[0].split(',0')
    pts2 = [a.split(',') for a in pts]
    pts3 = np.array(pts2[:-1], dtype=float)
    print pts3.shape

    # AWIPS 80-KM GRID POINTS
    awips_proj = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95)

    # CONVERT US LAT LONS TO MAP PROJ COORDS 
    us_path_proj = awips_proj(pts3[:,0], pts3[:,1])
    usaPath = Path(np.array(us_path_proj).T)

while thisdate <= enddate:
  yyyymmdd = thisdate.strftime('%Y%m%d')
  #thisdate = datetime.strptime(sys.argv[1], '%Y%m%d')
  
  fname = './track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
  #fname = './track_data_ncar_2016_csv/track_step_NCAR_mem1_%s.csv'%(yyyymmdd)

  conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_all.db')
  c = conn.cursor()

  if os.path.exists(fname):
    print 'thinning', fname
    df = pd.read_csv(fname, usecols=usecols.split(','))

    if remove_ocean_storms: df = usa_filter(df)

    storm_lats = df['Centroid_Lat'].values
    storm_lons = df['Centroid_Lon'].values
    storm_times = df['Forecast_Hour'].values    
    print 'Number of storms', len(storm_lats)

    # read storm reports from database
    sdate, edate = thisdate+timedelta(hours=12) - gmt2cst, thisdate+timedelta(hours=36) - gmt2cst
    #rpts = []
    
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
        #rpts.extend(c.fetchall())
        rpts = c.fetchall()

        print len(rpts), 'reports'
           
        if len(rpts) > 0:
            report_lats, report_lons, report_times = zip(*rpts)
            report_times = [ int((datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - thisdate).total_seconds()/3600.0 - 0.000001) + 6 + 1 for t in report_times ]
        
        # loop over each storm and find the reports within time and distance tolerances
        #all_tolerances = []
        all_distances = []
        for i in range(len(storm_lats)):
            #print 'storm %d/%d'%(i+1,len(storm_lats))
            if len(rpts) > 0:
                #find all reports w/in 1 hour of this storm
                report_mask = ( report_times >= storm_times[i]-time_tolerance ) & ( report_times <= storm_times[i]+time_tolerance ) #add 1 here so obs between 12-13Z are matched with proper storms? 
                report_mask = np.array(report_mask)
                these_report_lons, these_report_lats = np.array(report_lons)[report_mask], np.array(report_lats)[report_mask]

                reports = zip(these_report_lons, these_report_lats)
                storms  = (storm_lons[i], storm_lats[i])

                # see if any remain after filtering, if so compute distances from storm centroid
                if len(reports) > 0:
                    t = geo.inverse( storms , reports )
                    t = np.asarray(t)

                    distances_meters = t[:,0]
                    closest_report_distance = np.amin(distances_meters)
                else:
                    closest_report_distance = -9999
            else:
                closest_report_distance = -9999

            all_distances.append(closest_report_distance)

        df['%s_report_closest_distance'%type] = all_distances

    df.to_csv('./track_data_ncarstorm_3km_csv_preprocessed/track_step_NCARSTORM_d01_%s-0000_13_time2_filtered.csv'%(yyyymmdd), float_format='%.2f', index=False)
    #df.to_csv('./track_data_ncar_2016_csv_preprocessed/track_step_ncar_3km_%s_time2.csv'%(yyyymmdd), float_format='%.2f', index=False)
    
    thisdate += timedelta(days=1)
    forecasts_processed += 1    
    print 'forecasts processed', forecasts_processed
  else:
    thisdate += timedelta(days=1)
    
  conn.close()
