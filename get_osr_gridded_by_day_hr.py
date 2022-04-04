#!/usr/bin/env python

import numpy as np
from datetime import *
import time
from mpl_toolkits.basemap import *
import sqlite3

def get_osr_gridded_by_day_hr(sdate, edate, nx, ny, report_types=['wind', 'hail', 'torn']):
  # grid points for 81 km grid
  awips_proj = Basemap(projection='lcc', resolution=None, llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.385, urcrnrlat=57.29, lat_1=25.0, lat_2=25.0, lon_0=-95)
  gridpts81 = awips_proj.makegrid(nx, ny, returnxy=True)
  gridx, gridy = gridpts81[2], gridpts81[3]

  # READ STORM REPORTS FROM DATABASE
  conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_v20200626.db')
  c = conn.cursor()
  rpts = []
  for type in report_types:
          if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
          elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
          elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
          elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
          elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))
          rpts.extend(c.fetchall())
  conn.close()

  # PLACE INTO 3D ARRAY
  #osr81_all = np.zeros((366,ny,nx))
  #osr81_count = np.zeros((366,ny,nx))
  osr81_all = np.zeros((366,24,ny,nx))
  osr81_count = np.zeros((366,24,ny,nx))
      
  gmt2cst = timedelta(hours=6)

  if len(rpts) > 0:
      olats, olons, dt = list(zip(*rpts))
      xobs, yobs = awips_proj(olons, olats)

      # total for each day of the year
      for i,d in enumerate(dt):
          #database times are CST, convert to UTC
          # obs between 00z-01z (0 array index) are used to verify 01z forecasts
          st    =  datetime.strptime(d, '%Y-%m-%d %H:%M:%S') + gmt2cst
          tind  =  st.timetuple().tm_yday #aggregate by day of year
          hind  =  int(st.hour)
          xind  =  np.abs(gridx[0,:] - xobs[i]).argmin()
          yind  =  np.abs(gridy[:,0] - yobs[i]).argmin()

          #osr81_all[tind-1,yind,xind] = 1
          #osr81_count[tind-1,yind,xind] += 1
          osr81_all[tind-1,hind,yind,xind] = 1
          osr81_count[tind-1,hind,yind,xind] += 1

  return (osr81_all, osr81_count)
