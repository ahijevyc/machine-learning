#!/usr/bin/env python

import numpy as np
from datetime import *
import time
from mpl_toolkits.basemap import *
import sqlite3
from netCDF4 import Dataset

def get_osr_gridded(sdate, edate, nx, ny, report_types=['wind', 'hail', 'torn'], inc=1, db='reports_all.db'):
  # grid points for 81 km grid
  awips_proj = Basemap(projection='lcc', resolution=None, llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.385, urcrnrlat=57.29, lat_1=25.0, lat_2=25.0, lon_0=-95)
  gridpts81 = awips_proj.makegrid(nx, ny, returnxy=True)
  gridx, gridy = gridpts81[2], gridpts81[3]

  # grid points for 3km grid
  if nx == 1580 and ny == 985:
      wrffile = "/glade/scratch/sobash/RT2013/grid_3km.nc"
      wrffile = "/glade/scratch/sobash/RT2015/rt2015_grid.nc"
      f = Dataset(wrffile, 'r')
      lats = f.variables['XLAT'][0,:]
      lons = f.variables['XLONG'][0,:]
      f.close()
      gridx, gridy = awips_proj(lons, lats) 
  elif nx == 1199 and ny == 799:
      import pygrib
      f = pygrib.open("/glade/p/nmmm0001/schwartz/rt_ensemble/SSEO/sseo01/sseo01_2015062700f028.grb")
      lats, lons = f[1].latlons()
      ny, nx = lats.shape
      f.close()
      gridx, gridy = awips_proj(lons, lats)
 
  # READ STORM REPORTS FROM DATABASE
  #conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports.db')
  conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/%s'%db)
  c = conn.cursor()
  rpts = []
  for type in report_types:
          if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
          elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
          #elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
          elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
          elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 1.00 ORDER BY datetime asc" % (sdate,edate))

          elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_%s WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (type,sdate,edate))
          elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))
          rpts.extend(c.fetchall())
  conn.close()

  # PLACE INTO 3D ARRAY
  timediff = edate - sdate
  diffhrs = timediff.total_seconds()/3600
  osr81_all = np.zeros((int(diffhrs), ny, nx))
  
  if len(rpts) > 0:
      olats, olons, dt = list(zip(*rpts))
      xobs, yobs = awips_proj(olons, olats)

      for i,d in enumerate(dt):
          st    =  datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
          tind  =  int((st - sdate).total_seconds()/3600 - 0.000001)
          xind  =  np.abs(gridx[0,:] - xobs[i]).argmin()
          yind  =  np.abs(gridy[:,0] - yobs[i]).argmin()

          osr81_all[tind,yind,xind] = 1
  return osr81_all

if __name__ == "__main__":
    s = time.time()
    gmt2cst = timedelta(hours=6)
    test = get_osr_gridded(datetime(2015,5,30,12,0,0) - gmt2cst, datetime(2015,5,31,12,0,0) - gmt2cst, 65, 93, report_types=['all'])
    e = time.time()
    print(e - s)
