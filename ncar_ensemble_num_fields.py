#!/usr/bin/env python

import pygrib
import datetime

sdate = datetime.datetime(2016,1,1,0,0,0)
edate = datetime.datetime(2017,12,31,0,0,0)
tdate = sdate

while tdate <= edate:
    yyyy = tdate.strftime('%Y')
    yyyymmdd = tdate.strftime('%Y%m%d')
    yyyymmddhh = tdate.strftime('%Y%m%d%H')

    f = pygrib.open('/glade/collections/rda/data/ds300.0/%s/%s/ncar_3km_%s_mem1_f001.grb2'%(yyyy,yyyymmdd,yyyymmddhh))
    print(tdate, f.messages)
    f.close()

    tdate += datetime.timedelta(days=1)
