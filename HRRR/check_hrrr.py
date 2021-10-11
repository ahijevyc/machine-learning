#!/usr/bin/env python

import glob, datetime, subprocess, time

sdate = datetime.datetime(2018,7,15,0,0,0)
edate = datetime.datetime(2020,6,30,0,0,0)
dinc = datetime.timedelta(days=1)

tdate = sdate
field = "hrrr.t00z.wrfsfc"

while tdate <= edate:
    yyyymmddhh = tdate.strftime('%Y%m%d%H')

    files = glob.glob('/glade/scratch/sobash/HRRR/%s/%s*'%(yyyymmddhh,field))

    if len(files) < 37:
        print(yyyymmddhh, len(files))
       
    tdate += dinc
