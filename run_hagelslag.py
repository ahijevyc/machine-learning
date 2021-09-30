#!/usr/bin/env python

from datetime import *
from subprocess import *
import os, time, sys

sdate = datetime(2019,4,19,0,0,0)
edate = datetime(2019,4,30,0,0,0)
timeinc = timedelta(hours=24)
model = 'NSC3km-12sec'
#shr = 12
shr = 1
ehr = 36

thisdate = sdate
while thisdate <= edate:
     yyyymmddhh = thisdate.strftime('%Y%m%d%H')

     # for NSC runs
     #if not os.path.exists('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/%s'%yyyymmddhh):
     #    thisdate += timeinc
     #    continue
     
     # if file already exists
     #tstr = thisdate.strftime('%Y%m%d-0000')
     #if os.path.exists('/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s_13.csv'%tstr):
     #    thisdate += timeinc
     #    continue

     # config file
     if model in ['NSC3km-12sec']: config = 'ncar_storm_data_3km.config.refl'
     else: config = 'ncar_storm_data_1km.config'

     # for real-time HWT2019 data
     config = 'ncar_data_2019_hwt_grib.config'

     geyser_script = '/glade/work/sobash/NSC_objects/run_hagelslag_dav.csh'
     command = "sbatch %s"%geyser_script
     command = command.split()
     command.extend([yyyymmddhh, config, str(shr), str(ehr)])

     fout = open('/dev/null', 'w')
     #pid = Popen(execute, stdout=fout, stderr=fout).pid
     print time.ctime(time.time()),':', 'Running', ' '.join(command)
     call(command, stdout=fout, stderr=fout)

     #time.sleep(1)
 
     thisdate += timeinc
