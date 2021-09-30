#!/usr/bin/env python

from datetime import *
from subprocess import *
import os, time, sys

# NCAR ENSEMBLE DATA AVAILABLE FROM 2013-05-14 00UTC through 2013-06-15 12UTC (30-member starts 5/14 12 UTC)
sdate = datetime(2010,1,1,0,0,0)
edate = datetime(2017,12,31,0,0,0)
timeinc = timedelta(hours=24)
model = 'NSC3km-12sec'
#model = 'GEFS'
#mem = 10

thisdate = sdate
while thisdate <= edate:
         yyyymmddhh = thisdate.strftime('%Y%m%d%H')

         ## for NSC runs
         if not os.path.exists('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/%s'%yyyymmddhh):
             thisdate += timeinc
             continue

         #if os.path.exists('/glade/work/sobash/NSC_objects/grid_data/grid_data_NSC3km-12sec_d01_%s-0000.par'%yyyymmddhh):
         #    thisdate += timeinc
         #    continue

         geyser_script = '/glade/work/sobash/NSC_objects/run_compute_env_nsc_geyser.csh'
         command = "sbatch %s"%geyser_script
         command = command.split()
         #command.extend([yyyymmddhh, model, str(mem)])
         command.extend([yyyymmddhh, model])

         fout = open('/dev/null', 'w')
         #pid = Popen(execute, stdout=fout, stderr=fout).pid
         print(time.ctime(time.time()),':', 'Running', ' '.join(command))
         call(command, stdout=fout, stderr=fout)
 
         thisdate += timeinc
