#!/usr/bin/env python

from datetime import *
from subprocess import *
import os, time, sys

def run_script(command):
     #pid = Popen(execute, stdout=fout, stderr=fout).pid
     print(time.ctime(time.time()),':', 'Running', ' '.join(command))
     call(command, stdout=fout, stderr=fout)

sdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
     
fout = open('/dev/null', 'w')

# post for hagelslag objects
for hh in range(0,37,3):
     yyyymmddhh = sdate.strftime('%Y%m%d%H')
     shr = hh
     ehr = hh+2
     if hh==36: ehr = hh

     config = 'hagelslag.config.hrrr.refl.json'
     
     geyser_script = '/glade/work/sobash/NSC_objects/HRRR/run_hagelslag_dav_rt2020.csh'
     command = "sbatch %s"%geyser_script
     command = command.split()
     command.extend([yyyymmddhh, config, str(shr), str(ehr)])
     run_script(command)
