#!/usr/bin/env python

from datetime import *
from subprocess import *
import os, time, sys, glob

sdate = datetime(2020,12,3,0,0,0)
edate = datetime(2021,5,31,12,0,0)
timeinc = timedelta(hours=12)
model = 'HRRR'

thisdate = sdate
while thisdate <= edate:
    yyyymmddhh = thisdate.strftime('%Y%m%d%H')

    geyser_script = '/glade/work/sobash/NSC_objects/HRRR/run_compute_env_nsc_geyser.csh'
    #geyser_script = '/glade/work/sobash/NSC_objects/HRRR/run_ml_processing.csh'

    # if HRRR data doesnt exist for this date, then skip
    #if len(os.listdir('/glade/scratch/sobash/HRRRv4/%s'%yyyymmddhh)) < 1:
    #    thisdate += timeinc
    #    continue

    #command = "sbatch %s"%(geyser_script)
    #command = "qsubcasper -v yyyymmddhh=%s %s"%(yyyymmddhh, geyser_script)
    command = "qsub -v yyyymmddhh=%s %s"%(yyyymmddhh, geyser_script)
    command = command.split()
    #command.extend([yyyymmddhh, model])

    fout = open('/dev/null', 'w')
    #pid = Popen(execute, stdout=fout, stderr=fout).pid
    print(time.ctime(time.time()),':', 'Running', ' '.join(command))
    call(command, stdout=fout, stderr=fout)
 
    thisdate += timeinc
