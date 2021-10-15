#!/bin/bash
yyyymmddhh=$1
cat <<EOS | qsub -V -N $yyyymmddhh -A NMMM0021 -q htc -l select=1:ncpus=5:mem=4GB,walltime=00:05:00
/glade/work/ahijevyc/NSC_objects/upscale_HRRR-ZARR.py $*
EOS
