#!/bin/csh

#PBS -A NMMM0021
#PBS -N "training"
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:ngpus=1:mem=50GB
#PBS -o basic.out
#PBS -q casper
#PBS -V 

cd /glade/work/ahijevyc/NSC_objects

ncar_pylib /glade/work/ahijevyc/20201220_daa_casper

time python HWT_mode_train.py --nfit 100 --suite basic
time python HWT_mode_train.py --nfit 100 --suite basic2
time python HWT_mode_train.py --nfit 100 --suite long
