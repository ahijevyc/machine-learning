#!/bin/csh

#PBS -A NMMM0021
#PBS -N "training"
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:ngpus=1:mem=50GB
#PBS -o run.out
#PBS -q casper
#PBS -V 

cd /glade/work/ahijevyc/NSC_objects

source /glade/work/ahijevyc/NSC_objects/20201220_casper_daa/bin/activate.csh
#source /glade/work/ahijevyc/20190723/bin/activate.csh

time python HWT_mode_train.py --ep 50 --neurons 20 30 40 50  
