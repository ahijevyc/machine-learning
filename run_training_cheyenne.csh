#!/bin/tcsh

#SBATCH -J training
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
## tried --gres=gpu:v100:1 (no constraint) and it was taking at least twice as long
#SBATCH --constraint=gp100
#SBATCH --gres=gpu:gp100:1
##SBATCH --mem=290GB
#SBATCH --mem=120GB
##SBATCH -t 24:00:00
#SBATCH -t 7:00:00
#SBATCH -A nmmm0021
#SBATCH -p dav
#SBATCH -e training.err.%J
#SBATCH -o training.out.%J

##PBS -S /bin/csh
##PBS -N "training"
##PBS -A NMMM0021
##PBS -l walltime=03:00:00
##PBS -q economy
##PBS -o run_training.out
##PBS -j oe
##PBS -l select=1:ncpus=36:mpiprocs=36:mem=109GB
##PBS -V 

cd /glade/work/ahijevyc/NSC_objects

#used this version of ncar_pylib for training for paper (with python 3.6)
#ncar_pylib 20190326 ... kerasV2.2.4 tensorflowV1.13.1 scikit-learn 0.20.3

#source /glade/work/sobash/npl_clone_casper/bin/activate.csh
source /glade/work/ahijevyc/20190723/bin/activate.csh

module load cuda/10.1

#usage: neural_network_train_gridded.py [-h] [-s SPACE] [-t TIME]
#                                       [--edate EDATE] [--sdate SDATE]
#                                       [--members MEMBERS [MEMBERS ...]]
#                                       [--plot] [--nopredict] [--notrain]
#                                       [--year YEAR]
#                                       [--latlon_hash_buckets LATLON_HASH_BUCKETS]
#                                       [--force_new] [-d]
./neural_network_train_gridded.py $*
