#!/bin/tcsh

#SBATCH -J testing
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=110GB
#SBATCH -t 2:00:00
#SBATCH -A nmmm0021
#SBATCH -p dav
#SBATCH -e testing.err.%J
#SBATCH -o testing.out.%J

cd /glade/work/ahijevyc/NSC_objects

#used this version of ncar_pylib for training for paper (with python 3.6)
#ncar_pylib 20190326 ... kerasV2.2.4 tensorflowV1.13.1 scikit-learn 0.20.3

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
