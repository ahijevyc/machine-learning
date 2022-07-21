#!/bin/csh
set fits=$1
cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N ${fits}.wsm
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=1:30:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:mem=220GB
#PBS -o ${fits}.wsm.out
#PBS -q casper
#PBS -V 

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python train_stormrpts_dnn.py --fits ${fits} --suite with_storm_mode --split 20160701 --model NSC3km-12sec 
EOS
