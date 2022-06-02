#!/bin/csh
set fit=$1
cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N ${fit}.1024n
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=9:00:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:mem=60GB
#PBS -o ${fit}.1024n.out
#PBS -q casper
#PBS -V 

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python train_GLM_dnn.py --fit ${fit} --neurons 1024 --epoch 30 --batch 512 --layers 2 --suite sobash.noN7
EOS
