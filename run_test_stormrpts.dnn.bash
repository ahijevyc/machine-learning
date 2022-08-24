#!/bin/bash

cmdfile=cmds.txt

i=0

# Loop through commands file one line at a time
while read -r line
do

    cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.test
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=3:00:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=10:mem=130GB
#PBS -o $i.test.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python test_stormrpts_dnn.py --nprocs 10 $line
EOS
    let "i=i+1"
done < $cmdfile
