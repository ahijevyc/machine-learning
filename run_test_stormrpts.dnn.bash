#!/bin/bash

cmdfile=cmds.txt

if [ $# -eq 1 ] ; then
    cmdfile=$1
fi


i=0 # number each line of command file
ncpus=6
mem=433GB
walltime=3:00:00

echo read $cmdfile
# Loop through commands file one line at a time
while read -r line
do

    if [[ $line == *"--model NSC"* ]]; then
        mem=70GB
        ncpus=$ncpus
        walltime=2:00:00
    fi

    cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.test
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=$walltime
## if casper htc is sitting in the queue for a while, try ngpus=1
#PBS -l select=1:ncpus=${ncpus}:mem=$mem:ngpus=1
#PBS -o $i.test.out
#PBS -q main

cd /glade/work/ahijevyc/NSC_objects
module load conda
conda activate /glade/u/home/ahijevyc/miniconda3/envs/tf2

python test_stormrpts_dnn.py --nprocs $ncpus $line

EOS
    let "i=i+1"
done < $cmdfile
