#!/bin/bash

cmdfile=cmds.txt

if [ $# -eq 1 ] ; then
    cmdfile=$1
fi


i=0 # number each line of command file
ncpus=2
mem=299GB
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
#PBS -l gpu_type=v100
## if casper htc is sitting in the queue for a while, try ngpus=1
#PBS -l select=1:ncpus=${ncpus}:mem=$mem:ngpus=1
#PBS -o $i.test.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

conda activate tf2

python test_stormrpts_dnn.py --nprocs $ncpus $line

EOS
    let "i=i+1"
done < $cmdfile
