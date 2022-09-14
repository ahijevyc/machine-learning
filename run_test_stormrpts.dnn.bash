#!/bin/bash

cmdfile=cmds.txt

if [ $# -eq 1 ] ; then
    cmdfile=$1
fi


i=0 # number each line of command file
ncpus=8
mem=120GB
walltime=3:00:00

# Loop through commands file one line at a time
while read -r line
do

    if [[ $line == *"--model NSC3km-12sec"* ]]; then
        mem=60GB
        ncpus=5
        walltime=0:50:00
    fi

    cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.test
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=$walltime
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=$ncpus:mem=$mem
#PBS -o $i.test.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python test_stormrpts_dnn.py --nprocs $ncpus $line
EOS
    let "i=i+1"
done < $cmdfile
