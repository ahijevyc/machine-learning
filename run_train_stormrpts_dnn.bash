#!/bin/bash

# First argument must be the fit index

if [ $# -lt 1 ] ; then
    echo "need fit arg"
    exit 1
fi

fit=$1

# Use smaller fit-specific command file created by missing_models.py if it exists.
cmdfile=cmds_$fit.txt
if [[ ! -f "$cmdfile" ]]; then
    cmdfile=cmds.txt
fi

echo read command file $cmdfile

# keep track of cmd line
i=0
mem=140GB
walltime=2:45:00

# Loop through commands file one line at a time
while read -r line
do
    if [[ $line == *"--model NSC3km-12sec"* ]]; then
        mem=63GB
        walltime=0:20:00
    fi
    # Use GPU for 1024-neuron model for speed.
    # Run each fold separately or else you run out of memory on GPU
    if [[ $line == *"--neurons 1024"* || $line == *"--neurons 256"* ]]; then
        for fold in 0 1 2 3 4
        do
            cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.$fit.$fold
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=0:45:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:ngpus=1:mem=130GB
#PBS -o $i.$fit.$fold.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python train_stormrpts_dnn.py --fits $fit --folds $fold $line 
EOS
        done

    else 
        cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.$fit
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=$walltime
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:mem=$mem
#PBS -o $i.$fit.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

python train_stormrpts_dnn.py --fits $fit $line 
EOS
    fi
    let "i=i+1"
done < $cmdfile


echo finished command file $cmdfile
