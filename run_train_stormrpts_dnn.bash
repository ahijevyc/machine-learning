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

mem=120GB
walltime=2:50:00
gpu=""

# Loop through commands file one line at a time
# enumerate the cmd lines starting with zero
i=0
while read -r line
do
    if [[ $line == *"--model NSC"* ]]; then
        mem=80GB
        walltime=1:40:00
    fi
    # GPU is faster for large n-neuron model.
    # Tried each fold separately for less memory on GPU, but still not able to fit.
    for fold in 0 1 2 3 4
    do
        # Skip subsequent folds if kfold == 1
        if [[ $line == *"--kfold 1"* && $fold > 0 ]]; then
            continue
        fi
        cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $i.$fit.$fold
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=$walltime
#PBS -l select=1:ncpus=1${gpu}:mem=$mem
#PBS -o $i.$fit.$fold.out
#PBS -q casper

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf2

python train_stormrpts_dnn.py --fits $fit --folds $fold $line 
EOS
    done

    let "i=i+1"
done < $cmdfile

echo finished command file $cmdfile
