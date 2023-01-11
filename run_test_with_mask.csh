#!/bin/csh

set nproc=5
#foreach suite (default with_CNN_DNN_storm_mode_nprob)
foreach suite (with_CNN_DNN_storm_mode_prob)
    foreach field (CNN_1_QLCS_nprob CNN_1_Supercell_nprob DNN_1_QLCS_nprob DNN_1_Supercell_nprob CNN_1_Disorganized_nprob DNN_1_Disorganized_nprob)
        foreach thresh (0.05 0.1)
            cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N $field$thresh$suite
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=${nproc}:mem=65GB
#PBS -o $field.$thresh.$suite.out
#PBS -q casper

module load conda
conda activate tf2

python test_stormrpts_dnn_with_mask.py $field $thresh --suite $suite --nproc $nproc --neurons 1024 --layers 1 --optim sgd --learning_rate 0.01 --dropout 0 --epoch 10 --model NSC3km-12sec --splittime 20160701 --kfold 1
EOS

end
end
end

