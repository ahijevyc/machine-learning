#!/bin/csh
set fits=$1
cat <<EOS | qsub
#!/bin/csh
#PBS -A NMMM0021
#PBS -N ${fits}.d
#PBS -S /bin/csh
#PBS -j oe
#PBS -l walltime=1:00:00
#PBS -l gpu_type=v100
#PBS -l select=1:ncpus=1:ngpus=1:mem=150GB
#PBS -o ${fits}.d.out
#PBS -q casper
#PBS -V 

cd /glade/work/ahijevyc/NSC_objects

module load conda # if you need conda environment.
conda activate tf

#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim adam --glm --epochs 10 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim adam --glm --epochs 10 --L2
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim sgd --glm --epochs 10 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim sgd --glm --epochs 10 --L2
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim adam --glm --epochs 10 --dropout 0.1 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim adam --glm --epochs 10 --dropout 0.1 --L2 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim sgd --glm --epochs 10 --dropout 0.1 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim sgd --glm --epochs 10 --dropout 0.1 --L2 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 1024 --layer 1 --optim sgd --glm --epochs 10 --dropout 0.1 --L2 --batchnorm

python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 30 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 30 --dropout 0.1
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 10 --dropout 0.1 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 10 --dropout 0.1  --L2
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 10 --dropout 0.1  --L2 --batchnorm
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 2 --optim adam --glm --epochs 10 
#python train_stormrpts_dnn.py --fits ${fits} --batchsize 1024 --neurons 16 --layer 3 --optim adam --glm --epochs 10 --dropout 0.1
EOS
