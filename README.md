# machine-learning-tools

## Create conda enviroment with tensorflow and hwtmode

conda env create -f environment_from_history.yaml

git clone https://github.com/NCAR/HWT_mode.git

cd HWT_mode

git checkout -b ahijevyc remotes/origin/ahijevyc

pip install . 

## Train Dense Neural Network

train_stormrpts_dnn.py

```
usage: train_stormrpts_dnn.py [-h] [--batchnorm] [--batchsize BATCHSIZE]
                              [--clobber] [-d] [--dropout DROPOUT]
                              [--nfits NFITS] [--epochs EPOCHS]
                              [--flash FLASH] [--kfold KFOLD]
                              [--layers LAYERS]
                              [--learning_rate LEARNING_RATE]
                              [--model {HRRR,NSC3km-12sec}] [--glm]
                              [--neurons NEURONS [NEURONS ...]]
                              [--optimizer {adam,sgd}]
                              [--reg_penalty REG_PENALTY] [--rptdist RPTDIST]
                              [--savedmodel SAVEDMODEL]
                              [--teststart TESTSTART] [--suite SUITE]
                              [--twin TWIN] [--fhr FHR [FHR ...]]
                              [--fits FITS [FITS ...]]
                              [--folds FOLDS [FOLDS ...]] [--seed SEED]

train/test dense neural network

optional arguments:
  -h, --help            show this help message and exit
  --batchnorm           use batch normalization (default: False)
  --batchsize BATCHSIZE
                        nn training batch size (default: 1024)
  --clobber             overwrite any old outfile, if it exists (default:
                        False)
  -d, --debug
  --dropout DROPOUT     fraction of neurons to drop in each hidden layer (0-1)
                        (default: 0.0)
  --nfits NFITS         number of times to fit (train) model (default: 5)
  --epochs EPOCHS       number of training epochs (default: 30)
  --flash FLASH         GLM flash count threshold (default: 10)
  --kfold KFOLD         apply kfold cross validation to training set (default:
                        5)
  --layers LAYERS       number of hidden layers (default: 2)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --model {HRRR,NSC3km-12sec}
                        prediction model (default: HRRR)
  --glm                 Use GLM (default: False)
  --neurons NEURONS [NEURONS ...]
                        number of neurons in each nn layer (default: [16])
  --optimizer {adam,sgd}
                        optimizer (default: adam)
  --reg_penalty REG_PENALTY
                        L2 regularization factor (default: 0.01)
  --rptdist RPTDIST     severe weather report max distance (default: 40)
  --savedmodel SAVEDMODEL
                        filename of machine learning model (default: None)
  --teststart TESTSTART
                        train with storms before this time; test this time and
                        after (default: 202012021200)
  --suite SUITE         name for suite of training features (default: default)
  --twin TWIN           time window in hours (default: 2)
  --fhr FHR [FHR ...]   forecast hour (default: [1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
  --fits FITS [FITS ...]
                        work on specific fit(s) so you can run many in
                        parallel (default: None)
  --folds FOLDS [FOLDS ...]
                        work on specific fold(s) so you can run many in
                        parallel (default: None)
  --seed SEED           random number seed for reproducability (default: None)
```

## Test Dense Neural Network

test_stormrpts_dnn.py

```
usage: test_stormrpts_dnn.py [-h] [--batchnorm] [--batchsize BATCHSIZE]
                             [--clobber] [-d] [--dropout DROPOUT]
                             [--nfits NFITS] [--epochs EPOCHS] [--flash FLASH]
                             [--kfold KFOLD] [--layers LAYERS]
                             [--learning_rate LEARNING_RATE]
                             [--model {HRRR,NSC3km-12sec}] [--glm]
                             [--neurons NEURONS [NEURONS ...]]
                             [--optimizer {adam,sgd}]
                             [--reg_penalty REG_PENALTY] [--rptdist RPTDIST]
                             [--savedmodel SAVEDMODEL] [--teststart TESTSTART]
                             [--suite SUITE] [--twin TWIN] [--ifile IFILE]
                             [--nprocs NPROCS]

train/test dense neural network

optional arguments:
  -h, --help            show this help message and exit
  --batchnorm           use batch normalization (default: False)
  --batchsize BATCHSIZE
                        nn training batch size (default: 1024)
  --clobber             overwrite any old outfile, if it exists (default:
                        False)
  -d, --debug
  --dropout DROPOUT     fraction of neurons to drop in each hidden layer (0-1)
                        (default: 0.0)
  --nfits NFITS         number of times to fit (train) model (default: 5)
  --epochs EPOCHS       number of training epochs (default: 30)
  --flash FLASH         GLM flash count threshold (default: 10)
  --kfold KFOLD         apply kfold cross validation to training set (default:
                        5)
  --layers LAYERS       number of hidden layers (default: 2)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --model {HRRR,NSC3km-12sec}
                        prediction model (default: HRRR)
  --glm                 Use GLM (default: False)
  --neurons NEURONS [NEURONS ...]
                        number of neurons in each nn layer (default: [16])
  --optimizer {adam,sgd}
                        optimizer (default: adam)
  --reg_penalty REG_PENALTY
                        L2 regularization factor (default: 0.01)
  --rptdist RPTDIST     severe weather report max distance (default: 40)
  --savedmodel SAVEDMODEL
                        filename of machine learning model (default: None)
  --teststart TESTSTART
                        train with storms before this time; test this time and
                        after (default: 202012021200)
  --suite SUITE         name for suite of training features (default: default)
  --twin TWIN           time window in hours (default: 2)
  --ifile IFILE         parquet input file (default: None)
  --nprocs NPROCS       verify this many forecast hours in parallel (default:
                        0)
```

### history notes

#### Jan 2023

##### clean up

Cleaned up nn/ directory by moving hyperparameter search models to nn/hyperparam_search.HRRR/.

Removed nn_ prefix from saved model names.

##### NSC training period changed

The old time range of 3-km NSC was 20101024 - 20191020.

Now with 1-km and 15-km NSC data, and redone 3-km NSC data, the model initialization
time range tightens to 20101024 - 20170330.
With the old 2019 end time, it made sense to partition training and 
testing data into two ranges with 20160701 as the divider. That had 1+ years in the testing set.
Now that we stop at 20170330, it makes more sense to use an earlier partition, 20160101. 
Old models trained through 20160701 and tested through 2019 are in a subdirectory nn/trainend20160701.NSC/.

##### correct forecast hour range

Corrected fhr list going forward, both in config.yaml and output filenames. It was hard-coded to 
f01-f48 for a long time. That worked for HRRR. But NSC only goes to fhr=36. And if you want to train with 
storm mode, it is limited to f12-f36. 

#### Dec 2021

accidentally deleted all important .py scripts (except HWT_mode_train.py) by adding them to git
and removing .git directory. I was trying to change branch from master to main.

Removed files matching
catalog.py check\*py com\*py ens\*py get\*py HWT\*py loop_through_dates.py make_scaler.py ncar_ensemble_num_fields.py neural_network_train_gridded.py random_forest_preprocess_gridded.py read_pred.py run_HWT_mode_train.py saveNewMap.py scalar2vector.py show_importances.py showtop2021HWT.py verify_forecasts_bss_spatial.py
