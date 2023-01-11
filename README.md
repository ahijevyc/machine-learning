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
                              [--fhr FHR [FHR ...]] [--flash FLASH]
                              [--kfold KFOLD] [--ifile IFILE]
                              [--layers LAYERS]
                              [--learning_rate LEARNING_RATE]
                              [--model {HRRR,NSC1km,NSC3km-12sec,NSC15km}]
                              [--glm] [--neurons NEURONS [NEURONS ...]]
                              [--optimizer {adam,sgd}]
                              [--reg_penalty REG_PENALTY] [--rptdist RPTDIST]
                              [--savedmodel SAVEDMODEL] [--trainend TRAINEND]
                              [--trainstart TRAINSTART] [--testend TESTEND]
                              [--teststart TESTSTART] [--suite SUITE]
                              [--twin TWIN] [--fits FITS [FITS ...]]
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
  --fhr FHR [FHR ...]   forecast hours (default: [1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
  --flash FLASH         GLM flash count threshold (default: 10)
  --kfold KFOLD         apply kfold cross validation to training set (default:
                        5)
  --ifile IFILE         Read this parquet input file. Otherwise guess which
                        one to read. (default: None)
  --layers LAYERS       number of hidden layers (default: 2)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --model {HRRR,NSC1km,NSC3km-12sec,NSC15km}
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
  --trainend TRAINEND   training set end (default: None)
  --trainstart TRAINSTART
                        training set start (default: None)
  --testend TESTEND     testing set end (default: 20220101T00)
  --teststart TESTSTART
                        testing set start (default: 20201202T12)
  --suite SUITE         name for suite of training features (default: default)
  --twin TWIN           time window in hours (default: 2)
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
                             [--nfits NFITS] [--epochs EPOCHS]
                             [--fhr FHR [FHR ...]] [--flash FLASH]
                             [--kfold KFOLD] [--ifile IFILE] [--layers LAYERS]
                             [--learning_rate LEARNING_RATE]
                             [--model {HRRR,NSC1km,NSC3km-12sec,NSC15km}]
                             [--glm] [--neurons NEURONS [NEURONS ...]]
                             [--optimizer {adam,sgd}]
                             [--reg_penalty REG_PENALTY] [--rptdist RPTDIST]
                             [--savedmodel SAVEDMODEL] [--trainend TRAINEND]
                             [--trainstart TRAINSTART] [--testend TESTEND]
                             [--teststart TESTSTART] [--suite SUITE]
                             [--twin TWIN] [--nprocs NPROCS]

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
  --fhr FHR [FHR ...]   forecast hours (default: [1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
  --flash FLASH         GLM flash count threshold (default: 10)
  --kfold KFOLD         apply kfold cross validation to training set (default:
                        5)
  --ifile IFILE         Read this parquet input file. Otherwise guess which
                        one to read. (default: None)
  --layers LAYERS       number of hidden layers (default: 2)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --model {HRRR,NSC1km,NSC3km-12sec,NSC15km}
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
  --trainend TRAINEND   training set end (default: None)
  --trainstart TRAINSTART
                        training set start (default: None)
  --testend TESTEND     testing set end (default: 20220101T00)
  --teststart TESTSTART
                        testing set start (default: 20201202T12)
  --suite SUITE         name for suite of training features (default: default)
  --twin TWIN           time window in hours (default: 2)
  --nprocs NPROCS       verify this many forecast hours in parallel (default:
                        0)
```

### history notes

#### Jan 2023

##### clean up

Cleaned up nn/ directory by moving hyperparameter search models to nn/hyperparam_search.HRRR/.

Models trained in particular regions masked by convective mode probability moved to nn/modemask.NSC/.

Removed nn_ prefix from saved model names.

##### orphan scores.txt files

scores.txt files with no corresponding ML model tucked away in nn/orphan_score_files/. Unfortunately score.txt files
showing improvement with storm mode for tornado forecasts have no corresponding model. 

> NSC3km-12sec.default.rpt_40km_2hr.1024n.ep10.f01-f48.bs1024.1layer.SGD.L20.01.lr0.01.0.0.1fold.scores.txt
> nn_NSC3km-12sec.with_CNN_DNN_storm_mode_nprob.rpt_40km_2hr.1024n.ep10.f01-f48.bs1024.1layer.SGD.L20.01.lr0.01.0.0.1fold.scores.txt

Retrained a model with same parameters but it shows no improvement with storm mode. Could be a bug, small sample size (noise) or  
difference in 3-km training set (fixed W_MIN, UP_HELI_MIN time sin/cos component variables, shorter end time)

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
