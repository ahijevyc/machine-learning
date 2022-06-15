# machine-learning-tools

## Create conda enviroment

conda env create -f environment_from_history.yaml

git clone https://github.com/NCAR/HWT_mode.git

cd HWT_mode

git checkout -b ahijevyc remotes/origin/ahijevyc

pip install . 

## Train Dense Neural Network

train_stormrpts_dnn.py

```
usage: train_stormrpts_dnn.py [-h] [--batchsize BATCHSIZE] [--clobber] [-d]
                        [--dropouts DROPOUTS [DROPOUTS ...]]
                        [--fhr FHR [FHR ...]] [--fits FITS [FITS ...]]
                        [--nfits NFITS] [--epochs EPOCHS] [--flash FLASH]
                        [--layers LAYERS] [--model {HRRR,NSC3km-12sec}]
                        [--noglm] [--savedmodel SAVEDMODEL]
                        [--neurons NEURONS [NEURONS ...]] [--rptdist RPTDIST]
                        [--splittime SPLITTIME] [--suite SUITE] [--twin TWIN]

train neural network

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        nn training batch size (default: 512)
  --clobber             overwrite any old outfile, if it exists (default:
                        False)
  -d, --debug
  --dropouts DROPOUTS [DROPOUTS ...]
                        fraction of neurons to drop in each hidden layer (0-1)
                        (default: [0.0])
  --fhr FHR [FHR ...]   forecast hour (default: [1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
  --fits FITS [FITS ...]
                        work on specific fit(s) so you can run many in
                        parallel (default: None)
  --nfits NFITS         number of times to fit (train) model (default: 10)
  --epochs EPOCHS       number of training epochs (default: 30)
  --flash FLASH         GLM flash threshold (default: 10)
  --layers LAYERS       number of hidden layers (default: 2)
  --model {HRRR,NSC3km-12sec}
                        prediction model (default: HRRR)
  --glm                 use GLM (default: False)
  --savedmodel SAVEDMODEL
                        filename of machine learning model (default: None)
  --neurons NEURONS [NEURONS ...]
                        number of neurons in each nn layer (default: [16])
  --rptdist RPTDIST     severe weather report max distance (default: 40)
  --splittime SPLITTIME
                        train with storms before this time; test this time and
                        after (default: 202012021200)
  --suite SUITE         name for suite of training features (default: sobash)
  --twin TWIN           time window in hours (default: 2)

```

## Test Dense Neural Network

test_stormrpts_dnn.py

```
usage: test_stormrpts_dnn.py [-h] [--batchsize BATCHSIZE] [--clobber] [-d]
                       [--nfits NFITS] [--epochs EPOCHS] [--flash FLASH]
                       [--layers LAYERS] [--model {HRRR,NSC3km-12sec}]
                       [--noglm] [--savedmodel SAVEDMODEL]
                       [--neurons NEURONS [NEURONS ...]] [--nprocs NPROCS]
                       [--rptdist RPTDIST] [--suite SUITE] [--twin TWIN]

test neural network(s) in parallel. output truth and predictions from each
member and ensemble mean for each forecast hour

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        nn training batch size (default: 512)
  --clobber             overwrite any old outfile, if it exists (default:
                        False)
  -d, --debug
  --nfits NFITS         number of times to fit (train) model (default: 10)
  --epochs EPOCHS       number of training epochs (default: 30)
  --flash FLASH         GLM flash threshold (default: 10)
  --layers LAYERS       number of hidden layers (default: 2)
  --model {HRRR,NSC3km-12sec}
                        prediction model (default: HRRR)
  --glm                 use GLM (default: False)
  --savedmodel SAVEDMODEL
                        filename of machine learning model (default: None)
  --neurons NEURONS [NEURONS ...]
                        number of neurons in each nn layer (default: [16])
  --nprocs NPROCS       verify this many forecast hours in parallel (default:
                        12)
  --rptdist RPTDIST     severe weather report max distance (default: 40)
  --suite SUITE         name for suite of training features (default: sobash)
  --twin TWIN           time window in hours (default: 2)
```

### history notes
accidentally deleted all important .py scripts (except HWT_mode_train.py) by adding them to git
and removing .git directory. I was trying to change branch from master to main.

Removed files matching
catalog.py check\*py com\*py ens\*py get\*py HWT\*py loop_through_dates.py make_scaler.py ncar_ensemble_num_fields.py neural_network_train_gridded.py random_forest_preprocess_gridded.py read_pred.py run_HWT_mode_train.py saveNewMap.py scalar2vector.py show_importances.py showtop2021HWT.py verify_forecasts_bss_spatial.py
