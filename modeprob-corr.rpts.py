import argparse
import datetime
import logging
import matplotlib.pyplot as plt
from ml_functions import get_argparser, rptdist2bool, savedmodel_default
import numpy as np
import os
import pandas as pd
import pdb
import seaborn as sns
import sys

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = get_argparser()

args = parser.parse_args()

logging.info(args)


# Assign arguments to simple-named variables
clobber               = args.clobber
debug                 = args.debug
glm                   = args.glm
kfold                 = args.kfold
model                 = args.model
nfit                  = args.nfits
rptdist               = args.rptdist
savedmodel            = args.savedmodel
teststart             = args.teststart
twin                  = args.twin


if debug:
    logging.basicConfig(level=logging.DEBUG)


### saved model name ###
if savedmodel is None:
    savedmodel = savedmodel_default(args, fhr_str='f01-f48') # use model trained on f01-f48 regardless of the hour you are testing
logging.info(f"savedmodel={savedmodel}")


for ifold in range(kfold):
    for i in range(0,nfit):
        savedmodel_i = f"nn/nn_{savedmodel}_{i}/{kfold}fold{ifold}"
        assert os.path.exists(savedmodel_i), f"{savedmodel_i} not found"

    nextfit = f"nn/nn_{savedmodel}_{i+1}"
    if os.path.exists(nextfit):
        logging.warning(f"next fit exists ({nextfit}). Are you sure nfit only {nfit}?")


odir = os.path.join("/glade/scratch", os.getenv("USER"))
if glm: odir = os.path.join(odir, "GLM")
if not os.path.exists(odir):
    logging.info(f"making directory {odir}")
    os.mkdir(odir)

##################################


logging.info(f"Read {model} predictors")
if model == "HRRR":
    ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.par'
    if debug: ifile0 = f'/glade/work/ahijevyc/NSC_objects/{model}/HRRRX.32bit.fastdebug.par'
    scalingfile = "/glade/work/ahijevyc/NSC_objects/HRRR/scaling_values_all_HRRRX.pk"
    nfhr = 48
elif model == "NSC3km-12sec":
    ifile0 = f'{model}.par'
    scalingfile = f"scaling_values_{model}_{teststart:%Y%m%d_%H%M}.pk"
    nfhr = 36


if os.path.exists(ifile0):
    logging.info(f'reading {ifile0}')
    df = pd.read_parquet(ifile0, engine="pyarrow")
else:
    logging.error(f"why is there no parquet file for {model}?")
    logging.error(f"Do you need to run train_stormrpts_dnn.py to make {ifile0}?")
    sys.exit(1)

df, rptcols = rptdist2bool(df, args)


df2=df[['SS_Supercell_nprob','SS_QLCS_nprob','SS_Disorganized_nprob','CNN_1_Supercell_nprob','CNN_1_QLCS_nprob','CNN_1_Disorganized_nprob','DNN_1_Supercell_nprob','DNN_1_QLCS_nprob','DNN_1_Disorganized_nprob','any_rptdist_2hr','torn_rptdist_2hr','hailone_rptdist_2hr','wind_rptdist_2hr','sigwind_rptdist_2hr']]
corr = df2.corr().T
mask = np.triu(np.ones_like(corr,dtype=bool))
f, ax = plt.subplots(figsize=(9,9))
sns.heatmap(corr, mask=mask, cmap=sns.color_palette("viridis", as_cmap=True), vmax=0.18, square=True, linewidth=0.5, cbar_kws={"shrink":.5})
plt.show()
pdb.set_trace()
