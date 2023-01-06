"""
To explore sensitivity to ML hyperparamaters, hundreds of models are trained. 
These hyperparameter combinations are listed in text files like cmds.txt.

This script searches for models in the list that have not been created. 
"""

import argparse
import glob
import logging
from ml_functions import get_argparser, savedmodel_default
import os
import pdb
import sys
from train_stormrpts_dnn import make_fhr_str

if len(sys.argv) == 2:
    ifile = sys.argv[1]
else:
    ifile = "cmds.txt"
cmds = open(ifile,"r").readlines()

fhr = list(range(1,49))

c = f"missing_model_cmds.txt"
if os.path.exists(c):
    logging.warning(f"removing {c}")
    os.remove(c)

for cmd in cmds:
    words = cmd.split()
    parser = get_argparser()
    args = parser.parse_args(cmd.split())
    savedmodel = savedmodel_default(args, fhr_str=make_fhr_str(fhr))
    teststart = args.teststart.strftime('%Y%m%d%H')
    testend = args.testend.strftime('%Y%m%d%H')
    scores = f"nn/nn_{savedmodel}.{args.kfold}fold.{teststart}-{testend}scores.txt"
    search_str = scores.replace(f"{teststart}-{testend}","*")
    score_files = glob.glob(search_str)
    missing = False
    if len(score_files) == 1:
        pass
    elif len(score_files) > 1:
        logging.warning(f"found {len(score_files)} score files matching {search_str}")
    else:
        logging.error(f"No score file for {search_str}")
        missing = True
    for i in range(args.nfits):
        for ifold in range(args.kfold):
            model_i = f"nn/nn_{savedmodel}_{i}/{args.kfold}fold{ifold}"
            if not os.path.exists(model_i) or not os.path.exists(f"{model_i}/config.yaml"):
                missing = True
    if missing:
        f = open(c, "a")
        f.write(cmd)
        f.close()
