"""
To explore sensitivity to ML hyperparamaters, hundreds of models are trained. 
These hyperparameter combinations are listed in text files like cmds.txt.

This script searches for models in the list that have not been created. 
"""

import argparse
import glob
import logging
from ml_functions import full_cmd, get_argparser, get_savedmodel_path
import os
import pdb
import sys

if len(sys.argv) >= 2:
    ifile = sys.argv[1]
else:
    ifile = "cmds.txt"
cmds = open(ifile,"r").readlines()

odir = "nn"
# subdirectory like nn/hyperparam_search.HRRR
if len(sys.argv) >= 3:
    odir = sys.argv[2]

c = f"missing_model_cmds.txt"
if os.path.exists(c):
    logging.warning(f"removing {c}")
    os.remove(c)



for cmd in cmds:
    words = cmd.split()
    parser = get_argparser()
    args = parser.parse_args(cmd.split())
    savedmodel = get_savedmodel_path(args, odir=odir)
    teststart = args.teststart.strftime('%Y%m%d%H')
    testend = args.testend.strftime('%Y%m%d%H')
    scores = f"{savedmodel}.{args.kfold}fold.{teststart}-{testend}scores.txt"
    search_str = scores.replace(f"{teststart}-{testend}","*")
    score_files = glob.glob(search_str)
    missing = False
    if len(score_files) == 1:
        pass
    elif len(score_files) > 1:
        logging.warning(f"found {len(score_files)} score files matching {search_str}")
    else:
        logging.error(f"No matches for {search_str}")
        missing = True
    for i in range(args.nfits):
        for ifold in range(args.kfold):
            model_i = f"{savedmodel}_{i}/{args.kfold}fold{ifold}"
            if os.path.exists(model_i) and os.path.exists(f"{model_i}/config.yaml"):
                logging.debug(f"found {model_i} and config.yaml")
                # Initialize savedmodel. I think it is usually None. This helps
                # remember the odir for the test_stormrpts_dnn.py script.
                setattr(args, "savedmodel", savedmodel)
            elif not os.path.exists(model_i):
                logging.error(f"No {model_i}")
                missing = True
            else:
                logging.error(f"no {model_i}/config.yaml")
                missing = True
    if missing:
        f = open(c, "a")
        complete_cmd = full_cmd(args)
        f.write(complete_cmd)
        f.close()
