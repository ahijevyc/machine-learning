"""
To explore sensitivity to ML hyperparamaters, hundreds of models are trained. 
These hyperparameter combinations are listed in text files like cmds.txt.

This script lists the score files and model directories that can be moved to a subdirectory to get them out
of the way.
"""

import argparse
import glob
import logging
from ml_functions import get_argparser, get_savedmodel_path
import os
import pdb
import shutil
import sys

if len(sys.argv) >= 2:
    ifile = sys.argv[1]
else:
    ifile = "data/cmds.txt"

cmds = open(ifile,"r").readlines()

odir = "nn"
if len(sys.argv) == 3:
    odir = sys.argv[2]

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
    if len(score_files) == 0:
        logging.error(f"no match for {search_str}")

    for sf in score_files:
        print(sf)

    for i in range(args.nfits):
        for ifold in range(args.kfold):
            model_i = f"{savedmodel}_{i}/{args.kfold}fold{ifold}"
            if os.path.exists(model_i):
                print(os.path.dirname(model_i)) # not the 5fold0 part, but the directory above
