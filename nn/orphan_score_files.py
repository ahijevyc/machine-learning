import datetime
import logging
import os
import pdb
import re
import sys

"""
Find and list score files (supplied on command line) that have no corresponding ML model. 
How did this happen? I don't know.
"""

ifiles = sys.argv[1:]
nfit = 5
orphan_score_files = []
for ifile in ifiles:
    nfold = re.findall(r"\.(\d+)fold\.", ifile)
    if nfold:
        nfold = int(nfold[0])
    else:
        logging.error(f"no nfold in {ifile}")
        continue
    if ifile.endswith("scores.txt"):
        x = re.search(r"\.\d+fold\.", ifile)
        base = ifile[:x.start()]
    else:
        logging.warning(f"unexpected ending for {ifile}")
        continue
    for ifit in range(nfit):
        for ifold in range(nfold):
            d = f"{base}_{ifit}/{nfold}fold{ifold}"
            if os.path.exists(d):
                modelt = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(d,"saved_model.pb")))
                scoret = datetime.datetime.fromtimestamp(os.path.getmtime(ifile))
                if modelt > scoret:
                    logging.warning("model {d} modified after score file {ifile}")
                    orphan_score_files.append(ifile)
            else:
                logging.warning(f"missing model {d}")
                orphan_score_files.append(ifile)

orphan_score_files = list(set(orphan_score_files)) # unique elements of list
for f in orphan_score_files:
    print(f)
    #print(f"mv -i {f} orphan_score_files")

