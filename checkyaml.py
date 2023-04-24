import logging
from ml_functions import make_fhr_str
import numpy as np
import sys
import yaml

# read ifiles from stdin, like output of find -name config.yaml

# used to check parameters in yaml file like trainstart, trainend, teststart, and testend

for line in sys.stdin:
    ifile = line.rstrip()
    print(ifile, end=" ")
    with open(ifile, "rb") as stream:
        yl = yaml.load(stream, Loader=yaml.Loader)

    # Namespace
    args = yl["args"]

    model = args.model

    for arg in ["trainstart", "trainend", "teststart", "testend"]:
        if hasattr(args, arg):
            print(f"  {arg}={getattr(args,arg).strftime('%Y%m%dT%H')}", end=" ")
    columns = yl["columns"]
    if "forecast_hour" in columns: 
        fhr_scaling_factor_mean = yl["mean"][columns.index("forecast_hour")]
        if model != "HRRR" and fhr_scaling_factor_mean != np.mean(args.fhr):
            logging.warning(f"{ifile} fhr mean scaling factor {fhr_scaling_factor_mean} does not equal mean of requested fhrs {args.fhr}")
        fhr_str = make_fhr_str(args.fhr)
        #print(f' {fhr_str}', end=" ")
    print(yl["labels"], end=" ")
    print()
