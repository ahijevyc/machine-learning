import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
import pdb
import pygrib
import s3fs
import sys
import xarray

logging.basicConfig(level=logging.INFO)

# =============Arguments===================
parser = argparse.ArgumentParser(description = "Read AWS", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("itime", help='initialization time')
parser.add_argument("fhr", type=int, help='forecast hour')
parser.add_argument("-d", "--debug", action="store_true", help='debug mode')
args = parser.parse_args()


if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
fhr = args.fhr
itime = pd.to_datetime(args.itime)
odir    = os.getenv("TMPDIR") 

bucket = 's3://noaa-rrfs-pds'
fs = s3fs.S3FileSystem(anon=True)
url = os.path.join(
        "noaa-rrfs-pds",
        'rrfs_a', 
        "rrfs_a." +itime.strftime('%Y%m%d'),
        itime.strftime('%H'),
        "control"
        )
files = fs.ls(url)
pdb.set_trace()
logging.info(f"{len(files)} {url} files")
logging.debug(files)
for f in files:
    ofile   = os.path.join(odir, os.path.basename(f))
    fs.get(f,ofile)
    logging.info(ofile)

