import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
import logging
from metpy.units import units
import numpy as np
import os
import pandas as pd # for forward fill of NaNs
import pdb
import pytz
import re
import spc
import sys


# =============Arguments===================
parser = argparse.ArgumentParser(description = "Plot NARR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cart", action='store_true', help="plot cartopy version (not TC-centric)")
parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
parser.add_argument("-d", "--debug", action='store_true')
parser.add_argument("-e", "--extent", type=float, nargs=4, help="debug plot extent lonmin, lonmax, latmin, latmax", default=[-110, -74, 23, 48])
parser.add_argument("--hail", action='store_true', help="overlay hail reports")
parser.add_argument("--no-torn", action='store_false', help="don't overlay tornado reports")
parser.add_argument("-o", "--ofile", type=str, help="name of final composite image")
parser.add_argument("--torn", action='store_true', help="overlay tornado reports")
parser.set_defaults(torn=True)
parser.add_argument("--wind", action='store_true', help="overlay wind reports")


# Assign arguments to simple-named variables
args = parser.parse_args()
cart           = args.cart
clobber        = args.clobber
debug          = args.debug
extent         = args.extent
hail           = args.hail
torn           = args.torn
wind           = args.wind

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s')
if debug:
    logger.setLevel(logging.DEBUG)

logging.debug(args)

all_storm_reports = spc.get_storm_reports(start=datetime.datetime(2019,10,2, tzinfo=pytz.UTC), end=datetime.datetime(2020,12,2,tzinfo=pytz.UTC), event_types=["hail"])
logging.info(f"found {len(all_storm_reports)} storm reports")
rs_storm_reports = spc.RyanSobash(start=datetime.datetime(2019,10,2, tzinfo=pytz.UTC), end=datetime.datetime(2020,12,2,tzinfo=pytz.UTC), event_type="hail")
logging.info(f"Ryan Sobash found {len(rs_storm_reports)} storm reports")

sig = all_storm_reports["significant"]
storm_reports = all_storm_reports[sig]

logging.info("cartopy view for debugging...")
fig = plt.figure()
axc = plt.axes(projection=cartopy.crs.LambertConformal())
axc.set_extent(extent, crs=cartopy.crs.PlateCarree()) 
legend_items = spc.plot(storm_reports, axc, drawrange=0, colorbyfreq=True)
axc.legend(handles=legend_items.values(), fontsize='xx-small')

# *must* call draw in order to get the axis boundary used to add ticks:
fig.canvas.draw()

# Define gridline locations and draw the lines using cartopy's built-in gridliner:
xticks = list(range(-160,-50,10))
yticks = list(range(0,65,5))
axc.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
axc.yaxis.set_major_formatter(LATITUDE_FORMATTER)
axc.set_xlabel('')
axc.set_ylabel('')
axc.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.35, alpha=0.55)
axc.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), linewidth=0.5, alpha=0.55)
plt.show()

ofile = os.path.realpath("t.png")
plt.savefig(ofile)
logging.info(f"made {ofile}")

