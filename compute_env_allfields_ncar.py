#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
from datetime import *
import pygrib
import time, os, sys
import scipy.ndimage as ndimage
import pickle as pickle
import multiprocessing
from scipy import spatial
from mpl_toolkits.basemap import *
import scipy.ndimage.filters
from scipy.interpolate import griddata, RectBivariateSpline
import matplotlib.colors as colors
import matplotlib.pyplot as plt

### THIS CODE EVOLVED FROM CODE WITHIN /glade/u/home/sobash/NSC_scripts
### TO UPSCALE 3-KM CAM DATA TO AN 80-KM GRID

import pdb
def get_message(fh, tdate, validDate, f, level=None):
    select_keywords = ncar_grb(f, tdate=tdate, level=level)
    # TODO figure out how to select validDate when field has None (like 1-h max UH)
    # print(select_keywords)
    selected_messages = fh.select(analDate=tdate, **select_keywords)
    if len(selected_messages) > 1:
        pass
        #print(select_keywords, "matched more than one message")
    return selected_messages

def ncar_grb(f, tdate=datetime(2017,12,30,0), level=None):
    #print("ncar_grb():", f, tdate)
    grb_dict = {
            "PREC_ACC_NC" : {"shortName" : "tp", "level": 0},
            "GHT_PL" : {"shortName" : "gh", "level": level},
            "HAILCAST_DIAM_MAX": {"parameterName" : "AFWA Hailcast Diameter Max"}, 
            "MLLCL" : {"shortName" : "gh", "bottomLevel": 0, "topLevel": 9000},
            "MLCAPE" : {"shortName" : "cape", "bottomLevel": 0, "topLevel": 9000},
            "MLCINH" : {"shortName" : "cin", "bottomLevel": 0, "topLevel": 9000},
            "MUCAPE" : {"shortName" : "cape", "bottomLevel": 0, "topLevel": 25500},
            "SBCAPE" : {"shortName" : "cape", "level": 0},
            "SBCINH" : {"shortName" : "cin", "level": 0},
            "SRH01" : {"shortName" : "hlcy", "bottomLevel" : 0, "topLevel" : 1000 },
            "SRH03" : {"shortName" : "hlcy", "bottomLevel" : 0, "topLevel" : 3000 },
            "T_PL" : {"shortName" : "t", "level": level},
            "RH_PL" : {"parameterName" : "Relative humidity", "level": level},
            "T2" : {"shortName" : "2t", "level": 2},
            "TD2" : {"shortName" : "2d", "level": 2},
            "U_PL" : {"shortName" : "u", "level": level},
            "V_PL" : {"shortName" : "v", "level": level},
            "UP_HELI_MAX" : {"parameterName" : "199", "bottomLevel" : 2000, "topLevel" : 5000 },
            "USHR01" : {"shortName": "vucsh", "bottomLevel" : 1000, "topLevel" : 0 },
            "VSHR01" : {"shortName": "vvcsh", "bottomLevel" : 1000, "topLevel" : 0 },
            "USHR03" : {"shortName": "vucsh", "bottomLevel" : 3000, "topLevel" : 0 },
            "VSHR03" : {"shortName": "vvcsh", "bottomLevel" : 3000, "topLevel" : 0 },
            "USHR06" : {"shortName": "vucsh", "bottomLevel" : 6000, "topLevel" : 0 },
            "VSHR06" : {"shortName": "vvcsh", "bottomLevel" : 6000, "topLevel" : 0 },
            "WSPD10MAX" : {"shortName": "10si", "level" : 10 },
            }

    if datetime(2015,9,1,0) <= tdate <= datetime(2017,12,30,0):
        pass

    return grb_dict[f]


def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    import os
    rgb, appending = [], False
    rgb_dir_ys = '/glade/apps/opt/ncl/6.2.0/intel/12.1.5/lib/ncarg/colormaps'
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    if os.path.isdir(rgb_dir_ys): fh = open('%s/%s.rgb'%(rgb_dir_ys,name), 'r')
    else: fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def upscale(field, type='mean', maxsize=27):
    if model in ['NCARENS', 'NSC3km', 'NSC3km-12sec', 'GEFS']: kernel = np.ones((26,26)) #should this be 27?
    elif model in ['NSC1km']: kernel = np.ones((81,81))
    
    weights = kernel / float(kernel.size)

    if type == 'mean':
        #field = scipy.ndimage.filters.convolve(field, weights=weights, mode='constant', cval=0.0)
        field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode='constant', cval=0.0)
    elif type == 'max':
        field = scipy.ndimage.filters.maximum_filter(field, size=maxsize)
    
    field_interp = field.flatten()[nngridpts[1]].reshape((65,93))

    return field_interp

# INTERPOLATE NARR TO 80KM GRID
#fig, axes, m  = pickle.load(open('/glade/u/home/wrfrt/rt_ensemble_2018wwe/python_scripts/rt2015_CONUS.pk', 'r'))
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)

grid81 = awips.makegrid(93, 65, returnxy=True)
x81, y81 = awips(grid81[0], grid81[1])
#x81 = (x[1:,1:] + x[:-1,:-1])/2.0
#y81 = (y[1:,1:] + y[:-1,:-1])/2.0

mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask = np.logical_not(mask)
mask = mask.reshape((65,93))

sdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
edate = sdate + timedelta(hours=24)
tdate = sdate

model = 'NCARENS'
mem   = 1

if model == 'NSC1km': f = Dataset('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
if model == 'NSC3km-12sec': f = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
if model == 'GEFS': f = Dataset('/glade/scratch/sobash/ncar_ens/gefs_ics/2017042500/wrf_rundir/ens_1/diags_d02.2017-04-25_00:00:00.nc', 'r')
if model == 'NCARENS': f = pygrib.open('/glade/collections/rda/data/ds300.0/2016/20160701/ncar_3km_2016070100_mem4_f042.grb2')
lats, lons = f[1].latlons()
f.close()

# find closest 3-km or 1-km grid point to each 80-km grid point
print('finding closest grid points')
gpfname = 'nngridpts_80km_%s'%model
if os.path.exists(gpfname):
    nngridpts = pickle.load(open(gpfname, 'rb'))
else:
    xy = awips(lons.ravel(), lats.ravel())
    tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
    nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
    pickle.dump(nngridpts, open('nngridpts_80km_%s'%model, 'wb'))

ncar_grib_dict = { 44: 'UP_HELI_MAX', 45: 'UP_HELI_MIN', 48: 'UP_HELI_MAX03', 54: 'GRPL_MAX', 39: 'W_UP_MAX', 40: 'W_DN_MAX', 52: 'HAIL_MAX2D', 53: 'HAIL_MAXK1',\
                   51: 'REL_VORT_MAX01', 75: 'WSPD10MAX', 2: 'COMPOSITE_REFL_10CM', 42: 'REFD_MAX', \
                   115: 'UBSHR1', 116: 'VBSHR1', 117: 'UBSHR6', 118: 'VBSHR6', 99: 'PWAT', 124: 'SBLCL', 127: 'MLLCL', 97: 'SBCAPE', 98: 'SBCINH', \
                   128: 'MUCAPE', 129: 'MUCINH', 125: 'MLCAPE', 112: 'SRH01', 111: 'SRH03', 70: 'T2', 72: 'TD2', 62: 'PSFC' }

upscaled_fields = { 'UP_HELI_MAX':[], 'UP_HELI_MAX03':[], 'UP_HELI_MAX01':[], 'W_UP_MAX':[], 'W_DN_MAX':[], 'WSPD10MAX':[], 'STP':[], 'LR75':[], 'CAPESHEAR':[],
           'MUCAPE':[], 'SBCAPE':[], 'SBCINH':[], 'MLCINH':[], 'MLLCL':[], 'SHR06': [], 'SRH01':[], 'SRH03':[], 'T2':[], 'TD2':[], 'PSFC':[], 'PREC_ACC_NC':[], \
           'HAILCAST_DIAM_MAX':[], \
           'T925':[], 'T850':[], 'T700':[], 'T500':[], 'TD925':[], 'TD850':[], 'TD700':[], 'TD500':[], 'U925':[], 'U850':[], 'U700':[], 'U500':[], 'V925':[], 'V850':[], 'V700':[], 'V500':[], \
           'UP_HELI_MAX80':[], 'UP_HELI_MAX120':[], 'UP_HELI_MAX01-120':[] }
press_levels = [1000,925,850,700,600,500,400,300,250,200,150,100]


#TODO: find these
del(upscaled_fields["HAILCAST_DIAM_MAX"])
del(upscaled_fields["PSFC"])
del(upscaled_fields["T925"])
del(upscaled_fields["TD500"])
del(upscaled_fields["TD925"])
del(upscaled_fields["U925"])
del(upscaled_fields["V925"])
del(upscaled_fields["UP_HELI_MAX01"])
del(upscaled_fields["UP_HELI_MAX01-120"])
del(upscaled_fields["UP_HELI_MAX03"])
del(upscaled_fields["UP_HELI_MAX80"])
del(upscaled_fields["UP_HELI_MAX120"])
del(upscaled_fields["W_UP_MAX"]) # for Dec 2017 in netCDF
del(upscaled_fields["W_DN_MAX"]) # for Dec 2017 in netCDF

for fhr in range(37):
    yyyy = tdate.strftime('%Y')
    yyyymmddhh = tdate.strftime('%Y%m%d%H')
    yyyymmdd = tdate.strftime('%Y%m%d')
    yymmdd = tdate.strftime('%y%m%d')
    validDate = tdate + timedelta(hours=fhr)
    print(time.ctime(time.time()), 'reading', yyyymmddhh, fhr)
 

    ### read in NSSL dataset for this day
    try:
        wrfvalidstr = (tdate + timedelta(hours=fhr)).strftime('%Y-%m-%d_%H_%M_%S')
        wrfvalidstr2 = (tdate + timedelta(hours=fhr)).strftime('%Y-%m-%d_%H:%M:%S')
        if model == 'NSC3km-12sec': wrffile = "/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/%s/diags_d01_%s.nc"%(yyyymmddhh,wrfvalidstr)
        if model == 'NSC1km': wrffile = "/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/%s/diags_d01_%s.nc"%(yyyymmddhh,wrfvalidstr)
        if model == 'GEFS': wrffile = "/glade/scratch/sobash/ncar_ens/gefs_ics/%s/wrf_rundir/ens_%d/diags_d02.%s.nc"%(yyyymmddhh,mem,wrfvalidstr2)
        if model == 'NCARENS': wrffile = "/glade/collections/rda/data/ds300.0/%s/%s/ncar_3km_%s_mem%d_f%03d.grb2"%(yyyy,yyyymmdd,yyyymmddhh,mem,fhr)
        fh = pygrib.open(wrffile)
        if fhr == 0:
            for msg in fh:
                print(msg.validDate, msg.parameterName, msg.shortName, msg.units, msg.bottomLevel, msg.topLevel)
        #fh = Dataset(wrffile)
      
        # populate dictionary of upscaled fields
        for f in upscaled_fields.keys():
            print(time.ctime(time.time()), f)
            if f == 'SHR06':
                this_field1 = get_message(fh, tdate, validDate, 'USHR06')[0].values
                this_field2 = get_message(fh, tdate, validDate, 'VSHR06')[0].values
                this_field = np.sqrt(this_field1**2 + this_field2**2)
            elif f == 'SHR01':
                this_field1 = get_message(fh, tdate, validDate, 'USHR01')[0].values
                this_field2 = get_message(fh, tdate, validDate, 'VSHR01')[0].values
                this_field = np.sqrt(this_field1**2 + this_field2**2)
            elif f == 'LR75':
                t500,t700 = [x.values for x in get_message(fh, tdate, validDate, 'T_PL', level=[500,700])]
                ht500,ht700 = [x.values for x in get_message(fh, tdate, validDate, 'GHT_PL', level=[500,700])]
                this_field = -(t700-t500)/(ht700-ht500)
            elif f == 'CAPESHEAR':
                this_field1 = get_message(fh, tdate, validDate, 'USHR06')[0].values
                this_field2 = get_message(fh, tdate, validDate, 'VSHR06')[0].values
                shr06 = np.sqrt(this_field1**2 + this_field2**2)
                mlcape = get_message(fh, tdate, validDate, 'MLCAPE')[0].values
                this_field = mlcape * shr06
            elif f == 'STP':
                lcl = get_message(fh, tdate, validDate, 'MLLCL')[0].values
                sbcape = get_message(fh, tdate, validDate, 'SBCAPE')[0].values
                this_field1 = get_message(fh, tdate, validDate, 'USHR01')[0].values
                this_field2 = get_message(fh, tdate, validDate, 'VSHR01')[0].values
                srh01 = np.sqrt(this_field1**2 + this_field2**2)
                sbcinh = get_message(fh, tdate, validDate, 'SBCINH')[0].values

                this_field1 = get_message(fh, tdate, validDate, 'USHR06')[0].values
                this_field2 = get_message(fh, tdate, validDate, 'VSHR06')[0].values
                shr06 = np.sqrt(this_field1**2 + this_field2**2)

                lclterm = ((2000.0-lcl)/1000.0)
                lclterm = np.where(lcl<1000, 1.0, lclterm)
                lclterm = np.where(lcl>2000, 0.0, lclterm)

                shrterm = (shr06/20.0)
                shrterm = np.where(shr06 > 30, 1.5, shrterm)
                shrterm = np.where(shr06 < 12.5, 0.0, shrterm)

                cinterm = ((200+sbcinh)/150.0)
                cinterm = np.where(cinterm>-50, 1.0, cinterm)
                cinterm = np.where(cinterm<-200, 0.0, cinterm)

                this_field = (sbcape/1500.0) * lclterm * (srh01/150.0) * shrterm * cinterm
            elif f in ['T925', 'T850', 'T700', 'T500']:
                level = int(f[1:])
                this_field = get_message(fh, tdate, validDate, "T_PL", level=level)[0].values
            elif f in ['TD925', 'TD850', 'TD700', 'TD500']:
                level = int(f[2:])
                t = get_message(fh, tdate, validDate, "T_PL", level=level)[0].values
                rh = get_message(fh, tdate, validDate, "RH_PL", level=level)[0].values
                import metpy.calc
                from metpy.units import units
                this_field = metpy.calc.dewpoint_from_relative_humidity(t*units('K'),rh*units('percent'))
            elif f in ['U925', 'U850', 'U700', 'U500']:
                level = int(f[1:])
                idx = press_levels.index(level)
                this_field = get_message(fh, tdate, validDate, "U_PL", level=level)[0].values
            elif f in ['V925', 'V850', 'V700', 'V500']:
                level = int(f[1:])
                idx = press_levels.index(level)
                this_field = get_message(fh, tdate, validDate, "V_PL", level=level)[0].values
            elif f in ['UP_HELI_MAX80', 'UP_HELI_MAX120']:
                this_field = fh.variables['UP_HELI_MAX'][0,:]
            elif f in ['UP_HELI_MAX01-80', 'UP_HELI_MAX01-120']:
                this_field = fh.variables['UP_HELI_MAX01'][0,:]
            else:
                this_field = get_message(fh, tdate, validDate, f)[0].values
            print(f, this_field.max())
             
            # use maximum for certain fields, mean for others 
            if f in ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'HAILCAST_DIAM_MAX']:
                if model in ['NSC3km-12sec']: field_interp = upscale(this_field, type='max', maxsize=27)
                else: field_interp = upscale(this_field, type='max', maxsize=27*3)
            elif f in ['UP_HELI_MAX80', 'UP_HELI_MAX01-80']:
                if model in ['NSC3km-12sec']: field_interp = upscale(this_field, type='max', maxsize=53)
                else: field_interp = upscale(this_field, type='max', maxsize=53*3)
            elif f in ['UP_HELI_MAX120', 'UP_HELI_MAX01-120']:
                if model in ['NSC3km-12sec']: field_interp = upscale(this_field, type='max', maxsize=81)
                else: field_interp = upscale(this_field, type='max', maxsize=81*3)
            else:
                if model in ['NSC3km-12sec']: field_interp = upscale(this_field, type='mean', maxsize=26) #should be 27?
                else: field_interp = upscale(this_field, type='mean', maxsize=81)

            upscaled_fields[f].append(field_interp)

        fh.close()
    except Exception as e:
        print(e)
        sys.exit(1)
        continue

if model == 'GEFS': np.savez_compressed('/glade/work/sobash/NSC/%s_%s_mem%d_upscaled'%(sdate.strftime('%Y%m%d%H'),model,mem), a=upscaled_fields)
#else: np.savez_compressed('/glade/work/sobash/NSC/%s_%s_upscaled'%(sdate.strftime('%Y%m%d%H'),model), a=upscaled_fields)
else: np.savez_compressed('/glade/work/ahijevyc/NSC/%s_%s_upscaled'%(sdate.strftime('%Y%m%d%H'),model), a=upscaled_fields)

#plot_field = np.array(upscaled_fields['MUCAPE'])
#print plot_field.shape
#plot_field = np.amax(plot_field, axis=0)

# plotting
plotting = False 
if plotting:
    levels = np.arange(250,5000,250)
    #levels = np.arange(0,100,2.5)
    test = readNCLcm('MPL_Reds')[10:]
    cmap = colors.ListedColormap(test)
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    ##awips.pcolormesh(x81, y81, np.ma.masked_less(u_interp, 100.0), cmap=cmap, norm=norm)
    awips.pcolormesh(x81, y81, plot_field, cmap=cmap, norm=norm)
    #awips.pcolormesh(x81, y81, env['b'], cmap=cmap, norm=norm)
    awips.drawstates()
    awips.drawcountries()
    awips.drawcoastlines()
    plt.savefig('test.png')
