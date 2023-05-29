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
    if model in ['NCARENS', 'NSC3km', 'NSC3km-12sec', 'GEFS', 'HRRR', 'HRRRX']: kernel = np.ones((26,26)) #should this be 27?
    elif model in ['NSC1km']: kernel = np.ones((81,81))

    weights = kernel / float(kernel.size)

    if type == 'mean':
        #field = scipy.ndimage.filters.convolve(field, weights=weights, mode='constant', cval=0.0)
        field = scipy.ndimage.filters.uniform_filter(field, size=maxsize, mode='constant', cval=0.0)
    elif type == 'max':
        field = scipy.ndimage.filters.maximum_filter(field, size=maxsize)
    elif type == 'min':
        field = scipy.ndimage.filters.minimum_filter(field, size=maxsize)

    field_interp = field.flatten()[nngridpts[1]].reshape((65,93))

    return field_interp

def get_closest_gridbox():
    ### find closest 3-km or 1-km grid point to each 80-km grid point
    gpfname = 'nngridpts_80km_%s'%model
    if os.path.exists(gpfname):
        nngridpts = pickle.load(open(gpfname, 'rb'))
    else:
        # INTERPOLATE NARR TO 80KM GRID
        #fig, axes, m  = pickle.load(open('/glade/u/home/wrfrt/rt_ensemble_2018wwe/python_scripts/rt2015_CONUS.pk', 'r'))
        awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)

        grid81 = awips.makegrid(93, 65, returnxy=True)
        x81, y81 = awips(grid81[0], grid81[1])
        #x81 = (x[1:,1:] + x[:-1,:-1])/2.0
        #y81 = (y[1:,1:] + y[:-1,:-1])/2.0

        if model == 'HRRR': f = pygrib.open('/glade/scratch/sobash/HRRR/2019090100/hrrr.t00z.wrfprsf00.grib2')
        lats, lons = f[1].latlons()
        f.close()

        xy = awips(lons.ravel(), lats.ravel())
        tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
        nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
        pickle.dump(nngridpts, open('nngridpts_80km_%s'%model, 'wb'))

    return nngridpts

def upscale_forecast(fhr):
    print(time.ctime(time.time()), 'reading', yyyymmddhh, fhr)

    this_upscaled_fields = dict(upscaled_fields)
  
    ### read in NSSL dataset for this day
    try:
        wrfvalidstr = (tdate + timedelta(hours=fhr)).strftime('%Y-%m-%d_%H_%M_%S')
        wrfvalidstr2 = (tdate + timedelta(hours=fhr)).strftime('%Y-%m-%d_%H:%M:%S')
     
        wrffile = "/glade/scratch/sobash/HRRR/%s/hrrr.t%sz.wrfsfcf%02d.grib2"%(yyyymmddhh,hh,fhr)
        fh = pygrib.open(wrffile)

        if fhr < 2: this_grib_dict = grib_dict_fhr01
        else: this_grib_dict = grib_dict

        # populate dictionary of upscaled fields
        for f in this_upscaled_fields.keys():
            if f == 'SHR06':
                this_field1 = fh[this_grib_dict['USHR6']].values
                this_field2 = fh[this_grib_dict['VSHR6']].values
                this_field = np.sqrt(this_field1**2 + this_field2**2)
            elif f == 'SHR01':
                this_field1 = fh[this_grib_dict['USHR1']].values
                this_field2 = fh[this_grib_dict['VSHR1']].values
                this_field = np.sqrt(this_field1**2 + this_field2**2)
            elif f == 'LR75':
                t500 = fh[this_grib_dict['T500']].values
                t700 = fh[this_grib_dict['T700']].values
                ht500 = fh[this_grib_dict['Z500']].values
                ht700 = fh[this_grib_dict['Z700']].values
                this_field = -(t700-t500)/(ht700-ht500)
            elif f == 'CAPESHEAR':
                ushr = fh[this_grib_dict['USHR6']].values
                vshr = fh[this_grib_dict['VSHR6']].values
                shr06 = np.sqrt(ushr**2 + vshr**2)
                mlcape = fh[this_grib_dict['MLCAPE']].values
                this_field = mlcape * shr06
            elif f == 'STP':
                lcl = fh[this_grib_dict['SBLCL']].values
                sbcape = fh[this_grib_dict['SBCAPE']].values
                srh01 = fh[this_grib_dict['SRH01']].values
                sbcinh = fh[this_grib_dict['SBCINH']].values

                ushr = fh[this_grib_dict['USHR6']].values
                vshr = fh[this_grib_dict['VSHR6']].values
                shr06 = np.sqrt(ushr**2 + vshr**2)

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
            elif f in ['UP_HELI_MAX80', 'UP_HELI_MAX120']:
                this_field = fh[this_grib_dict['UP_HELI_MAX']].values
            else:
                this_field = fh[this_grib_dict[f]].values

            #print(f, this_field.max())

            # use maximum for certain fields, mean for others 
            if f in ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'HAILCAST_DIAM_MAX']:
                if model in ['NSC3km-12sec', 'HRRR', 'HRRRX']: field_interp = upscale(this_field, type='max', maxsize=27)
                else: field_interp = upscale(this_field, type='max', maxsize=27*3)
            elif f in ['UP_HELI_MAX80']:
                if model in ['NSC3km-12sec', 'HRRR', 'HRRRX']: field_interp = upscale(this_field, type='max', maxsize=53)
                else: field_interp = upscale(this_field, type='max', maxsize=53*3)
            elif f in ['UP_HELI_MAX120']:
                if model in ['NSC3km-12sec', 'HRRR', 'HRRRX']: field_interp = upscale(this_field, type='max', maxsize=81)
                else: field_interp = upscale(this_field, type='max', maxsize=81*3)
            else:
                if model in ['NSC3km-12sec', 'HRRR', 'HRRRX']: field_interp = upscale(this_field, type='mean', maxsize=26) #should be 27?
                else: field_interp = upscale(this_field, type='mean', maxsize=81)

            this_upscaled_fields[f] = field_interp

        fh.close()
    except Exception as e:
        print(e)
        return {fhr: this_upscaled_fields}

    return {fhr: this_upscaled_fields}

### read in date from command line
model = 'HRRR'

sdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
edate = sdate + timedelta(hours=24)
tdate = sdate
    
yyyy = tdate.strftime('%Y')
yyyymmddhh = tdate.strftime('%Y%m%d%H')
yyyymmdd = tdate.strftime('%Y%m%d')
yymmdd = tdate.strftime('%y%m%d')
hh = tdate.strftime('%H')

if sdate >= datetime(2020,12,2,12,0,0):
    # FYI - trained with 90-0 mb above ground for MLCAPE and MLCIN, fixed on 5/10 (prev code was using incorrect field for fhr>1 mlcinh and wrong mlcape field)
    # these grib numbers are for HRRR sfc files from V4 - beginning 12z 2020 2 December
    grib_dict = { 'UP_HELI_MAX':45, 'UP_HELI_MIN':46, 'UP_HELI_MAX03':49, 'GRPL_MAX':56, 'W_UP_MAX':38, 'W_DN_MAX':39,\
              #'SBCAPE':108, 'SBCINH':109, 'MLCINH':152, 'MLCAPE':151, 'MUCAPE':157, 'SRH01':135, 'SRH03':134, 'T2':71, 'TD2':74, 'PSFC':62, \
              'SBCAPE':108, 'SBCINH':109, 'MLCINH':156, 'MLCAPE':155, 'MUCAPE':157, 'SRH01':135, 'SRH03':134, 'T2':71, 'TD2':74, 'PSFC':62, \
              'USHR6':140, 'VSHR6':141, 'USHR1':138, 'VSHR1':139, 'PREC_ACC_NC':90, 'WSPD10MAX':79, 'U10MAX': 80, 'V10MAX':81, \
              'SBLCL':154, 'CREF':1, 'REFL1KM':6, 'REFL4KM':7, 'HGT0C':142, 'RVORT1':52, \
              'UP_HELI_MAX02':47, 'LTG1':57, 'LTG2':58, 'LTG3':59, 'HAIL_SFC':55, \
              'U500':17, 'V500':18, 'Z500':14, 'T500':15, 'TD500':16, \
              'U700':23, 'V700':24, 'Z700':19, 'T700':20, 'TD700':21, \
              'U850':28, 'V850':29, 'Z850':25, 'T850':26, 'TD850':27, \
              'U925':32, 'V925':33, 'Z925':-999, 'T925':30, 'TD925':31 }

    #if grib number > 89 then subtract 3 (except accumulated precip)
    grib_dict_fhr01 = { 'UP_HELI_MAX':45, 'UP_HELI_MIN':46, 'UP_HELI_MAX03':49, 'GRPL_MAX':56, 'W_UP_MAX':38, 'W_DN_MAX':39,\
              #'SBCAPE':105, 'SBCINH':106, 'MLCINH':152, 'MLCAPE':148, 'MUCAPE':154, 'SRH01':132, 'SRH03':131, 'T2':71, 'TD2':74, 'PSFC':62, \
              'SBCAPE':105, 'SBCINH':106, 'MLCINH':153, 'MLCAPE':152, 'MUCAPE':154, 'SRH01':132, 'SRH03':131, 'T2':71, 'TD2':74, 'PSFC':62, \
              'USHR6':137, 'VSHR6':138, 'USHR1':135, 'VSHR1':136, 'PREC_ACC_NC':84, 'WSPD10MAX':79, 'U10MAX': 80, 'V10MAX':81, \
              'SBLCL':151, 'CREF':1, 'REFL1KM':6, 'REFL4KM':7, 'HGT0C':139, 'RVORT1':52, \
              'UP_HELI_MAX02':47, 'LTG1':57, 'LTG2':58, 'LTG3':59, 'HAIL_SFC':55, \
              'U500':17, 'V500':18, 'Z500':14, 'T500':15, 'TD500':16, \
              'U700':23, 'V700':24, 'Z700':19, 'T700':20, 'TD700':21, \
              'U850':28, 'V850':29, 'Z850':25, 'T850':26, 'TD850':27, \
              'U925':32, 'V925':33, 'Z925':-999, 'T925':30, 'TD925':31 }

elif sdate >= datetime(2018,7,12,12,0,0):
    # these grib numbers are for HRRR sfc files from V3 - beginning 12z 2018 12 July
    # if forecast hour > 2
    # LTG1, LTG2, and HAIL_SFC not available in HRRRv3
    grib_dict = { 'UP_HELI_MAX':43, 'UP_HELI_MIN':44, 'UP_HELI_MAX03':47, 'GRPL_MAX':53, 'W_UP_MAX':36, 'W_DN_MAX':37,\
                'SBCAPE':98, 'SBCINH':99, 'MLCINH':136, 'MLCAPE':135, 'MUCAPE':141, 'SRH01':119, 'SRH03':118, 'T2':66, 'TD2':69, 'PSFC':57, \
                'USHR6':124, 'VSHR6':125, 'USHR1':122, 'VSHR1':123, 'PREC_ACC_NC':84, 'WSPD10MAX':73, 'U10MAX': 74, 'V10MAX':75, \
                'SBLCL':138, 'CREF':1, 'REFL1KM':5, 'REFL4KM':6, 'HGT0C':126, 'RVORT1':50, \
                'UP_HELI_MAX02':45, 'LTG3': 54, \
                'U500':16, 'V500':17, 'Z500':13, 'T500':14, 'TD500':15, \
                'U700':21, 'V700':22, 'Z700':18, 'T700':19, 'TD700':20, \
                'U850':26, 'V850':27, 'Z850':23, 'T850':24, 'TD850':25, \
                'U925':30, 'V925':31, 'Z925':-999, 'T925':28, 'TD925':29 }

    #if grib number > 83 then subtract 3 (except accumulated precip)
    grib_dict_fhr01 = { 'UP_HELI_MAX':43, 'UP_HELI_MIN':44, 'UP_HELI_MAX03':47, 'GRPL_MAX':53, 'W_UP_MAX':36, 'W_DN_MAX':37,\
                'SBCAPE':95, 'SBCINH':96, 'MLCINH':133, 'MLCAPE':132, 'MUCAPE':138, 'SRH01':116, 'SRH03':115, 'T2':66, 'TD2':69, 'PSFC':57, \
                'USHR6':121, 'VSHR6':122, 'USHR1':119, 'VSHR1':120, 'PREC_ACC_NC':78, 'WSPD10MAX':73, 'U10MAX': 74, 'V10MAX':75, \
                'SBLCL':135, 'CREF':1, 'REFL1KM':5, 'REFL4KM':6, 'HGT0C':123, 'RVORT1':50, \
                'UP_HELI_MAX02':45, 'LTG3': 54, \
                'U500':16, 'V500':17, 'Z500':13, 'T500':14, 'TD500':15, \
                'U700':21, 'V700':22, 'Z700':18, 'T700':19, 'TD700':20, \
                'U850':26, 'V850':27, 'Z850':23, 'T850':24, 'TD850':25, \
                'U925':30, 'V925':31, 'Z925':-999, 'T925':28, 'TD925':29 }
else:
    sys.exit('Grib IDs not available for HRRRv2 data')

# these fields will be written out 
upscaled_fields = { 'SBCAPE':[], 'MLCAPE':[], 'SBCINH':[], 'MLCINH':[], 'UP_HELI_MAX':[], 'UP_HELI_MAX03':[], 'W_UP_MAX':[], 'W_DN_MAX':[],\
                    'SRH01':[], 'SRH03':[], 'SHR01':[], 'SHR06':[], 'CAPESHEAR':[], 'T2':[], 'TD2':[], 'PSFC':[], 'PREC_ACC_NC':[], 'WSPD10MAX':[], \
                    'UP_HELI_MAX80':[], 'UP_HELI_MAX120':[], 'SBLCL':[], 'STP':[], 'U500':[], 'V500':[], 'T500':[], 'TD500':[], 'U700':[],\
                    'V700':[], 'T700':[], 'TD700':[], 'U850':[], 'V850':[], 'T850':[], 'TD850':[], 'U925':[], 'V925':[], 'T925':[], 'TD925':[], \
                    'LR75':[], 'CREF':[], 'GRPL_MAX':[], 'HGT0C':[], 'RVORT1':[], 'MUCAPE':[] }

# get closest grid boxes
print('finding closest grid points')
nngridpts = get_closest_gridbox()

# use multiprocess to run different forecast hours in parallel
print('running upscaling in parallel')
#nfhr      = 37
if sdate.hour in [0,6,12,18]: nfhr      = 49 #for HRRRv4
else: nfhr = 19

nprocs    = 6
chunksize = int(math.ceil(nfhr / float(nprocs)))
pool      = multiprocessing.Pool(processes=nprocs)
data      = pool.map(upscale_forecast, range(nfhr), chunksize)
pool.close()

# results returned as list of dicts?
# merge dictionaries
combined = {}
for d in data: combined.update(d)
for f in upscaled_fields.keys():
    for fhr in range(nfhr):
        if (len(combined[fhr][f]) > 0): upscaled_fields[f].append(combined[fhr][f])
        else: upscaled_fields[f].append(np.ones((65,93))*np.nan)

# save file
np.savez_compressed('/glade/work/sobash/NSC/%s_%s_upscaled'%(sdate.strftime('%Y%m%d%H'),model), a=upscaled_fields)

# plotting
plotting = False 
if plotting:
    plot_field = np.array(upscaled_fields['SBCAPE'])
    print(plot_field.shape)
    plot_field = np.amax(plot_field, axis=0)
    
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
