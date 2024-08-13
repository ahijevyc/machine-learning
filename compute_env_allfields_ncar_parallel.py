#!/usr/bin/env python

from netCDF4 import Dataset
from ml_functions import upscale
import gzip
import numpy as np
from datetime import *
import pygrib
import time, os, sys
import pickle
#from fieldinfo import readNCLcm
import multiprocessing
from mpl_toolkits.basemap import Basemap
from scipy import spatial

### THIS CODE EVOLVED FROM CODE WITHIN /glade/u/home/sobash/NSC_scripts
### TO UPSCALE 3-KM CAM DATA TO AN 80-KM GRID

def get_message(fh, tdate, validDate, f, level=None):
    select_keywords = ncar_grb(f, tdate=tdate, level=level)
    # TODO figure out how to select validDate when field has None (like 1-h max UH)
    try:
        selected_messages = fh.select(analDate=tdate, **select_keywords)
        #print('selected', len(selected_messages), 'messages')
    except:
        print("pygrib.select failed with", select_keywords, end=" ")
        selected_messages = None
    return selected_messages

def is2016HWT(tdate):
    # verified this with COMPOSITE_REFL_10CM
    return datetime(2016,5,2,0) <= tdate < datetime(2016,6,4,0)

def is2017HWT(tdate):
    # verified this with COMPOSITE_REFL_10CM
    return datetime(2017,4,19,0) <= tdate < datetime(2017,6,3,0)

def ncar_grb(f, tdate=datetime(2017,12,30,0), level=None):
    #print("ncar_grb():", f, tdate)
    grb_dict = {
            "COMPOSITE_REFL_10CM" : {"shortName" : "refc"}, 
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

    if is2016HWT(tdate) or is2017HWT(tdate):
        #grb_dict["COMPOSITE_REFL_10CM"] = {"parameterName" : "refc", "units" : "dB"} # looks like this in wgrib2
        grb_dict["COMPOSITE_REFL_10CM"] = {"parameterName" : "5"} # but looks like this in pygrib.

    if datetime(2015,4,1,0) <= tdate < datetime(2015,9,1,0):
        grb_dict["COMPOSITE_REFL_10CM"] = {"indicatorOfParameter" : 212} 
        grb_dict["MLCAPE"] = {"indicatorOfParameter": 157, "typeOfLevel" : "pressureFromGroundLayer", "bottomLevel": 0, "topLevel": 90}
        grb_dict["MLCINH"] = {"indicatorOfParameter": 156, "typeOfLevel" : "pressureFromGroundLayer", "bottomLevel": 0, "topLevel": 90}
        grb_dict["MUCAPE"] = {"indicatorOfParameter": 157, "typeOfLevel" : "pressureFromGroundLayer", "bottomLevel": 0, "topLevel": 2147483647}
        grb_dict["PSFC"]  = {"indicatorOfParameter": 1, "typeOfLevel" : "surface", "units" : "Pa"} # This surface pressure has a mean around 85000 Pa, while netCDF surface pressure available starting 2015091600 has a mean around 930. Obviously there is something different.  
        grb_dict["SBCAPE"] = {"indicatorOfParameter": 157, "typeOfLevel" : "surface"}
        grb_dict["SBCINH"] = {"indicatorOfParameter": 156, "typeOfLevel" : "surface"}
        grb_dict["SRH01"]  = {"parameterName": "190", "typeOfLevel" : "heightAboveGroundLayer",  "bottomLevel": 0, "topLevel": 10 }
        grb_dict["SRH03"]  = {"parameterName": "190", "typeOfLevel" : "heightAboveGroundLayer",  "bottomLevel": 0, "topLevel": 30 }
    if tdate < datetime(2016,5,3,0):
        grb_dict["MLLCL"] = None # MLLCL only starts 2016050200, and it is bad on 2016050200, so start on 2016050300.

    return grb_dict[f]


# INTERPOLATE NARR TO 80KM GRID
#fig, axes, m  = pickle.load(open('/glade/u/home/wrfrt/rt_ensemble_2018wwe/python_scripts/rt2015_CONUS.pk', 'r'))
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)

grid81 = awips.makegrid(93, 65, returnxy=True)
x81, y81 = awips(grid81[0], grid81[1])
#x81 = (x[1:,1:] + x[:-1,:-1])/2.0
#y81 = (y[1:,1:] + y[:-1,:-1])/2.0
mask = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
mask = mask.reshape((65,93))

sdate = datetime.strptime(sys.argv[1], '%Y%m%d%H')
edate = sdate + timedelta(hours=24)
tdate = sdate

model = 'NCARENS'
mem   = 4
debug=False
ofile = '/glade/work/sobash/NSC/%s_%s_mem%d_upscaled.npz'%(sdate.strftime('%Y%m%d%H'),model,mem )
if os.path.exists(ofile):
    print(ofile, "exists. exiting cleanly.")
    sys.exit(0)

if model == 'NSC1km': f = Dataset('/glade/p/mmm/parc/sobash/NSC/1KM_WRF_POST/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
if model == 'NSC3km-12sec': f = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/2011062500/diags_d01_2011-06-25_00_00_00.nc', 'r')
if model == 'GEFS': f = Dataset('/glade/scratch/sobash/ncar_ens/gefs_ics/2017042500/wrf_rundir/ens_1/diags_d02.2017-04-25_00:00:00.nc', 'r')
if model == 'NCARENS': f = pygrib.open('/glade/collections/rda/data/ds300.0/2016/20160701/ncar_3km_2016070100_mem4_f042.grb2')
lats, lons = f[1].latlons()
f.close()

# find closest 3-km or 1-km grid point to each 80-km grid point
gpfname = 'data/nngridpts_80km_%s'%model
if os.path.exists(gpfname):
    nngridpts = pickle.load(open(gpfname, 'rb'))
else:
    print('finding closest grid points')
    xy = awips(lons.ravel(), lats.ravel())
    tree = spatial.KDTree(list(zip(xy[0].ravel(),xy[1].ravel())))
    nngridpts = tree.query(list(zip(x81.ravel(),y81.ravel())))
    pickle.dump(nngridpts, open(gpfname, 'wb'))

ncar_grib_dict = { 44: 'UP_HELI_MAX', 45: 'UP_HELI_MIN', 48: 'UP_HELI_MAX03', 54: 'GRPL_MAX', 39: 'W_UP_MAX', 40: 'W_DN_MAX', 52: 'HAIL_MAX2D', 53: 'HAIL_MAXK1',\
                   51: 'REL_VORT_MAX01', 75: 'WSPD10MAX', 2: 'COMPOSITE_REFL_10CM', 42: 'REFD_MAX', \
                   115: 'UBSHR1', 116: 'VBSHR1', 117: 'UBSHR6', 118: 'VBSHR6', 99: 'PWAT', 124: 'SBLCL', 127: 'MLLCL', 97: 'SBCAPE', 98: 'SBCINH', \
                   128: 'MUCAPE', 129: 'MUCINH', 125: 'MLCAPE', 112: 'SRH01', 111: 'SRH03', 70: 'T2', 72: 'TD2', 62: 'PSFC' }

upscaled_fields = { 'UP_HELI_MAX':[], 'UP_HELI_MAX03':[], 'UP_HELI_MAX01':[], 'W_UP_MAX':[], 'W_DN_MAX':[], 'WSPD10MAX':[], 'STP':[], 'LR75':[], 'CAPESHEAR':[],
        'MUCAPE':[], 'SBCAPE':[], 'SBCINH':[], 'MLCINH':[], 'MLLCL':[], 'SHR01':[], 'SHR06': [], 'SRH01':[], 'SRH03':[], 'T2':[], 'TD2':[], 'PSFC':[], 'RAINNC_1H':[], \
           'HAILCAST_DIAM_MAX':[], \
           'COMPOSITE_REFL_10CM':[], 'REFD_MAX':[], \
           'T925':[], 'T850':[], 'T700':[], 'T500':[], 'TD925':[], 'TD850':[], 'TD700':[], 'TD500':[], 'U925':[], 'U850':[], 'U700':[], 'U500':[], 'V925':[], 'V850':[], 'V700':[], 'V500':[], \
           'UP_HELI_MAX80':[], 'UP_HELI_MAX120':[], 'UP_HELI_MAX01-120':[] }
press_levels = [1000,925,850,700,600,500,400,300,250,200,150,100]

if False: 
    # Aside from Spring HWTs, these pressure-level fields are not in NCAR Ensemble archive.
    # Present after Aug 2017.
    del(upscaled_fields["LR75"]) # No Lapse rate before 20170901. No derived field or pressure-level temperature components with which to derive it.
    del(upscaled_fields["T850"])
    del(upscaled_fields["T700"])
    del(upscaled_fields["T500"])
    del(upscaled_fields["TD850"])
    del(upscaled_fields["TD700"])
    del(upscaled_fields["U850"])
    del(upscaled_fields["U700"])
    del(upscaled_fields["U500"])
    del(upscaled_fields["V850"])
    del(upscaled_fields["V700"])
    del(upscaled_fields["V500"])

    # Missing even after Aug 2017
    del(upscaled_fields["T925"])
    del(upscaled_fields["TD500"])
    del(upscaled_fields["TD925"])
    del(upscaled_fields["U925"])
    del(upscaled_fields["V925"])


def show_some_grib_keys(fh):
    for msg in fh:
        print(msg.validDate, end=" ")
        for k in ["indicatorOfParameter", "parameterName", "parameterUnits", "indicatorOfTypeOfLevel", "pressUnits",
                "bottomLevel", "topLevel", "name"]:
            if msg.valid_key(k):
                print(f"{k}={msg._get_key(k)}", end=" ")
        print(msg.typeOfLevel, end=" ")
        print(msg.level, end=" ")
        print(msg.shortName, end=" ")
        print(msg.units)

def read_ncgz(ncf, f, units=None):
    if debug:
        print("read_ncgz(): netCDF file", ncf)
    with gzip.open(ncf) as gz:
        with Dataset('dummy', mode='r', memory=gz.read()) as nc:
            # Check for variable existence before checking units.
            if f not in nc.variables:
                return None
            if units:
                if nc.variables[f].units != units:
                    print("read_ncgz(): expected units",units,"for",f,"got",nc.variables[f].units)
                    sys.exit(1)
            this_field = nc.variables[f][:]
    return this_field



def get_this_field(f, fh, tdate, fhr, mem=1, fh_previous_hour=None, debug=False):
    yyyy = tdate.strftime('%Y')
    yyyymmddhh = tdate.strftime('%Y%m%d%H')
    yyyymmdd = tdate.strftime('%Y%m%d')
    yymmdd = tdate.strftime('%y%m%d')
    validDate = tdate + timedelta(hours=fhr)
    ncf = "/glade/collections/rda/data/ds300.0/%s/%s/diags_d02_%s_mem_%d_f%03d.nc.gz"%(yyyy,yyyymmdd,yyyymmddhh,mem,fhr)
    #print("get_this_field(): netCDF file", ncf)

    if debug:
        show_some_grib_keys(fh)

    if f == 'SHR06':
        this_field1 = get_message(fh, tdate, validDate, 'USHR06')
        if this_field1 is None: return None
        this_field1 = this_field1[0].values
        this_field2 = get_message(fh, tdate, validDate, 'VSHR06')[0].values
        this_field = np.sqrt(this_field1**2 + this_field2**2)
    elif f == 'SHR01':
        if get_message(fh, tdate, validDate, 'USHR01') is None: return None
        this_field1 = get_message(fh, tdate, validDate, 'USHR01')[0].values
        this_field2 = get_message(fh, tdate, validDate, 'VSHR01')[0].values
        this_field = np.sqrt(this_field1**2 + this_field2**2)
    elif f == 'LR75':
        if get_message(fh, tdate, validDate, 'T_PL', level=[500,700]) is None: return None
        t500,t700 = [x.values for x in get_message(fh, tdate, validDate, 'T_PL', level=[500,700])]
        ht500,ht700 = [x.values for x in get_message(fh, tdate, validDate, 'GHT_PL', level=[500,700])]
        this_field = -(t700-t500)/(ht700-ht500)
    elif f == 'CAPESHEAR':
        if get_message(fh, tdate, validDate, 'USHR06') is None: return None
        this_field1 = get_message(fh, tdate, validDate, 'USHR06')[0].values
        this_field2 = get_message(fh, tdate, validDate, 'VSHR06')[0].values
        shr06 = np.sqrt(this_field1**2 + this_field2**2)
        mlcape = get_message(fh, tdate, validDate, 'MLCAPE')
        if mlcape is None: return None
        mlcape = mlcape[0].values
        this_field = mlcape * shr06
    elif f == 'STP':
        lcl = get_message(fh, tdate, validDate, 'MLLCL')
        if lcl is None: return None
        lcl = lcl[0].values
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
        this_field = get_message(fh, tdate, validDate, "T_PL", level=level)
        if this_field is None: return None
        this_field = this_field[0].values
    elif f in ['TD925', 'TD850', 'TD700', 'TD500']:
        level = int(f[2:])
        t = get_message(fh, tdate, validDate, "T_PL", level=level)
        if t is None: return None
        t = t[0].values
        rh = get_message(fh, tdate, validDate, "RH_PL", level=level)
        if rh is None: return None
        rh = rh[0].values
        import metpy.calc
        from metpy.units import units
        this_field = metpy.calc.dewpoint_from_relative_humidity(t*units('K'),rh*units('percent'))
    elif f in ['U925', 'U850', 'U700', 'U500']:
        level = int(f[1:])
        idx = press_levels.index(level)
        this_field = get_message(fh, tdate, validDate, "U_PL", level=level)
        if this_field is None: return None
        this_field = this_field[0].values
    elif f == "RAINNC_1H":
        # If you use PRECIP_ACC_NC from grib file, it changes from 1-hr to accumulated precip during HWT.
        previous_hour_ncf = "/glade/collections/rda/data/ds300.0/%s/%s/diags_d02_%s_mem_%d_f%03d.nc.gz"%(yyyy,yyyymmdd,yyyymmddhh,mem,max([0,fhr-1]))
        previous_hour_field = read_ncgz(previous_hour_ncf, "RAINNC")
        this_field = read_ncgz(ncf, "RAINNC")
        this_field = this_field - previous_hour_field
    elif tdate < datetime(2015,9,1) and f == 'PSFC':
        this_message = get_message(fh, tdate, validDate, f)[0]
        assert this_message.units == "Pa"
        this_field = this_message.values
        if this_field.max() < 1100:
            print("max value < 1100. expected units of Pa")
            sys.exit(1)
    elif f in ['V925', 'V850', 'V700', 'V500']:
        level = int(f[1:])
        idx = press_levels.index(level)
        this_field = get_message(fh, tdate, validDate, "V_PL", level=level)
        if this_field is None: return None
        this_field = this_field[0].values
    elif tdate < datetime(2015,4,21,0) and f in ["T2"]:
        #print("2-m T is bad in grib file before 20150421")
        # it has range from -32 to a fraction above zero with units Kelvin
        this_field = read_ncgz(ncf, f)
    elif tdate <= datetime(2016,5,1) and f in ['UP_HELI_MAX', 'UP_HELI_MAX03', 'WSPD10MAX', 'W_DN_MAX', 'W_UP_MAX']:
        #print("trying netCDF")
        this_field = read_ncgz(ncf, f)
    elif f in ['UP_HELI_MAX80', 'UP_HELI_MAX120']:
        this_field = read_ncgz(ncf, 'UP_HELI_MAX')
    elif f in ['UP_HELI_MAX01-80', 'UP_HELI_MAX01-120']:
        this_field = read_ncgz(ncf, 'UP_HELI_MAX01')
    elif f == 'PSFC':
        # netCDF PSFC only starts 2015091600.
        this_field = read_ncgz(ncf, f, units="Pa")
    elif f in ['HAILCAST_DIAM_MAX', 'REFD_MAX', 'UP_HELI_MAX01', 'UP_HELI_MAX03', 'W_UP_MAX', 'W_DN_MAX']:
        this_field = read_ncgz(ncf, f)
    else:
        this_field = get_message(fh, tdate, validDate, f)
        if this_field is None: return None
        this_field = this_field[0].values


    return this_field

def get_ungz_fh(wrffile):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=True) 
    ungz = gzip.open(wrffile)
    tmp.write(ungz.read())
    tmp.flush() # TODO necessary?
    tmp.seek(0) # TODO necessary?
    fh = pygrib.open(tmp.name)
    return fh


def upscale_forecast(fhr): 
    yyyy = tdate.strftime('%Y')
    yyyymmddhh = tdate.strftime('%Y%m%d%H')
    yyyymmdd = tdate.strftime('%Y%m%d')
    yymmdd = tdate.strftime('%y%m%d')
    print(time.ctime(time.time()), 'reading', yyyymmddhh, fhr)
        
    this_upscaled_fields = upscaled_fields.copy()


    ds300="/glade/collections/rda/data/ds300.0/%s/%s/"%(yyyy,yyyymmdd)
    if tdate >= datetime(2015,9,1):
        wrffile = ds300+"ncar_3km_%s_mem%d_f%03d.grb2"%(yyyymmddhh,mem,fhr)
        wrffile_previous_hour = ds300+"ncar_3km_%s_mem%d_f%03d.grb2"%(yyyymmddhh,mem,max([0,fhr-1]))
    else:
        wrffile = ds300+"ncar_3km_%s_mem%d_f%03d.grb.gz"%(yyyymmddhh,mem,fhr)
        wrffile_previous_hour = ds300+"ncar_3km_%s_mem%d_f%03d.grb.gz"%(yyyymmddhh,mem,max([0,fhr-1]))
   
    print("grib file", wrffile)
    if wrffile[-3:] == ".gz":
        fh = get_ungz_fh(wrffile)
        fh_previous_hour = get_ungz_fh(wrffile_previous_hour)

    else:
        fh = pygrib.open(wrffile)
        fh_previous_hour = pygrib.open(wrffile_previous_hour)

      
    # populate dictionary of upscaled fields
    for f in this_upscaled_fields.keys():
        print('    {:20s}'.format(f), end="")
        #print(time.ctime(time.time()), f)

        print('getting f{:02d}'.format(fhr), '...', end=" ")
        this_field = get_this_field(f, fh, tdate, fhr, mem=mem, fh_previous_hour = fh_previous_hour, debug=debug)
        if this_field is None:
            print('None')
            continue
        print('done.', end=" ")
        if debug:
            print(f, "before upscale", np.nanmin(this_field), np.nanmedian(this_field), np.nanmean(this_field), np.nanmax(this_field))


        # use maximum for certain fields, mean for others 
        if f in ['UP_HELI_MAX', 'UP_HELI_MAX03', 'UP_HELI_MAX01', 'W_UP_MAX', 'W_DN_MAX', 'WSPD10MAX', 'HAILCAST_DIAM_MAX']:
            field_interp = upscale(this_field, nngridpts, type='max', maxsize=27)
        elif f in ['UP_HELI_MAX80', 'UP_HELI_MAX01-80']:
            field_interp = upscale(this_field, nngridpts, type='max', maxsize=53)
        elif f in ['UP_HELI_MAX120', 'UP_HELI_MAX01-120']:
            field_interp = upscale(this_field, nngridpts, type='max', maxsize=81) # suppose I could use maxsize=80, but I don't know how even filter size works.
        else:
            field_interp = upscale(this_field, nngridpts, type='mean', maxsize=81)


        print(f"upscaled {np.nanmin(field_interp[mask]):10.3f} {np.nanmedian(field_interp[mask]):10.3f} {np.nanmean(field_interp[mask]):10.3f} {np.nanmax(field_interp[mask]):10.3f}")
        this_upscaled_fields[f] = field_interp

    return {fhr: this_upscaled_fields}

    fh.close()
print('running upscaling in parallel')
nfhr      = 37
nprocs    = 6
chunksize = int(np.ceil(nfhr / float(nprocs)))
pool      = multiprocessing.Pool(processes=nprocs)
fhrs      = range(0,nfhr) # random_forest_preprocess_gridded.py assume fhrs starts at 0.
data      = pool.map(upscale_forecast, fhrs, chunksize)
pool.close()

# merge dictionaries
combined = {}
for d in data: combined.update(d)
for f in upscaled_fields.keys():
    for fhr in fhrs:
        upscaled_fields[f].append(combined[fhr][f])

# this script produces files that are slightly larger than the original files - ~2KB - not sure why..
np.savez_compressed(ofile, a=upscaled_fields)


# plotting
plotting = False
if plotting:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    plot_field = np.array(upscaled_fields['MLLCL'])
    plot_field = plot_field * mask
    print( plot_field.shape)
    plot_field = np.amax(plot_field, axis=0)
    levels = np.arange(69200,98400,1600)
    levels = np.linspace(plot_field.min(), plot_field.max(), 25)

    test = readNCLcm('MPL_gist_ncar')[3:]
    cmap = colors.ListedColormap(test)
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    ##awips.pcolormesh(x81, y81, np.ma.masked_less(u_interp, 100.0), cmap=cmap, norm=norm)
    pc = awips.pcolormesh(x81, y81, plot_field, cmap=cmap, norm=norm)
    #awips.pcolormesh(x81, y81, env['b'], cmap=cmap, norm=norm)
    awips.drawstates()
    awips.drawcountries()
    awips.drawcoastlines()
    pc.get_figure().colorbar(pc)
    plt.savefig('test.png')
