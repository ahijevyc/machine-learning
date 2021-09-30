import pdb
import numpy as np
import ncepgrib2
import time, os, sys

nx, ny = 93, 65

ifile = "/glade/p/mmm/parc/ahijevyc/NSC/2015043000_NCARENS_mem1_upscaled.npz"
data = np.load(ifile, allow_pickle=True)
upscaled_fields = data['a'].item()
fhrs  = range(len(upscaled_fields['MUCAPE']))

year   = os.path.basename(ifile)[0:4]
month  = os.path.basename(ifile)[4:6]
day    = os.path.basename(ifile)[6:8]
hour   = os.path.basename(ifile)[8:10]
minute = 0
second = 0

# Section 1 - Identification Section
discipline = 0 # 0 for meteorological
centreid = 7 # 7=US National Weather Service, National Centres for Environmental Prediction (NCEP); 60=United States National Centre for Atmospheric Research (NCAR)
subcentreid = 0
grbMasterTablesV = 2
grbLocalTablesV = 1 
sig_reftime = 1 # 1 for start of forecast
status = 2 # production status of data: 2 = research
datatype = 1 # type of data: 1 = forecast products
idsect = [centreid, subcentreid, grbMasterTablesV, grbLocalTablesV, sig_reftime, year, month, day, hour, minute, second, status, datatype]

# Section 3 - Grid Definition Section
src_griddef = 0 #Tried specifying 211 but got seg fault
npts = nx * ny
noct = 0 # =0 for regular grid
opt =  0 # there is no appended list
gdtn = 30 # 30=Lambert Conformal
gdsinfo = [src_griddef, npts, noct, opt, gdtn]
e_shape = 6
e_sf = 0
e_sv = 0
ose_sf_major = 0
earthRmajor = 0
ose_sf_minor = 0
earthRminor = 0
la1=12190000
lo1=226541000
rcflag = 8 
LaD = 25000000 # latitude where dx and dy are specified
LoV = 265000000
dx, dy = 81270500, 81270500
proj_centre_flag = 0
scanning_mode = 64 # 01000000
Latin1 = 25000000
Latin2 = 25000000
LatSP = 0
LonSP = 0
gdtmpl = [e_shape, e_sf, e_sv, ose_sf_major, earthRmajor, ose_sf_minor, earthRminor, nx, ny, la1, lo1, rcflag, 
        LaD, LoV, dx, dy, proj_centre_flag, scanning_mode, Latin1, Latin2, LatSP, LonSP]

# Section 4 - Product Definition Section
pdtnum = 0 # 0=Analysis or forecast at a horizontal level or in a horizontal layer at a point in time.  9=Probability forecasts at a horizontal level or in a horizontal layer in a continuous or non-continuous time interval

parameter_category = 7 # thermo stability=7
parameter_num = 6 # CAPE=6
generating_process = 2 # 2=Forecast, 5=Probability Forecast
backgrd_generating_process = 0
analys_generating_process = 116 # 116=WRF-EM model, generic resolution (Used in various runs) EM - Eulerian Mass-core (NCAR - aka Advanced Research WRF)
obs_data_cutoff_hours = 0
obs_data_cutoff_minutes = 0
time_range_unit_indicator = 1 # 1=Hour
fhr = 0
fixed_sfc_type1 = 108 # 108=Level at Specified Pressure Difference from Ground to Level
fixed_sfc_scale_factor1 = 0
fixed_sfc_scaled_value1 = 9000
fixed_sfc_type2 = 108 # 108=Level at Specified Pressure Difference from Ground to Level
fixed_sfc_scale_factor2 = 0
fixed_sfc_scaled_value2 = 0
pdtmpl = [parameter_category,parameter_num,generating_process,backgrd_generating_process,
    analys_generating_process,obs_data_cutoff_hours,obs_data_cutoff_minutes,time_range_unit_indicator,fhr,
    fixed_sfc_type1,fixed_sfc_scale_factor1,fixed_sfc_scaled_value1,
    fixed_sfc_type2,fixed_sfc_scale_factor2,fixed_sfc_scaled_value2]

# Section 5 - Data Representation Section
drtnum=0 # 0=simple packing (4=ieee floating point and 40=JPEG2000 compression aren't implemented) 
field = np.array(upscaled_fields['MUCAPE'], dtype=np.float64)
precision = 1.
nvalues = 1 + np.ceil( (field.max()-field.min()) / (2 * precision))
DRTnbits = np.ceil(np.log2(nvalues))
DRTref = 1
#https://www.nws.noaa.gov/mdl/synop/gmos/binaryscaling.php 
DRTbinary_scale_factor = 0 # -1 = twice the precision, 0 = unchanged, 1 = half the precision, 2=1/4 the precision 
DRTdecimal_scale_factor = 0 # -1 = tenth the precision, 0=unchanged, 1 = 10x the precision
DRTorigType = 0 # floating point=0
drtmpl = [DRTref,DRTbinary_scale_factor,DRTdecimal_scale_factor,DRTnbits,DRTorigType]
ofile = "test.grb"
fh = open(ofile, "wb")
# Tried writing global GRB section once (skipping with subsequent messages) but got error 'addgrid must be called before addfield'
for fhr in fhrs:
    field1 = field[fhr]
    grbo = ncepgrib2.Grib2Encode(discipline, idsect)
    grbo.addgrid(gdsinfo, gdtmpl)
    pdtmpl[8] = fhr
    grbo.addfield(pdtnum, pdtmpl, drtnum, drtmpl, field1)
    grbo.end()
    fh.write(grbo.msg)
fh.close()
