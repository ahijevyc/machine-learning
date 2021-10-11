import numpy as np
import ncepgrib2
import datetime as dt

# read in 2D array and write to grib file
def write_grib(field, tdate, fhr, ofile):
    ny, nx = field.shape[-2], field.shape[-1]

    # Section 1 - Identification Section
    discipline = 0
    idsect = {'centreid':7, 'subcentreid':0, 'grbMasterTablesV':2, 'grbLocalTablesV':1, 'sig_reftime':1, \
          'year':tdate.year, 'month':tdate.month, 'day':tdate.day, 'hour':tdate.hour, 'minute':tdate.minute, 'second':tdate.second,\
          'status':2, 'datatype':1}

    # Section 3 - Grid Definition Section
    gdsinfo = {'src_griddef':0, 'npts':nx*ny, 'noct':0, 'opt':0, 'gdtn':30}
    gdtmpl = {'e_shape':6, 'e_sf':0, 'e_sv':0, 'ose_sf_major':0, 'earthRmajor':0, 'ose_sf_minor':0, 'earthRminor':0, 'nx':nx, 'ny':ny, \
          'la1':12190000, 'lo1':226541000, 'rcflag':8, 'LaD':25000000, 'LoV':265000000, 'dx':81270500, 'dy':81270500, \
          'proj_centre_flag':0, 'scanning_mode':64, 'Latin1':25000000, 'Latin2':25000000, 'LatSP':0, 'LonSP':0}

    # Section 4 - Product Definition Section
    # 0=Analysis or forecast at a horizontal level or in a horizontal layer at a point in time.
    # 9=Probability forecasts at a horizontal level or in a horizontal layer in a continuous or non-continuous time interval
    #pdtnum = 0
    # see here: https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
    pdtnum = 0
    #change generating process to 5 for prob forecast?
    #pdtmpl = {'parameter_category':7, 'parameter_num':6, 'generating_process':2, 'backgrd_generating_process':0,
    #     'analys_generating_process':116, 'obs_data_cutoff_hours':0, 'obs_data_cutoff_minutes':0, 'time_range_unit_indicator':1, 'fhr':0,
    #     'fixed_sfc_type1':108, 'fixed_sfc_scale_factor1':0, 'fixed_sfc_scaled_value1':9000,
    #     'fixed_sfc_type2':108, 'fixed_sfc_scale_factor2':0, 'fixed_sfc_scaled_value2':0}
    pdtmpl = {'parameter_category':19, 'parameter_num':2, 'generating_process':13, 'backgrd_generating_process':0,
         'forecast_generating_process':31, 'obs_data_cutoff_hours':0, 'obs_data_cutoff_minutes':0, 'time_range_unit_indicator':1, 'fhr':0,
         'fixed_sfc_type1':1, 'fixed_sfc_scale_factor1':1, 'fixed_sfc_scaled_value1':0,
         'fixed_sfc_type2':1, 'fixed_sfc_scale_factor2':1, 'fixed_sfc_scaled_value2':0}

    # Section 5 - Data Representation Section
    drtnum=0 # 0=simple packing (4=ieee floating point and 40=JPEG2000 compression aren't implemented) 
    field = np.array(field, dtype=np.float64)
    precision = 1.
    nvalues = 1 + np.ceil( (field.max()-field.min()) / (2 * precision))

    #https://www.nws.noaa.gov/mdl/synop/gmos/binaryscaling.php 
    DRTbinary_scale_factor = 0 # -1 = twice the precision, 0 = unchanged, 1 = half the precision, 2=1/4 the precision 
    DRTdecimal_scale_factor = 0 # -1 = tenth the precision, 0=unchanged, 1 = 10x the precision
    DRTorigType = 0 # floating point=0
    #drtmpl = {'DRTref':1, 'DRTbinary_scale_factor':0, 'DRTdecimal_scale_factor':0, 'DRTnbits':np.ceil(np.log2(nvalues)), 'DRTorigType':0}
    drtmpl = {'DRTref':0, 'DRTbinary_scale_factor':0, 'DRTdecimal_scale_factor':4, 'DRTnbits':0, 'DRTorigType':0}
    
    # Tried writing global GRB section once (skipping with subsequent messages) but got error 'addgrid must be called before addfield'
    if field.ndim == 2:
        field = field[np.newaxis,:]
        fhrs = [fhr]
    else:
        fhrs = range(1,field.shape[0]+1)

    fh = open(ofile, "wb")
    
    # write fields to one grib file
    #for i,fhr in enumerate(fhrs):
    #    grbo = ncepgrib2.Grib2Encode(discipline, list(idsect.values()))
    #    field1 = field[i,:] 
    #    grbo.addgrid(list(gdsinfo.values()), list(gdtmpl.values()))
    #    pdtmpl['fhr'] = fhr
    #    grbo.addfield(pdtnum, list(pdtmpl.values()), drtnum, list(drtmpl.values()), field1)
    #    grbo.end()
    #    fh.write(grbo.msg)

    pdtmpl['fhr'] = fhr
    params = [2,199,198,197,201,202]
    for i in range(field.shape[0]):
        grbo = ncepgrib2.Grib2Encode(discipline, list(idsect.values()))
        field1 = field[i,:]
        pdtmpl['parameter_num'] = params[i] 
        grbo.addgrid(list(gdsinfo.values()), list(gdtmpl.values()))
        grbo.addfield(pdtnum, list(pdtmpl.values()), drtnum, list(drtmpl.values()), field1)
        grbo.end()
        fh.write(grbo.msg)

    fh.close()
