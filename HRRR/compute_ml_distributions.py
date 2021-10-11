import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pygrib
import datetime as dt
import pandas as pd
import xarray as xr
import pickle
import os, sys

def output_uh_max():
    all_data = np.zeros((len(dates),2,len(fhrs),65*93))

    for i,tdate in enumerate(dates):
        print(tdate)
        
        fname1 = '/glade/work/sobash/NSC/%s00_HRRR_upscaled.npz'%(tdate.strftime('%Y%m%d'))
        fname2 = '/glade/work/sobash/NSC/%s00_HRRRX_upscaled.npz'%(tdate.strftime('%Y%m%d'))
        
        if os.path.exists(fname1) and os.path.exists(fname2):
            data1 = np.load(fname1, allow_pickle=True)
            upscaled_fields1 = data1['a'].item()
            
            data2 = np.load(fname2, allow_pickle=True)
            upscaled_fields2 = data2['a'].item()

            all_data[i,0,:,:] = np.array(upscaled_fields1[field])[1:,:,:].reshape((len(fhrs),-1))
            all_data[i,1,:,:] = np.array(upscaled_fields2[field])[1:37,:,:].reshape((len(fhrs),-1))

        else:
            print('file missing')
            all_data[i,:,:,:] = np.nan

    ds = xr.Dataset( data_vars = { field: (["date", "model", "fhr", "idx"], all_data.astype(np.float32)) },
                     coords=dict(date=dates, model=['hrrr','hrrrx'], fhr=fhrs),
                    )

    ds.to_netcdf('hrrr_%s.nc'%field)

def plot_qq(d1, d2):
    #percs = list(range(0,100))
    percs =np.append(  np.arange(90,99.99,0.01), np.arange(99.99,99.99999,0.0001) )
    print(percs)
    qn_a = np.nanpercentile(d1, percs)
    qn_b = np.nanpercentile(d2, percs)
    print(qn_a)
    plt.plot(qn_a, qn_b, ls="", marker="o")
    
    x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
    plt.plot(x,x, color="k", ls="--")

    plt.savefig('qq.png')

sdate = dt.datetime(2020,4,1,0,0,0)
edate = dt.datetime(2020,6,30,0,0,0)
dates = pd.date_range(sdate, edate, freq='1D')
fhrs = range(1,37)
field = 'UP_HELI_MAX'

output_uh_max()

mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb')) #[65*93]

ds = xr.load_dataset('hrrr_%s.nc'%field)
hrrr_values = ds[field].sel(model='hrrr').where(mask)
hrrrx_values = ds[field].sel(model='hrrrx').where(mask)

#all_probs.plot.hist()
#plt.savefig('test.png')

plot_qq(hrrr_values.values, hrrrx_values.values)

#for fhr in fhrs:
#    print(all_probs[0,fhr-1].values, all_probs[1,fhr-1].values)
