import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pygrib
import datetime as dt
import pandas as pd
import xarray as xr
import pickle

sdate = dt.datetime(2021,3,1,0,0,0)
edate = dt.datetime(2021,7,31,0,0,0)
dates = pd.date_range(sdate, edate, freq='1D')
fhrs  = range(1,49)
inits = [0,12]

def output_uh_max():
    all_data = np.zeros((len(dates),len(inits),len(fhrs),65*93))

    for i,tdate in enumerate(dates):
        print(tdate)
        for j,init in enumerate(inits):
            fname = '/glade/work/sobash/NSC/%s%02d_HRRR_upscaled.npz'%(tdate.strftime('%Y%m%d'),init)
            try:
                data = np.load(fname, allow_pickle=True)
                upscaled_fields = data['a'].item()
                all_data[i,j,:,:] = np.array(upscaled_fields['CREF'])[1:,:,:].reshape((48,-1))
            except OSError:
                print(fname)
                all_data[i,j,:,:] = np.nan

    ds = xr.Dataset( data_vars = dict(probs=(["date", "init", "fhr", "idx"], all_data.astype(np.float32)) ),
                 coords=dict(date=dates, init=inits, fhr=fhrs),
                )

    ds.to_netcdf('probs_cref.nc')

def output_probs():
    all_data = np.zeros((len(dates),len(inits),len(fhrs),65*93))

    for i,tdate in enumerate(dates):
        print(tdate)
        for j,init in enumerate(inits):
            for k,fhr in enumerate(fhrs):
                fname = 'grib/hrrr_ml_120km_2hr_%s%02df%03d.grb'%(tdate.strftime('%Y%m%d'),init,fhr)
                try:
                    fh = pygrib.open(fname)
                    all_data[i,j,k,:] = fh[1].values.flatten()
                    fh.close()
                except OSError:
                    print(fname) 
                    all_data[i,j,k,:] = np.nan 

    ds = xr.Dataset( data_vars = dict(probs=(["date", "init", "fhr", "idx"], all_data.astype(np.float32)) ),
                 coords=dict(date=dates, init=inits, fhr=fhrs),
                )

    ds.to_netcdf('probs_120km.nc')

def output_probs_csv():
    all_data = np.zeros((len(dates),len(inits),len(fhrs),65*93))

    for i,tdate in enumerate(dates):
        print(tdate)
        for j,init in enumerate(inits):
            fname = 'probs/probs_nn_%s%02d_40km.out'%(tdate.strftime('%Y%m%d'),init)
            try:
                df = pd.read_csv(fname)
                for idx,row in df.iterrows():
                    all_data[i,j,row['fhr']-1,row['idx']] = row['puh']
            except OSError:
                print(fname)
                all_data[i,j,:,:] = np.nan
    
    ds = xr.Dataset( data_vars = dict(probs=(["date", "init", "fhr", "idx"], all_data.astype(np.float32)) ),
                 coords=dict(date=dates, init=inits, fhr=fhrs),
                )

    ds.to_netcdf('probs_uh_csv.nc')

def plot_qq(d1, d2):
    percs = list(range(0,100))
    percs = np.arange(95,99.9999,0.0001)
    print(percs)
    qn_a = np.percentile(d1, percs)
    qn_b = np.percentile(d2, percs)
    print(qn_a)
    plt.plot(qn_a, qn_b, ls="", marker="o")
    
    x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
    plt.plot(x,x, color="k", ls="--")

    plt.savefig('qq.png')

#output_probs()
#output_probs_csv()
#output_uh_max()

mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb')) #[65*93]

ds = xr.load_dataset('probs.nc')
#ds = xr.load_dataset('probs_wmax.nc')
#ds = xr.load_dataset('probs_sbcape.nc')
all_probs = ds['probs']
all_probs = all_probs.where(mask) #apply us mask
#all_probs = all_probs.where(all_probs>=0.01)
#all_probs = all_probs.mean(dim=["date","idx"], skipna=True)

mask = ~np.isnan( all_probs.values )
#plot_qq(all_probs[:,0,18:24,:].values[mask[:,0,18:24,:]], all_probs[:,1,6:12,:].values[mask[:,1,6:12,:]])
plot_qq(all_probs[:,0,18+24:24+24,:].values[mask[:,0,18+24:24+24,:]], all_probs[:,1,6+24:12+24,:].values[mask[:,1,6+24:12+24,:]])

#for fhr in fhrs:
#    print(all_probs[0,fhr-1].values, all_probs[1,fhr-1].values)
