#!/usr/bin/env python

import datetime as dt
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mpl_toolkits.basemap import *
from matplotlib.ticker import Formatter

def bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

def plot_2d_hist(predx, predy):
    cmap = plt.get_cmap('Blues')
    #norm = colors.BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
    norm = colors.BoundaryNorm(np.logspace(1,6,num=15), ncolors=cmap.N, clip=True)

    histox, bins = np.histogram(predx, bins=np.arange(0,1.1,0.1))
    histoy, bins = np.histogram(predy, bins=np.arange(0,1.1,0.1))

    fig = plt.figure(figsize=(6,6))
    h = plt.hist2d(predx, predy, bins=30, cmin=10, cmap=cmap, norm=norm)
    plt.style.use('seaborn-white')
    plt.plot([0,1], [0,1], color='k')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid()
    plt.xlabel('ML')
    plt.ylabel('UH')
    plt.savefig('hist2d.png')

def make_gridded_forecast(predictions, labels, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    unique_forecasts, unique_fhr = np.unique(dates), np.unique(fhr)
    print(unique_forecasts[dloc])
    num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)
    predictions = predictions.reshape((num_dates, num_fhr, -1))

    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype='f')
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93), dtype='f')

    thismask = mask.flatten()

    # just grid predictions for this class
    predictions = predictions.reshape((num_dates, num_fhr, -1))
    labels      = labels.reshape((num_dates, num_fhr, -1))

    print(gridded_predictions.shape, predictions.shape)

    for m, dt in enumerate(unique_forecasts):
        for n, f in enumerate(unique_fhr):
            gridded_predictions[m,n,thismask] = predictions[m,n,:]
            gridded_labels[m,n,thismask]      = labels[m,n,:]      
        #print(dt, gridded_predictions[i,:].max())

    # return only predictions for US points
    return (gridded_predictions.reshape((num_dates, num_fhr, 65, 93)), gridded_labels.reshape((num_dates, num_fhr, 65, 93)))

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}    
i = 0 
numclasses = 6
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
smooth_sigma = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]

predictions_all_nn, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open('predictions_nn_120km_2hr_all', 'rb'))
#predictions_all_rf, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open('predictions_rf_120km_2hr_n100_all', 'rb'))
predictions_all = predictions_all_nn

#months_all = np.array([ int(d[5:7]) for d in dates_all ])
unique_forecasts, unique_fhr = np.unique(dates_all), np.unique(fhr_all)

print('Verifying %d forecast points'%predictions_all.shape[0])

#### mask and flatten (only verify over CONUS points)
#mask = mask.reshape((65,93))
#predictions_gridded = predictions_gridded[:,:,mask]

fig = plt.figure(figsize=(9,9))
gs = gridspec.GridSpec(4,1)
gs.update(hspace=0.17)

labels = ['Any severe',\
          'Wind gust > 50 kts (blue) and > 65 kts (orange)',\
          'Hail >= 1" (blue) and >= 2" (orange)',\
          'Tornado']

awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
grid81 = awips.makegrid(93, 65, returnxy=True)
lons, lats = grid81[0], grid81[1]

xloc, yloc = 60, 23
xloc, yloc = 55, 25
#xloc, yloc = 65, 31
#xloc, yloc = 48, 26
#thisdate = dt.datetime(2011,4,27,0,0,0)
thisdate = dt.datetime(2012,12,19,0,0,0)
dloc = np.argwhere(unique_forecasts == thisdate.strftime('%Y-%m-%d %H:%M:%S'))[0][0]
#plotdates = np.arange('2012-04-14T01:00:00', '2012-04-15T13:00:00', dtype='datetime64[h]')
plotdates = [ thisdate + dt.timedelta(hours=h) for h in range(1,37) ]
print(lons[yloc,xloc], lats[yloc,xloc])

class MyFormatter(Formatter):
    def __init__(self, fmt='%H'):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        tdate = mdates.num2date(x)
        thisstr = tdate.strftime('%H')
        if thisstr == "00": return tdate.strftime('%H UTC\n%b %-d\n%Y')
        else: return thisstr

for i in range(0,6):
    print(i)
    if i < 4: panel = i
    if i == 4: panel = 2
    if i == 5: panel = 1
    ax = plt.subplot(gs[panel])

    predictions_gridded, labels_gridded = make_gridded_forecast(predictions_all[:,i], labels_all[:,i], dates_all, fhr_all)

    #if i==0: maxprob = np.amax(predictions_gridded[dloc,:,yloc,xloc])

    if i < 4:  ax.text(0,1.01, labels[panel], fontsize=8, va='bottom', transform=ax.transAxes)
    #if i == 0: ax.text(0,1.01, 'Convective hazard probabilities w/in 120-km and 2-hr', ha='right', va='bottom', transform=ax.transAxes, fontsize=10) 
    if i == 1: ax.text(0,1.1, 'Convective hazard forecast guidance', ha='left', va='bottom', transform=ax.transAxes, fontsize=10, fontweight='bold') 
    if i == 1: ax.text(1,1.01, 'Probs for hazard w/in 75-mi and 2-hr of %.2fN, %.2fW'%(lats[yloc,xloc], np.abs(lons[yloc,xloc])), ha='right', va='bottom', transform=ax.transAxes, fontsize=8) 
    
    ax.bar(plotdates, predictions_gridded[dloc,:,yloc,xloc], width=0.03)

    # plot observed severe weather report times
    for h in range(0,36):
        if labels_gridded[dloc,h,yloc,xloc] > 0:
            if i < 4: ax.scatter(plotdates[h],predictions_gridded[dloc,h,yloc,xloc]+0.03,s=20,marker='o',color='#1f77b4',edgecolor='k', linewidth=0.25, alpha=0.75, zorder=2)
            if i >= 4: ax.scatter(plotdates[h],predictions_gridded[dloc,h,yloc,xloc]+0.03,s=20,marker='o',color='#ff7f0e', edgecolor='k', linewidth=0.25, alpha=0.75, zorder=2)
    print(predictions_gridded.shape)

    ax.set_xlim(thisdate, thisdate + dt.timedelta(hours=37))    
    
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,3)))
    ax.xaxis.set_major_formatter(MyFormatter())

    ax.set_ylabel('Probability', fontsize=8)
    ax.set_ylim((0,1.0))
    ax.grid(lw=0.25)

    ax.tick_params(bottom='on', axis='both', width=0.5, direction='out', labelsize=8, labelbottom='off')
    if i == 3:
        ax.tick_params(bottom='on', labelbottom='on')
        #ax.set_xlabel('Hour (UTC)', fontsize=8)

    for s in ax.spines: ax.spines[s].set_linewidth(0.25)

plt.savefig('timeseries.png', dpi=150, bbox_inches='tight')
