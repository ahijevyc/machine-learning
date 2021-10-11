#!/usr/bin/env python

import datetime
import pickle, sys
import numpy as np
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.ndimage.filters import uniform_filter, gaussian_filter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import *
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap,BoundaryNorm
import multiprocessing
import pandas as pd

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in list(fh.read().splitlines()):
        if appending: rgb.append(list(map(float,line.split())))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

def plot_stats_hourly(ptype='fss'):
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.gca()
    numrows, numcols = 2,1
    numpanels = numrows*numcols
    gs = gridspec.GridSpec(numrows,numcols,height_ratios=[4,1])
    gs.update(hspace=0.07)

    fontsize=10
    lw=3.5; ms=3
    fig.suptitle('', fontsize=fontsize+2)
   
    ### top panel 
    ax1 = plt.subplot(gs[0]) 
    ax1.tick_params(bottom='on', axis='both', width=0.5, direction='out', labelsize=fontsize-2, labelbottom='off')
    ax1.set_xlim((1,36))
    ax1.set_xticks([1,6,12,18,24,30,36])
    ax1.grid(color='0.7', linewidth=0.25) 
    for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(0.5)
    for i in range(0,37,24): ax1.axvspan(i,i+12,ymin=0,ymax=1,facecolor='gray',alpha=0.25)

    if ptype=='bss':
        ax1.set_ylabel('Brier Skill Score', fontsize=fontsize-1)
        ax1.set_ylim((0,0.4))
    
        ax1.plot(range(1,37), bss_fhr_uh, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw)
        ax1.plot(range(1,37), bss_fhr_ml, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw)
          
        ax1.fill_between(range(1,37), bss_fhr_uh_boot[:,0], bss_fhr_uh_boot[:,2], color='k', alpha=0.2, lw=0)
        ax1.fill_between(range(1,37), bss_fhr_ml_boot[:,0], bss_fhr_ml_boot[:,2], color='k', alpha=0.2, lw=0)

    if ptype=='auc':
        ax1.set_ylabel('Area Under Curve', fontsize=fontsize-1)
        ax1.set_ylim((0.5,1.0)) 
        
        ax1.plot(range(1,37), auc_fhr_uh, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw)
        ax1.plot(range(1,37), auc_fhr_ml, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw)

        ax1.fill_between(range(1,37), auc_fhr_uh_boot[:,0], auc_fhr_uh_boot[:,2], color='k', alpha=0.2, lw=0)
        ax1.fill_between(range(1,37), auc_fhr_ml_boot[:,0], auc_fhr_ml_boot[:,2], color='k', alpha=0.2, lw=0)

    ### bottom panel
    ax2 = plt.subplot(gs[1])
    
    ax2.tick_params(bottom='on', axis='both', width=0.5, direction='out', labelsize=fontsize-2, labelbottom='on')
    ax2.set_xlabel('Forecast Hour (UTC)', fontsize=fontsize-1, labelpad=4)
    ax2.set_xlim((1,36))
    ax2.set_xticks([1,6,12,18,24,30,36])
    ax2.grid(color='0.7', linewidth=0.25) 
    for axis in ['top','bottom','left','right']: ax2.spines[axis].set_linewidth(0.5)
    for i in range(0,37,24): ax2.axvspan(i,i+12,ymin=0,ymax=1,facecolor='gray',alpha=0.25)
    
    if ptype=='bss':
        ax2.set_ylim((0,0.2))
    
        ax2.plot(range(1,37), bss_fhr_ml-bss_fhr_uh, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw)
        ax2.fill_between(range(1,37), bss_fhr_boot_diff[:,0], bss_fhr_boot_diff[:,2], color='k', alpha=0.2, lw=0)
    if ptype=='auc':
        ax2.set_ylim((0,0.3))

        ax2.plot(range(1,37), auc_fhr_ml-auc_fhr_uh, marker='o', markersize=ms, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', linewidth=lw) 
        ax2.fill_between(range(1,37), auc_fhr_diff_boot[:,0], auc_fhr_diff_boot[:,2], color='k', alpha=0.2, lw=0) 

    plt.savefig('%s.pdf'%ptype)

def plot_2d_hist(predx, predy):
    cmap = plt.get_cmap('Greys')
    #norm = colors.BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)
    norm = colors.BoundaryNorm(np.logspace(0,6,num=10), ncolors=cmap.N, clip=True)

    histox, bins = np.histogram(predx, bins=np.arange(0,1.1,0.1))
    histoy, bins = np.histogram(predy, bins=np.arange(0,1.1,0.1))

    plt.rcParams.update({'font.size': 14})

    # 2D histogram figure
    fig = plt.figure(figsize=(9,9))
    h = plt.hist2d(predx, predy, bins=np.arange(0,1.01,0.025), cmin=1, cmap=cmap, norm=norm)
    plt.style.use('seaborn-white')
    plt.plot([0,1], [0,1], color='k')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid()
    plt.xlabel('SSPF')
    plt.ylabel('NN')
    plt.savefig('hist2d.png')

    # 2D histogram figure
    fig = plt.figure(figsize=(8,8))
    plt.style.use('seaborn-white')
    n_sspf, bins, patches = plt.hist(predx, bins=np.arange(0,1.01,0.01), alpha=0.5, histtype='stepfilled', edgecolor='none', log=True)
    n_ml, bins, patches = plt.hist(predy, bins=np.arange(0,1.01,0.01), alpha=0.5, histtype='stepfilled', edgecolor='none', log=True)
    plt.grid()
    plt.xlim((0,1))
    plt.ylim((1,1e8))
    plt.xlabel('Probability')
    plt.ylabel('Number of grid points')
    plt.savefig('hist_sspf.png')

    #print((predx<0.0001).sum(), (predy<0.0001).sum())
    #print(n_sspf, n_ml, n_ml/n_sspf)

    #print(histox, histoy)
    #print(np.histogram(predx-predy, bins=np.arange(-1,1.1,0.1)))

def plot_daily_bss_scatter(bss1, bss2):
    plt.rcParams.update({'font.size': 14})

    colormonth = np.where(np.isin(months_all, [4,5,6,7]), '#fc9272', '#9ecae1')

    # read in and reorder cape removal time scale data
    #all_cape_tscale = []
    #for i in range(1,36):
    #    cape_tscale = np.genfromtxt('/glade/p/mmm/parc/schwartz/3vs1/environmental_stats/EAST_CONUS/data_cape_removal_time_scale_ncar_3km_12sec_ts_f%03d.txt'%i)
    #    all_cape_tscale.append(cape_tscale)
    #cape_tscale = np.mean(all_cape_tscale, axis=0)
    #cape_tscale = dict(list(zip(cape_tscale[:,0].astype(np.int).astype(np.str), cape_tscale[:,2])))
    #dates_int   = np.array([ datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H') for d in unique_forecasts ])
    #cape_tscale = [ cape_tscale[d] for d in dates_int ]
    
    #cmap = plt.get_cmap('RdGy_r')
    #norm = BoundaryNorm(np.arange(0,10), ncolors=cmap.N, clip=True)

    # 2D histogram figure
    fig = plt.figure(figsize=(8,8))
    plt.style.use('seaborn-white')
    #plt.scatter(bss1, bss2, marker='o', c=cape_tscale, cmap=cmap, norm=norm)
    plt.scatter(bss1, bss2, marker='o', c='#AFAFAF')
    plt.scatter(bss_all_ml, bss_all_uh, marker='o', s=30, c='black')
    plt.plot([-1,1], [-1,1], color='0.4', lw=1)
    plt.plot([-1,1], [0,0], color='0.4', lw=1)
    plt.plot([0,0], [-1,1], color='0.4', lw=1)
    plt.grid()
    plt.xlim((-0.2,0.8))
    plt.ylim((-0.2,0.8))
    plt.xlabel('NNPF BSS')
    plt.ylabel('SSPF BSS')
    plt.savefig('bss_scatter.png')

def compute_reliability_all(obs, pred):
    # if obs/pred are on grid, need to remove points outside of US mask 
    obs = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask]
    pred = pred.reshape((num_dates, num_fhr, -1))[:,:,thismask]
    
    fcst_yes_bins, obs_yes_bins, fcst_bin_avg_prob = [], [], []
    prob_bins        = np.arange(0,1.06,0.05)
    prob_bins        = np.arange(0,1.01,0.1)
    prob_bins_center = np.array((np.array(prob_bins) + 0.025)[:-1])
    prob_bins_center = np.array((np.array(prob_bins) + 0.05)[:-1])
 
    for i in range(0,prob_bins_center.size):
        fcst_bin_mask  =  (pred >= prob_bins[i]) & (pred < prob_bins[i+1])
        fcst_bin_sums  =  np.sum(fcst_bin_mask, axis=(1,2)) #[numdays]

        fcst_bin_avg_prob.append( pred[fcst_bin_mask].mean() )

        osr_hits       =  np.where(fcst_bin_mask, obs, 0)
        obs_bin_sums   =  np.sum(osr_hits, axis=(1,2)) #[numdays]

        fcst_yes_bins.append(fcst_bin_sums) #[numbins,numdays]
        obs_yes_bins.append(obs_bin_sums) #[numbins,numdays]

        #rel = (obs_bin_sums / fcst_bin_sums) #[numdays]
        #true_prob.append(rel) #[numdays,numbins]

    fcst_yes_bins, obs_yes_bins = np.array(fcst_yes_bins), np.array(obs_yes_bins)
    rel = (obs_yes_bins.sum(axis=1) / fcst_yes_bins.sum(axis=1)) #need to sum over days

    # need to transpose so array is [numdays,numbins]
    cis = bootstrap_rel(fcst_yes_bins.T, obs_yes_bins.T)
    
    return (rel, prob_bins_center, cis, fcst_yes_bins.sum(axis=1), fcst_bin_avg_prob)

def readSevereClimo(fname, day_of_year, hr):
    from scipy.interpolate import RectBivariateSpline
    data = np.load(fname)
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    grid81 = awips.makegrid(93, 65, returnxy=True)
    x, y = awips(data['lons'], data['lats'])

    #spline = RectBivariateSpline(x[0,:], y[:,0], data['severe'][day_of_year-1,hr,:].T, kx=3, ky=3)
    #interp_data = spline.ev(grid81[2].ravel(), grid81[3].ravel())
    return np.reshape(interp_data, (65,93))

def make_gridded_forecast(predictions, labels, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)

    # just grid predictions for this class
    predictions = predictions.reshape((num_dates, num_fhr, -1))
    labels      = labels.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_predictions[i,j,thismask] = predictions[i,j,:]
            gridded_labels[i,j,thismask]      = labels[i,j,:]      
        #print(dt, gridded_predictions[i,:].max())

    # return only predictions for US points
    return (gridded_predictions.reshape((num_dates, num_fhr, 65, 93)), gridded_labels.reshape((num_dates, num_fhr, 65, 93)))

def grid_data(field):
    # convert 1d array into 4d array with shape (num_dates, num_fhr, 65, 93)
    gridded_field = np.zeros((num_dates,num_fhr,65*93), dtype=np.float64)
    field = field.reshape((num_dates, num_fhr, -1))

    for i, dt in enumerate(unique_forecasts):
        for j, f in enumerate(unique_fhr):
            gridded_field[i,j,thismask] = field[i,j,:]

    return gridded_field.reshape((num_dates, num_fhr, 65, 93))

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

def plot_bss_spatial(data2d, fname='bss_spatial.png'):
    ### PLOT bss ###
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb'))
    lons, lats = awips.makegrid(93, 65, returnxy=False)

    x, y = m(lons, lats)
 
    #test = readNCLcm('MPL_Greys')[35::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
    test = readNCLcm('MPL_Greys')[45::] + [[1,1,1]] + readNCLcm('MPL_Reds')[30::]
    cmap = ListedColormap(test)
    norm = BoundaryNorm(np.arange(0,0.5,0.05), ncolors=cmap.N, clip=True)        
    norm = BoundaryNorm(np.arange(-0.25,0.25,0.05), ncolors=cmap.N, clip=True)       
    #norm = BoundaryNorm(np.arange(-0.05,0.06,0.01), ncolors=cmap.N, clip=True)        

    labels_gridded_summed = labels_gridded.sum(axis=(0,1)).flatten()

    for i,b in enumerate(data2d.flatten()):
        color = cmap(norm([b])[0])
        if not np.isnan(b) and not np.isinf(b) and thismask[i] and labels_gridded_summed[i] > 25:
            bss_val = int(round(b*100))
            if bss_val > 99: bss_val = 99
            if bss_val < -99: bss_val = -99
            
            #if b<0: a = axes.text(x.flatten()[i], y.flatten()[i], '<0', fontsize=12, ha='center', va='center', family='monospace', color='#bdd7e7', fontweight='bold')
            #else:   a = axes.text(x.flatten()[i], y.flatten()[i], bss_val, fontsize=12, ha='center', va='center', family='monospace', color=color, fontweight='bold')
            a = axes.text(x.flatten()[i], y.flatten()[i], bss_val, fontsize=12, ha='center', va='center', family='monospace', color=color, fontweight='bold')

    plt.savefig(fname)

def plot_forecast(data2d, fname='forecast.png'):
    awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb'))
    lons, lats = awips.makegrid(93, 65, returnxy=False)

    x, y = m(lons, lats)
    
    #test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    test = readNCLcm('MPL_Greys')[35::] + [[1,1,1]] + readNCLcm('MPL_Reds')[20::]
    cmap = ListedColormap(test)
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)        

    labels_flatten = np.amax(labels_gridded[fmask,:], axis=0).flatten()
    x, y = x.flatten(), y.flatten()
    for i,b in enumerate(data2d.flatten()):
        color = cmap(norm([b])[0])

        if labels_flatten[i]: axes.scatter(x[i], y[i], color='black', marker='o', s=13**2, lw=1, facecolors='None', edgecolors='0.6')

        if not np.isnan(b) and not np.isinf(b) and thismask[i] and b>0.05:
        #if not np.isnan(b) and not np.isinf(b) and thismask[i] and b>5:
            
            #val = int(round(b))
            val = int(round(b*100))
            #if val > 99: val = 99
            #if val < -99: val = -99

            a = axes.text(x[i], y[i], val, fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')

    plt.savefig(fname, dpi=150)

def plot_forecast_old(predictions, prefix="", fhr=36):
    test = readNCLcm('MPL_Greys')[25::] + [[1,1,1]] + readNCLcm('MPL_Reds')[10::]
    #test = readNCLcm('perc2_9lev')[1::]
    cmap = ListedColormap(test)
    #cmap = plt.get_cmap('RdGy_r')
    norm = BoundaryNorm(np.arange(0,1.1,0.1), ncolors=cmap.N, clip=True)

    print(predictions)

    #awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution='l', area_thresh=10000.)

    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/ch_pk_files/rt2015_ch_CONUS.pk', 'r'))
    #fig, axes, m  = pickle.load(open('/glade/u/home/sobash/NSC_scripts/dav_pk_files/rt2015_ch_CONUS.pk', 'rb'))
    fig, axes, m = pickle.load(open('rt2015_ch_CONUS.pk', 'rb'))

    lats, lons = predictions['lat'].values, predictions['lon'].values
    x, y = m(lons, lats)

    # do something convoluted here to only plot each point once
    probmax = {}
    for i,p in enumerate(predictions['predict_proba'].values):
        thiskey = '%f%f'%(lats[i],lons[i])
        if thiskey in probmax:
            if p > probmax[thiskey]:
                probmax[thiskey] = p
        else:
           probmax[thiskey] = p

    # need to do this before calling text
    #m.set_axes_limits(ax=axes)

    for i,p in enumerate(predictions['predict_proba'].values):
        thiskey = '%f%f'%(lats[i],lons[i])
        thisvalue = probmax[thiskey]

        color = cmap(norm([thisvalue])[0])
        probmax[thiskey] = -999
        if x[i] < m.xmax and x[i] > m.xmin and y[i] < m.ymax and y[i] > m.ymin and thisvalue > 0.05:
        #if thisvalue >= 0.15:
            a = axes.text(x[i], y[i], int(round(thisvalue*100)), fontsize=10, ha='center', va='center', family='monospace', color=color, fontweight='bold')
           # a = axes.text(x[i], y[i], int(round(thisvalue*100)), fontsize=12, ha='center', va='center', family='monospace', color=color, fontweight='bold')
    #a = m.scatter(x, y, s=50, c=predictions['predict_proba'].values, lw=0.5, edgecolors='k', cmap=cmap, norm=norm)

    ax = plt.gca()
    cdate = sdate + dt.timedelta(hours=fhr)
    sdatestr = (cdate - dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    edatestr = (cdate + dt.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S UTC')
    plt.text(0,1.01,'Probability of tornado within 75-mi of a point valid %s - %s'%(sdatestr, edatestr), fontsize=14, transform=ax.transAxes)

    # ADD COLORBAR
    #cax = fig.add_axes([0.02,0.1,0.02,0.3])
    #cb = plt.colorbar(a, cax=cax, orientation='vertical', extendfrac=0.0)
    #cb.outline.set_linewidth(0.5)
    #cb.ax.tick_params(labelsize=10)

    # plot reflectivity
    initstr = sdate.strftime('%Y%m%d00')
    wrfcdate = cdate.strftime('%Y-%m-%d_%H_%M_%S')
    fh = Dataset('/glade/p/mmm/parc/sobash/NSC/3KM_WRF_POST_12sec_ts/%s/diags_d01_%s.nc'%(initstr,wrfcdate), 'r')
    lats = fh.variables['XLAT'][0,:]
    lons = fh.variables['XLONG'][0,:]
    cref = fh.variables['REFL_COM'][0,:]
    fh.close()

    x, y = m(lons, lats)
    plt.contourf(x, y, cref, levels=[35,1000], colors='k', alpha=0.5)

    plt.savefig('forecast%s.png'%prefix, dpi=150)

# COMPUTE bss
# (469,36,65,93) input dimension
def compute_bss_spatial(pred, obs):
    bs_spatial = (pred - obs)**2

    # use climo per grid box
    #climo = labels_gridded.mean(axis=(0,1)) #take mean over days and forecast hour 
    #climo = uniform_filter(climo, size=3) #take mean within 1 grid box
    #bs_climo = (climo[np.newaxis,np.newaxis,:] - labels_gridded)**2
    
    #use 30-year climo
    bs_climo = ( climo_all - obs )**2
    bs_climo = bs_climo.mean(axis=(0,1))
    bs_climo = uniform_filter(bs_climo, size=3)

    # compute brier skill score for each grid box
    bs_spatial = bs_spatial.mean(axis=(0,1)) #aggregate num fcst/fhr dimensions
    bs_spatial = uniform_filter(bs_spatial, size=3)
    bss_spatial = 1 - (bs_spatial / bs_climo)

    return bss_spatial

def compute_bss_daily(pred, obs):
    diffs = (pred - obs)**2
    diffs = diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]
    #diffs = diffs[:,12,:][:,np.newaxis,:]
    bs_daily = diffs.mean(axis=(1,2))

    # use 30-year climo
    climo_diffs = ( climo_all - obs )**2
    climo_diffs = climo_diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]

    #climo_diffs = climo_diffs[:,12,:][:,np.newaxis,:]
    bs_climo_daily = np.mean( climo_diffs, axis=(1,2) )

    # compute brier skill score for each grid box
    bss_daily = 1 - (bs_daily / bs_climo_daily)

    return bss_daily

def createCI(data, B, quantile):
    data.sort()

    index_low = int(B*(quantile/2.0)-1)
    index_high = int(B*(1-(quantile/2.0))-1)
    index_middle = int((B/2.0)-1)
    ci_low = data[index_low]
    ci_high = data[index_high]
    bs_mean = data[index_middle]
    bs_median = np.median(data)

    return (ci_low, bs_mean, ci_high, bs_median)

def bootstrap_rel(fcst_yes, obs_yes, alpha=0.9, B=10000):
    #fcst_yes has shape [numdays,numbins]
    #obs_yes has shape [numdays,numbins]
    n = fcst_yes.shape[0]
    bins = fcst_yes.shape[1]
    idx = np.random.randint(0, n, (B,n))

    fcst_yes_draw = fcst_yes[idx,:] #shape becomes [B,numdays,numbins]
    obs_yes_draw  = obs_yes[idx,:]
    print(fcst_yes_draw.shape, obs_yes_draw.shape)

    # sum over number of days 
    fcst_yes_draw_sum = np.sum(fcst_yes_draw, axis=1)
    obs_yes_draw_sum  = np.sum(obs_yes_draw, axis=1)

    rel = obs_yes_draw_sum/fcst_yes_draw_sum

    cis = []
    for k in range(bins):
        cis.append(createCI(rel[:,k], B, 1-alpha))

    return np.array(cis)

def bootstrap_bss(bss1=None, bss2=None, alpha=0.9, B=10000):
    bs, bs_ref = bss1

    n = bs.size
    idx = np.random.randint(0, n, (B,n))

    bs_draw = bs[idx]
    bs_ref_draw = bs_ref[idx]

    bs_sum = np.sum(bs_draw, axis=1)
    bs_ref_sum = np.sum(bs_ref_draw, axis=1)
    bss = (1 - (bs_sum/bs_ref_sum))

    if bss2 is not None:
        bs, bs_ref = bss2
        bs_draw = bs[idx]
        bs_ref_draw = bs_ref[idx]

        bs_sum = np.sum(bs_draw, axis=1)
        bs_ref_sum = np.sum(bs_ref_draw, axis=1)

        bss2 = (1 - (bs_sum/bs_ref_sum))
        stat = bss2 - bss
    else:
        stat = bss
    return createCI(stat, B, 1-alpha)

def roc_auc_score_parallel(a):
    auc = metrics.roc_auc_score(obs_draw[a,:], fcst_draw[a,:])
    return auc

def bootstrap_auc(auc1=None, auc2=None, alpha=0.99, B=10000): 
    obs, fcst = auc1
    n = fcst.shape[0]
    idx = np.random.randint(0, n, (B,n))
    
    global fcst_draw
    global obs_draw
    
    fcst_draw = fcst[idx,:].reshape((B,-1))
    obs_draw  = obs[idx,:].reshape((B,-1))

    nprocs    = 30
    chunksize = int(math.ceil(B / float(nprocs)))
    pool      = multiprocessing.Pool(processes=nprocs)
    aucs      = pool.map(roc_auc_score_parallel, range(0,B), chunksize)
    pool.close()
    #for a in range(0,B):
    #auc = metrics.roc_auc_score(obs_draw[a,:].flatten(), fcst_draw[a,:].flatten())
    #aucs.append(auc)

    if auc2 is not None:
        obs2, fcst2 = auc2
    
        # use same idx here so they are paired
        fcst_draw = fcst2[idx,:].reshape((B,-1))
        obs_draw  = obs2[idx,:].reshape((B,-1))
        
        pool      = multiprocessing.Pool(processes=nprocs)
        aucs2      = pool.map(roc_auc_score_parallel, range(0,B), chunksize)
        pool.close()
        #aucs2 = []
        #for a in range(0,B):
        #    auc = metrics.roc_auc_score(fcst_draw[a,:].flatten(), obs_draw[a,:].flatten())
        #    aucs2.append(auc)

        stat = np.array(aucs2) - np.array(aucs)

    else:
        stat = aucs 

    return createCI(np.array(stat), B, 1-alpha)

def compute_bss_fhr(pred, obs):
    diffs = (pred - obs)**2

    obs_gridded_masked = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask]    

    # compute climo by forecast hour
    climo_by_fhr    = np.mean( obs_gridded_masked, axis=(0,2) )
    climo_diffs     = ( obs_gridded_masked - climo_by_fhr[np.newaxis,:,np.newaxis] )**2
    bs_climo_by_fhr = np.mean( climo_diffs, axis=(0,2) )
    #print(climo_by_fhr)

    # use 30-year climo
    climo_diffs = ( climo_all - obs )**2
    climo_diffs = climo_diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]
    bs_climo_by_fhr = np.mean( climo_diffs, axis=(0,2) )
 
    # compute brier skill score for each forecast hour
    diffs = diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]
    bs_fhr = diffs.mean(axis=(0,2)) #average over days and space

    bss_fhr = 1 - (bs_fhr / bs_climo_by_fhr)

    bss_boot_all = []
    for n,f in enumerate(range(sfhr,efhr+1)):
        bss_fhr_bs = bootstrap_bss(( diffs[:,n,:].sum(axis=1), climo_diffs[:,n,:].sum(axis=1) ))
        bss_boot_all.append(bss_fhr_bs)
        print(f, bss_fhr_bs)
 
    return ( bss_fhr , np.array(bss_boot_all), diffs, climo_diffs )

def compute_bss(pred, obs):
    # compute climo by forecast hour
    #obs_gridded_masked = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask]
    #climo_by_fhr    = np.mean( obs_gridded_masked, axis=(0,2) )
    #climo_diffs     = ( obs_gridded_masked - climo_by_fhr[np.newaxis,:,np.newaxis] )**2
    #bs_climo_by_fhr = np.mean( climo_diffs, axis=(0,2) )
    #print(climo_by_fhr)

    # compute 30-year climo brier score
    climo_diffs = ( climo_all - obs )**2
    climo_diffs = climo_diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]
    bs_climo    = climo_diffs.mean()

    # compute forecast brier score
    fcst_diffs  = ( pred - obs )**2
    fcst_diffs  = fcst_diffs.reshape((num_dates,num_fhr,-1))[:,:,thismask]
    bs_fcst     = fcst_diffs.mean()

    # compute brier skill score
    return (1 - (bs_fcst / bs_climo))

def compute_auc_fhr(pred, obs, pred2=None, obs2=None):
    obs_masked = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask].astype(np.float32)
    pred_masked = pred.reshape((num_dates, num_fhr, -1))[:,:,thismask].astype(np.float32)

    if pred2 is not None:
        obs2_masked = obs2.reshape((num_dates, num_fhr, -1))[:,:,thismask].astype(np.float32)
        pred2_masked = pred2.reshape((num_dates, num_fhr, -1))[:,:,thismask].astype(np.float32)

    auc_all, auc_bs_all = [], []
    for f in range(num_fhr):
        #fpr, tpr, thresholds = metrics.roc_curve(obs_masked[:,f,:].flatten(), pred_masked[:,f,:].flatten())
        if pred2 is None:
            auc = metrics.roc_auc_score(obs_masked[:,f,:].flatten(), pred_masked[:,f,:].flatten())
            auc_all.append(auc)

            auc_bs = bootstrap_auc((obs_masked[:,f,:], pred_masked[:,f,:]), B=1000)
            auc_bs_all.append(auc_bs)
            print(datetime.datetime.now(), f, auc, auc_bs)
        else:
            auc_bs = bootstrap_auc( (obs_masked[:,f,:], pred_masked[:,f,:]),\
                                    (obs2_masked[:,f,:], pred2_masked[:,f,:]), B=1000)
            auc_bs_all.append(auc_bs)
            print(datetime.datetime.now(), f, auc_bs)

    return (np.array(auc_all), np.array(auc_bs_all))
    #return (np.array(auc_bs_all)[:,1], np.array(auc_bs_all))

def compute_auc_fhr_old(pred, obs):
    obs_masked = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask]
    pred_masked = pred.reshape((num_dates, num_fhr, -1))[:,:,thismask]

    prob_thresh = np.arange(0,1.01,0.02)
    for f in range(num_fhr):
        pod_all, pofd_all = [], []
        for p in prob_thresh:
            cm = metrics.confusion_matrix(obs_masked[:,f,:].flatten(), (pred_masked[:,f,:]>=p).flatten())
            hits, fals, miss, neg = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        
            pod  = hits / ( hits + miss )
            pofd = fals / ( neg + fals )
            pod, pofd = np.nan_to_num(pod), np.nan_to_num(pofd)
            pod_all.append(pod)
            pofd_all.append(pofd)

        auc = 0
        for i in range(prob_thresh.size-1):
            auc += ((pod_all[i] + pod_all[i+1])/2.0)*(pofd_all[i]-pofd_all[i+1])
        print(auc)

def compute_auc_all(pred, obs):
    obs_masked = obs.reshape((num_dates, num_fhr, -1))[:,:,thismask]
    pred_masked = pred.reshape((num_dates, num_fhr, -1))[:,:,thismask] 

    fpr, tpr, thresholds = metrics.roc_curve(obs_masked.flatten(), pred_masked.flatten())
    
    #fig, ax = plt.subplots()
    #ax.plot(fpr, tpr)
    #ax.grid()
    #fig.savefig("roc.png")

    return metrics.roc_auc_score(obs_masked.flatten(), pred_masked.flatten())

def compute_2d_histo(pred, obs, fname='histo2d.png'):
    #idx = np.nonzero(obs) #should be 4d (days, fhr, ny, nx)
    prob_histo_fcst = np.zeros((20,20))
    num_rpts = 0

    for d,f,y,x in list(zip(*np.nonzero(obs))):
        sx = x-10
        ex = x+10
        sy = y-10
        ey = y+10
        if sx < 0: sx =0
        if ex > 92: ex=92
        if sy < 0: sy=0
        if ey > 64: ey=64

        if f in [10,11,12,13,14]:
            prob_histo_fcst += pred[d,f,sy:ey,sx:ex]
            num_rpts += 1

    frequency = prob_histo_fcst / float(num_rpts)

    return frequency

def output_csv():
    # output 80-km grid locations
    #awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95, resolution=None, area_thresh=10000.)
    #lons, lats = awips.makegrid(93, 65)
    #np.savetxt('grid.out', np.array([lons.flatten(), lats.flatten(), thismask]).T, fmt='%.3f,%.3f,%.0d', header='lon,lat,mask')

    all_probs = []
    for i in range(6):
        print(i)
        predictions_gridded, labels_gridded = make_gridded_forecast(predictions_all, labels_all, dates_all, fhr_all)
        all_probs.append(predictions_gridded)

    all_probs.append(predictions_gridded_uh_smoothed[0,:]) #append smoothed UH forecasts
    all_probs = np.array(all_probs)

    fmask    = np.where( (unique_forecasts == '2011-04-27 00:00:00') )[0][0]
    idxarray = np.tile(np.arange(0,93*65)[np.newaxis,:], (36,1)).flatten()
    fhrarray = np.tile(np.arange(1,37)[:,np.newaxis], (1,93*65)).flatten()
    usmask   = np.tile(thismask[np.newaxis,:], (36,1)).flatten()

    all_probs = 100*all_probs[:,fmask,:,:].reshape((7,-1)) #should become (7,36*93*65)
    all_probs = np.where(all_probs<1, 0, all_probs)

    # want to only include areas where ANY prob is non-zero and within US mask area (smoothed UH likely has probs outside of US, maybe ML too)
    probmask =  ( np.any(all_probs, axis=0) & usmask )
 
    np.savetxt('test2.out', np.array([idxarray[probmask], fhrarray[probmask], all_probs[0,probmask], all_probs[1,probmask], all_probs[2,probmask], all_probs[3,probmask], all_probs[4,probmask], all_probs[5,probmask], all_probs[6,probmask]]).T,\
                            delimiter=',', fmt='%.0d', comments='', header='idx,fhr,psvr,pwind,phail,ptorn,psighail,psigwind,puh')

    #probarray =100*predictions_gridded[fmask,:,:].flatten()
    #probmask = (probarray >= 1)
    #np.savetxt('test.out', np.array([idxarray[probmask], fhrarray[probmask], probarray[probmask]]).T, delimiter=',', fmt='%.0d', header='idx,fhr,prob')


classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}
i = 0
print(classes[i])
numclasses = 6
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
#mask  = pickle.load(open('usamask_mod.pk', 'rb'))
smooth_sigma = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
smooth_sigma = [2.0]
thismask = mask.flatten()

fcst_file = 'predictions_nn_40km_2hr_hrrrv4-epoch30'
#fcst_file  = 'predictions_nn_40km_2hr_NSCmodel'
fcst_file = 'predictions_nn_40km_2hr_forhrrr'
#fcst_file  = 'predictions_nn_40km_2hr'
#fcst_file  = 'predictions_rf_40km_2hr_rt2020_test'
#fcst_file2  = 'predictions_rf_120km_2hr_rt2020'
#fcst_file  = 'predictions_nn_120km_2hr_uhonly_all'
#fcst_file  = 'predictions_nn_120km_2hr_envonly_all'
#fcst_file  = 'predictions_nn_120km_2hr_basicplus_all'
climo_file = '../climo_severe_40km_2hr.npz'
#climo_file = '../climo_severe_40km_2hr_torn.npz'

print('reading data')
# not converting to float32 due to small changes in computations?
#predictions_all = []
#for mem in range(1,6):
#    fcst_file = 'predictions_nn_40km_2hr_hrrrv4-epoch30-model%d'%mem
#    predictions_all_nn, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open(fcst_file, 'rb'))
#    predictions_all.append(predictions_all_nn)
#predictions_all = np.array(predictions_all)
#predictions_all_nn = np.mean(predictions_all, axis=0)
#predictions_all_rf, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open(fcst_file2, 'rb'))

df = pd.read_pickle(fcst_file)
print(df)
predictions_all_nn = df['predict%d'%i].values
labels_all = df['label%d'%i].values
dates_all = df['Date'].values
uh120_all = df['maxuhwin'].values
fhr_all = df['fhr'].values

#unique_forecasts, unique_fhr = np.unique(dates_all), np.unique(fhr_all)

unique_forecasts, unique_fhr = df['Date'].unique(), df['fhr'].unique()
num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)

print('making date arrays')
dates_dt   = np.array([ datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in unique_forecasts ])
months_all  = np.array([ d.month for d in dates_dt ])
doy_unique  = np.array([ d.timetuple().tm_yday for d in dates_dt ])

dates_dt   = np.repeat(dates_dt, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))
months_all = np.repeat(months_all, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))
doy_all    = np.repeat(doy_unique, num_fhr*65*93).reshape((num_dates,num_fhr,65,93))

print('reading climo')
data = np.load(climo_file)
climo = data['severe'][:]
climo_all = []
for doy in doy_unique:
    #arr3 = np.append( climo[doy,:,:,:], climo[doy+1,:12,:,:], axis=0 )
    arr3 = np.append( climo[doy,:,:,:], climo[doy+1,:24,:,:], axis=0 )
    climo_all.append(arr3)
climo_all = np.array(climo_all)

#predictions_all = (predictions_all_nn + predictions_all_rf) / 2.0
predictions_all = predictions_all_nn

print('Verifying %d forecast points'%predictions_all.shape[0])

#print(np.histogram(predictions_all[uh120_all<0.01,i], bins=np.arange(0,1.1,0.1)))
#highprobmask = (predictions_all[:,i] >= 0.1) & (uh120_all < 0.01) & (fhr_all >= 12) & (fhr_all <= 23)
#for d in np.unique(dates_all[highprobmask]): print(d, (dates_all[highprobmask]==d).sum())
#print(np.histogram(fhr_all[highprobmask], bins=np.arange(0,37)))

#predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>60), labels_all[:,i], dates_all, fhr_all)
#predictions_gridded_uh_smoothed        = smooth_gridded_forecast(predictions_gridded_uh)

#cape_shear_uh = False
#if cape_shear_uh:
#    # select UH threshold based on cape/shear or forecast hour
#    cape_thresh = [0,100,500,1000,2000,3000,4000,5000]
#    shear_thresh = [0,10,20,30,40,50]
#    uh_thresh_matrix = [[52,70,76,66,65],[54,68,76,68,60],[50,62,70,66,62],[40,58,68,68,72],[34,54,64,64,72],[30,48,58,58,22],[24,44,58,20,-999]]
#    uh_binary = np.ones_like(uh120_all)*-999
#    #for c in range(len(cape_thresh)-1):
#    #    for s in range(len(shear_thresh)-1):
#    uh_thresh_matrix = [14,20,28,38,46,54,60,64,68,78,76,74,70,68,66,62,60,60,58,60,60,60,60,60,60,62,64,68,72,74,76,76,76,76,76,78]
#    for f in range(1,37):
#        #print(c,s)
#        #envmask = (cape_all >= cape_thresh[c]) & (cape_all < cape_thresh[c+1]) & (shear_all >= shear_thresh[s]) & (shear_all < shear_thresh[s+1])
#        #uh_binary[envmask] = (uh120_all[envmask] >= uh_thresh_matrix[c][s])
#        print(f)
#        envmask = (fhr_all == f)
#        uh_binary[envmask] = (uh120_all[envmask] >= uh_thresh_matrix[f-1])
#    uh_binary = np.where(uh_binary < 0, uh120_all>50.0, uh_binary)
#    print(uh_binary, uh_binary.shape)

optimal_uh = False
if optimal_uh:
    predictions_gridded_uh, labels_gridded = make_gridded_forecast(uh120_all, labels_all, dates_all, fhr_all)

    optimal_uh_warmseason, num_rpts_warm = pickle.load(open('../optimal_uh_warmseason', 'rb'))
    optimal_uh_coolseason, num_rpts_cool = pickle.load(open('../optimal_uh_coolseason', 'rb'))

    months_all = months_all.reshape((num_dates, num_fhr, -1))
    months_all = months_all[:,0,0]

    uh_binary = []
    for k,m in enumerate(months_all):
        if m in [4,5,6,7]: this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_warmseason )
        else:              this_uh = ( predictions_gridded_uh[k,:] >= optimal_uh_coolseason )
        this_uh = this_uh.reshape((num_fhr,-1))[:,thismask]
        uh_binary.append(this_uh)

    uh_binary = np.array(uh_binary).flatten()

# compute BSS on spatial grid for fixed UH threshold
#bss_spatial_all = []
#for uh in range(30,90,10):
#    print(uh)
#    predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>uh).astype(np.int32), labels_all[:,i], dates_all, fhr_all)
#    predictions_gridded_uh_smoothed         = smooth_gridded_forecast(predictions_gridded_uh)
    
#    bss_spatial = compute_bss_spatial(predictions_gridded_uh_smoothed[0,:], labels_gridded)
#    bss_spatial_all.append(bss_spatial)
#bss_spatial = np.array(bss_spatial_all)
#bss_spatial = np.where(bss_spatial > 1, -999, bss_spatial)
#bss_spatial = np.where(bss_spatial < -1, -999, bss_spatial)
#bss_spatial = np.amax(bss_spatial, axis=0)
   
### compute BSS for updraft helicity forecasts
print('computing BSS for UH forecasts')
predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>75).astype(np.int32), labels_all, dates_all, fhr_all)
#predictions_gridded_uh, labels_gridded  = make_gridded_forecast((uh_binary).astype(np.int32), labels_all, dates_all, fhr_all)
predictions_gridded_uh_smoothed         = smooth_gridded_forecast(predictions_gridded_uh)

predictions_gridded, labels_gridded     = make_gridded_forecast(predictions_all, labels_all, dates_all, fhr_all)

print('outputting forecasts')
#output_csv()
#sys.exit()

# filter by forecast hour
#sfhr, efhr = 1,47
#sfhr, efhr = 2,22
sfhr, efhr = 14,34

num_fhr = (efhr - sfhr) + 1
predictions_gridded  =  predictions_gridded[:,sfhr:efhr+1,:]
labels_gridded = labels_gridded[:,sfhr:efhr+1,:]
predictions_gridded_uh = predictions_gridded_uh[:,sfhr:efhr+1,:]
predictions_gridded_uh_smoothed = predictions_gridded_uh_smoothed[:,:,sfhr:efhr+1,:]
climo_all = climo_all[:,sfhr:efhr+1,:]

#apply lower threshold to predictions
predictions_gridded_uh_smoothed = np.where(predictions_gridded_uh_smoothed<0.001, 0.0, predictions_gridded_uh_smoothed)

#compute BSS/AUC
bss_fhr_uh, bss_fhr_uh_boot, diffs_uh, diffs_climo  = compute_bss_fhr( predictions_gridded_uh_smoothed[0,:], labels_gridded )
bss_spatial_uh                                      = compute_bss_spatial( predictions_gridded_uh_smoothed[0,:], labels_gridded )
#auc_fhr_uh                                         = compute_auc_fhr_old( predictions_gridded_uh_smoothed[0,:], labels_gridded )
#auc_fhr_uh, auc_fhr_uh_boot                         = compute_auc_fhr( predictions_gridded_uh_smoothed[0,:], labels_gridded )
auc_all_uh                                          = compute_auc_all( predictions_gridded_uh_smoothed[0,:], labels_gridded )
bss_all_uh                                          = compute_bss( predictions_gridded_uh_smoothed[0,:], labels_gridded )
#true_prob_uh, fcst_prob_uh                          = calibration_curve(labels_gridded.flatten(), predictions_gridded_uh_smoothed[0,:].flatten(), n_bins=10)
true_prob_uh, fcst_prob_uh, boot_rel_uh, fcst_bin_sums_uh, avg_prob_uh = compute_reliability_all(labels_gridded.flatten(), predictions_gridded_uh_smoothed[0,:].flatten())

### compute BSS for ML predictions
print('computing BSS for ML forecasts')

#apply lower threshold to predictions
predictions_gridded = np.where(predictions_gridded<0.001, 0.0, predictions_gridded)
predictions_all     = np.where(predictions_all<0.001, 0.0, predictions_all)

#compute BSS/AUC
bss_fhr_ml, bss_fhr_ml_boot, diffs_ml, diffs_climo  = compute_bss_fhr(predictions_gridded, labels_gridded)
bss_spatial_ml                                      = compute_bss_spatial(predictions_gridded, labels_gridded)
#auc_fhr_ml, auc_fhr_ml_boot                         = compute_auc_fhr(predictions_gridded, labels_gridded)
auc_all_ml                                          = compute_auc_all(predictions_gridded, labels_gridded)
bss_all_ml                                          = compute_bss(predictions_gridded, labels_gridded)
#true_prob_ml, fcst_prob_ml                          = calibration_curve(labels_gridded.flatten(), predictions_gridded.flatten(), n_bins=10)
true_prob_ml, fcst_prob_ml, boot_rel_ml, fcst_bin_sums_ml, avg_prob_ml = compute_reliability_all(labels_gridded.flatten(), predictions_gridded.flatten())

bss_spatial_diff  =  bss_spatial_ml - bss_spatial_uh

print(bss_fhr_ml)
print('AUC ALL FHR UH/ML:', auc_all_uh, auc_all_ml)
print('BSS ALL FHR UH/ML:', bss_all_uh, bss_all_ml)

print('reliability for ML')
for a,p in enumerate(avg_prob_ml): print('%.3f, %.3f, %d'%(p, true_prob_ml[a], fcst_bin_sums_ml[a]))
print(repr(boot_rel_ml))

print('reliability for UH')
for a,p in enumerate(avg_prob_uh): print('%.3f, %.3f, %d'%(p, true_prob_uh[a], fcst_bin_sums_uh[a]))
print(repr(boot_rel_uh))
print(labels_gridded.mean())

#freq_uh = compute_2d_histo(predictions_gridded_uh_smoothed[0,:], labels_gridded, 'histo2d_uh.png')
#freq_ml = compute_2d_histo(predictions_gridded, labels_gridded, 'histo2d_ml.png')
    
#cmap = plt.get_cmap('RdBu_r')
#norm = colors.BoundaryNorm(np.arange(-0.05,0.06,0.01), ncolors=cmap.N, clip=True)
#plt.imshow(freq_ml-freq_uh,interpolation='nearest',cmap = cmap, norm=norm)
#plt.savefig('hist2d.png')
sys.exit()


### compute average difference of ML-SSPF forecast
average_diff = (predictions_gridded - predictions_gridded_uh).mean(axis=(0))

### compute BSS bootstrapped diffs ###
### bootstrap_bss takes a list of daily values (e.g., 497 forecasts)
bss_fhr_boot_diff, auc_fhr_boot_diff = [], []
for n,f in enumerate(range(sfhr,efhr+1)):
    bss_fhr_bs = bootstrap_bss( ( diffs_uh[:,n,:].sum(axis=1), diffs_climo[:,n,:].sum(axis=1) ), \
                                ( diffs_ml[:,n,:].sum(axis=1), diffs_climo[:,n,:].sum(axis=1) ) )
    bss_fhr_boot_diff.append(bss_fhr_bs)

    print(f, bss_fhr_bs)
bss_fhr_boot_diff = np.array(bss_fhr_boot_diff)

### compute ROCA bootstrapped diffs ###
auc_fhr_diff, auc_fhr_diff_boot = compute_auc_fhr( predictions_gridded_uh_smoothed[0,:], labels_gridded ,
                                               predictions_gridded, labels_gridded )

### plot BSS and ROCA timeseries ###
plot_stats_hourly(ptype='bss')
plot_stats_hourly(ptype='auc')
sys.exit()

### compute daily BSS ###
bss_daily_ml = compute_bss_daily(predictions_gridded, labels_gridded)
bss_daily_uh = compute_bss_daily(predictions_gridded_uh_smoothed[0,:], labels_gridded)
bss_daily_diff = bss_daily_ml - bss_daily_uh

### compute daily BSS with daily varying UH threshold ###
#bss_daily_uh_fixed_all = []
#for uh in range(10,50,10):
#    print(uh)
#    predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>uh).astype(np.int32), labels_all[:,i], dates_all, fhr_all)
#    predictions_gridded_uh_smoothed        = smooth_gridded_forecast(predictions_gridded_uh)
#
#    fmask = np.where( (unique_forecasts == '2012-12-19 00:00:00') )[0][0]
#    plot_forecast( np.amax( predictions_gridded_uh_smoothed[0,fmask,:], axis=0 ) , 'predictions_uh%d.png'%uh )
#    
#    bss_daily_uh_fixed = compute_bss_daily(predictions_gridded_uh_smoothed[0,:], labels_gridded)
#    bss_daily_uh_fixed_all.append(bss_daily_uh_fixed)
#    
#    print(bss_daily_uh_fixed[fmask])
#bss_daily_uh_fixed = np.amax(bss_daily_uh_fixed_all, axis=0)

### print out daily BSS
for d in range(num_dates):
    #print(unique_forecasts[d], bss_daily_ml[d], bss_daily_uh[d], bss_daily_diff[d], bss_daily_uh_fixed[d])
    print(unique_forecasts[d], bss_daily_ml[d], bss_daily_uh[d], bss_daily_diff[d])
print('UH better than ML:', num_dates, (bss_daily_diff>0).sum())

### plot stuff here   
preduh = predictions_gridded_uh_smoothed[0,:].reshape((num_dates,num_fhr,-1))[:,:,thismask]
plot_2d_hist(preduh.flatten(), predictions_all[:,i])

plot_daily_bss_scatter(bss_daily_ml, bss_daily_uh) 

plot_bss_spatial(bss_spatial_diff, 'bss_spatial_diff.png') #takes 2d array to plot
#plot_bss_spatial(bss_spatial_ml, 'bss_spatial_ml.png') #takes 2d array to plot
#plot_bss_spatial(bss_spatial_uh, 'bss_spatial_uh.png') #takes 2d array to plot

# plot forecasts for given date
fmask = np.where( (unique_forecasts == '2012-12-19 00:00:00') )[0][0]
#fmask = np.where( (unique_forecasts == '2012-07-01 00:00:00') )[0][0]
plot_forecast( np.amax( predictions_gridded_uh_smoothed[0,fmask,:], axis=0 ) , 'predictions_uh.png' )
#print(bss_daily_uh[fmask])
plot_forecast( np.amax( predictions_gridded[fmask,:], axis=0 ), 'predictions_ml.png' )
#print(bss_daily_ml[fmask])

#gridded_uh_values = grid_data(uh_all)
#plot_forecast( np.amax( gridded_uh_values[fmask,:], axis=0 ), 'gridded_uh.png' )
