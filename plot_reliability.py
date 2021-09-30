#!/usr/bin/env python

# PLOT FSS DATA COMPUTED USING compute_sspf_fss.py
import matplotlib
matplotlib.use('Agg')
from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import *
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pickle
import sys
import matplotlib.ticker as mtick

mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16.0

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    import os
    rgb, appending = [], False
    rgb_dir_ch = '/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps'
    fh = open('%s/%s.rgb'%(rgb_dir_ch,name), 'r')

    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def plot_2d_hist():
    predictions_rf = pickle.load(open('predictions_rf_2016', 'rb'))
    predictions_nn = pickle.load(open('predictions_nn_2016', 'rb'))

    predictions_rf = np.array(predictions_rf)[:,0]
    predictions_nn = np.array(predictions_nn)[:,0]

    rf_histo, bins = np.histogram(predictions_rf, bins=np.arange(0,1.1,0.1))
    nn_histo, bins = np.histogram(predictions_nn, bins=np.arange(0,1.1,0.1))

    for i in range(0,10):
        print(rf_histo[i], nn_histo[i])

    fig = plt.figure(figsize=(9,9))
    plt.hist2d(predictions_rf, predictions_nn, bins=50, norm=colors.LogNorm())
    plt.plot([0,1], [0,1], color='k') 
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('predictions_rf')
    plt.ylabel('predictions_nn')
    plt.savefig('hist2d.png')

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

def plot_reliability_grid():
    #fig = plt.figure(figsize=(6,6.5))
    fig = plt.figure(figsize=(6,6))
    #gs = gridspec.GridSpec(2,1,height_ratios=[5,1])
    numrows, numcols = 1,1
    numpanels = numrows*numcols
    gs = gridspec.GridSpec(numrows,numcols)
    gs.update(hspace=0.07)

    if numpanels > 1: fontsize=6
    else: fontsize=13

    panels = ['40km', '80km', '120km', '160km']
    for n,ax in enumerate(gs):
        ax1 = plt.subplot(ax)

        # AXES1: LINE CHART OF RELIABILITY 
        # add gridlines, reference lines, adjust spine widths 
        ax1.grid(color='0.7', linewidth=0.25) #gridlines 
        ax1.plot([0,1], [0,1], color='0.25', linewidth=0.5) #perfect reliability

        if n == 0:
            ax1.text(0.94,0.96, 'perfect', fontdict={'fontsize': fontsize}, ha='center', va='center', rotation=40)
            ax1.text(0.94,0.49, 'no skill', fontdict={'fontsize': fontsize}, ha='center', va='center', rotation=25)

        # adjust axes labels/ticks/etc
        ax1.set_xlim((0,1.0))
        ax1.set_ylim((0,1.0))
        ax1.set_xticks(np.arange(0,1.01,0.1))
        ax1.set_yticks(np.arange(0,1.01,0.1))
        #ax1.set_xticklabels(np.arange(0,1.01,0.1))
        ax1.tick_params(bottom=True, axis='both', width=0.5, direction='out', labelsize=fontsize, labelbottom=False)
        if n > (numpanels)-numcols-1:
            ax1.tick_params(labelbottom=True)
            ax1.set_xlabel('Forecast Probability', fontsize=fontsize, labelpad=4)
        if n%numcols < 1: ax1.set_ylabel('Observed relative frequency', fontsize=fontsize)
        for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(0.5)

        # 40-km reliability 
        #data = [ 
        #         [[0.00120739, 0.11923251, 0.21987432, 0.31511414, 0.41412506, 0.4729771, 0.41486068, 0.28571429],
        #           [6.78699441e-04, 1.39616714e-01, 2.41611512e-01, 3.41300943e-01, 4.40067477e-01, 5.39261812e-01, 6.34115246e-01, 7.26199782e-01]],
        #         [[0.00102719, 0.15541352, 0.26122159, 0.34233644, 0.43460071, 0.51375815, 0.56154629, 0.62184874, 0.65909091, 0.75],
        #          [0.00093607, 0.13861964, 0.23965505, 0.34023176, 0.44085164, 0.54233039, 0.64152779, 0.73657118, 0.84121603, 0.91969828]],
       # 
       #        ]

        # 120-km reliability
        #data = [  
        #         [[0.00340333, 0.12301829, 0.20235975, 0.29556196, 0.40254509, 0.51459317, 0.62047644, 0.71622143, 0.77653673, 0.7776873],
        #          [0.00093482, 0.14413546, 0.24654935, 0.3464336, 0.44662406, 0.54633743, 0.64578715, 0.74284782, 0.83693879, 0.92048133]],
        #          [[0.00155997, 0.13992981, 0.2433555, 0.34550066, 0.44555966, 0.54566238, 0.63952615, 0.71616302, 0.80501849, 0.87874315],
        #          [0.00153965, 0.14360895, 0.24607046, 0.34680939, 0.44685448, 0.54656278, 0.6457147, 0.7446683, 0.8414275, 0.92941808]],
        #       ]
        #climo = 0.008 

        # 120-km
        fcst_probs = np.arange(0.05,1.05,0.1)
        true_probs_ml = [0.00792971, 0.13992981, 0.2433555, 0.34550066, 0.44555966, 0.54566238, 0.6395261,0.71616302, 0.80501849, 0.87874315]
        true_probs_uh = [0.01686787,0.14241948,0.21930795,0.30760426,0.40733623,0.51598233,0.62079043,0.71622143,0.77653673,0.7776873]
        num_fcsts_ml = [19628589,916095,500038,324859,221582,147846,94545,53344,23792,4742]
        num_fcsts_uh = [20222708,611693,383438,258592,180556,123355,79046,41476,13340,1228]
        climo = 0.008 
        
        true_probs_ml_boot = np.array([[0.00754523, 0.00792603, 0.00831967, 0.00792613],
        [0.13630881, 0.13999735, 0.14356978, 0.13999769],
        [0.23779388, 0.2434317 , 0.24894266, 0.24343247],
        [0.33765709, 0.34558716, 0.35331962, 0.34558787],
        [0.43537898, 0.44567427, 0.45581711, 0.44567524],
        [0.53339636, 0.54578594, 0.55804524, 0.54578785],
        [0.62504035, 0.63966028, 0.65457679, 0.63966165],
        [0.6974519 , 0.71635781, 0.73491375, 0.71635942],
        [0.78135283, 0.8053446 , 0.8274639 , 0.80534886],
        [0.8340508 , 0.88017311, 0.9183543 , 0.88018026]])
        
        true_probs_uh_boot = np.array([[0.01608559, 0.01687165, 0.01766053, 0.01687165],
        [0.13825954, 0.14245768, 0.14663352, 0.14245771],
        [0.21382959, 0.21935324, 0.22478837, 0.21935361],
        [0.3007503 , 0.30766366, 0.31457597, 0.30766406],
        [0.39849105, 0.40752082, 0.41636593, 0.40752114],
        [0.50484722, 0.51618883, 0.52727691, 0.51618903],
        [0.60630952, 0.62091695, 0.63505995, 0.62091712],
        [0.69766512, 0.7166529 , 0.73473154, 0.71665298],
        [0.74649198, 0.77741083, 0.80591589, 0.77741245],
        [0.68857356, 0.78087279, 0.86206897, 0.7808818 ]])  
 
        # 40-km
        '''
        true_probs_ml = [0.00486867,0.15541352,0.26122159,0.34233644,0.43460071,0.51375815,0.56154629,0.62184874,0.65909091,0.75]
        num_fcsts_ml =  [21431114,354686,91520,25954,8127,2762,983,238,44,4]
        true_probs_uh = [0.00571203,0.12503337,0.22362869,0.31679939,0.41468468,0.4729771,0.41486068,0.28571429,np.nan,np.nan]
        num_fcsts_uh = [21477985,292138, 96222,34031,11100,3275,646,35,0,0]
        climo = 0.002

        true_probs_ml_boot = np.array([[0.00463171, 0.00486868, 0.00511231, 0.0048687 ],
        [0.14984927, 0.15540201, 0.1612947 , 0.15540221],
        [0.25082515, 0.26121765, 0.27232332, 0.26122044],
        [0.32532776, 0.34229198, 0.35987798, 0.34229304],
        [0.40487866, 0.43490235, 0.46357874, 0.43490253],
        [0.46723869, 0.51398052, 0.55227955, 0.51398896],
        [0.49204864, 0.5620915 , 0.62320574, 0.56209838],
        [0.49799197, 0.62582781, 0.74879227, 0.62582781],
        [0.38888889, 0.66071429, 0.83783784,        np.nan],
        [0.        , 0.8       , 1.        ,        np.nan]])

        true_probs_uh_boot = np.array([[0.00545171, 0.00570945, 0.0059753 , 0.00570946],
        [0.1208006 , 0.12503725, 0.12935181, 0.12503754],
        [0.21459877, 0.22367844, 0.23309508, 0.22368112],
        [0.30012752, 0.31697711, 0.33378616, 0.31697715],
        [0.38432226, 0.4147866 , 0.44674731, 0.41479006],
        [0.42107143, 0.4738806 , 0.52600075, 0.47388809],
        [0.29382304, 0.41609823, 0.54102921, 0.41610832],
        [0.        , 0.28571429,        np.nan,        np.nan],
        [       np.nan,        np.nan,        np.nan,        np.nan],
        [       np.nan,        np.nan,        np.nan,        np.nan]])
        '''
        ax1.plot([0,1], [(climo/2.0),climo + (1-climo)/2.0], color='0.25', linewidth=0.5) #no skill line
        ax1.plot([0,1], [climo,climo], color='0.25', linewidth=0.5) #climo line
        if n==0: ax1.text(0.95,climo+0.01, 'climo', fontdict={'fontsize': fontsize}, ha='center', va='center', rotation=0)

        # plot reliability curves - dont plot where CIs have NaNs
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for n,row in enumerate(true_probs_uh_boot): ax1.axvline(x=fcst_probs[n]+0.0025, ymin=row[0], ymax=row[2], color=colors[0], lw=1.5, alpha=0.9)
        for n,row in enumerate(true_probs_ml_boot): ax1.axvline(x=fcst_probs[n]-0.0025, ymin=row[0], ymax=row[2], color=colors[1], lw=1.5, alpha=0.9)
        
        p = ax1.plot(fcst_probs, true_probs_uh_boot[:,1], marker='o', lw=5, ms=4, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white')
        p = ax1.plot(fcst_probs, true_probs_ml_boot[:,1], marker='o', lw=5, ms=4, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white')

        # legend parameters
        #leg = ax1.legend(loc=0, fontsize=4, borderaxespad=0.75, borderpad=0.5, numpoints=1, fancybox=True)
        #leg.get_frame().set_lw(0.25)

    plt.savefig('reliability.pdf')

def plot_bars():
    fig = plt.figure(figsize=(6,2.1))
    numrows, numcols = 1,1
    numpanels = numrows*numcols
    gs = gridspec.GridSpec(numrows,numcols)
    gs.update(hspace=0.07)

    if numpanels > 1: fontsize=6
    else: fontsize=13

    ax1 = plt.subplot(gs[0])

    # adjust axes labels/ticks/etc
    ax1.grid(color='0.7', linewidth=0.25) #gridlines 
    ax1.set_xlim((0,0.95))
    ax1.set_ylim((1e0,1e8))
    ax1.set_xticks(np.arange(-0.05,0.96,0.1))
    ax1.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax1.set_yscale('log')
    ax1.set_yticks([1e0,1e2,1e4,1e6,1e8])
    ax1.tick_params(bottom=True, axis='both', width=0.5, direction='out', labelsize=fontsize, labelbottom=True)
    for axis in ['top','bottom','left','right']: ax1.spines[axis].set_linewidth(0.5)
    
    #120-km    
    num_fcsts_ml = [19628589,916095,500038,324859,221582,147846,94545,53344,23792,4742]
    num_fcsts_uh = [20222708,611693,383438,258592,180556,123355,79046,41476,13340,1228]

    #40-km
    #num_fcsts_ml =  [21431114,354686,91520,25954,8127,2762,983,238,44,4]
    #num_fcsts_uh = [21477985,292138, 96222,34031,11100,3275,646,35,0,0]

    ax1.bar(np.arange(0,1,0.1)-0.04, num_fcsts_uh, width=0.04, align='edge',zorder=1000)
    ax1.bar(np.arange(0,1,0.1), num_fcsts_ml, width=0.04, align='edge', zorder=1000)

    plt.savefig('bar.pdf')

plot_reliability_grid()
plot_bars()
#plot_2d_hist()
