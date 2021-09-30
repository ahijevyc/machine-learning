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

def plot_reliability_grid():
    #fig = plt.figure(figsize=(6,6.5))
    fig = plt.figure(figsize=(6,6))
    #gs = gridspec.GridSpec(2,1,height_ratios=[5,1])
    numrows, numcols = 2,2
    numpanels = numrows*numcols
    gs = gridspec.GridSpec(numrows,numcols)
    gs.update(hspace=0.07)

    if numpanels > 1: fontsize=6
    else: fontsize=8

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

        climo = 0.02
        ax1.plot([0,1], [(climo/2.0),climo + (1-climo)/2.0], color='0.25', linewidth=0.5) #no skill line
        ax1.plot([0,1], [climo,climo], color='0.25', linewidth=0.5) #climo line
        if n==0: ax1.text(0.95,climo+0.01, 'climo', fontdict={'fontsize': fontsize}, ha='center', va='center', rotation=0)


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

        fh = open('nn_validation_fhr13-36_all', 'r')
        data = [ line.split(',') for line in sorted(fh.readlines()) ]
        fh.close()

        # plot reliability curves - dont plot where CIs have NaNs
        inc = 0
        for exp in data:
            probbin = list(map(float, exp[4::2]))
            rel_values = list(map(float, exp[3::2]))
            exp_name = exp[0].replace('_2016', '')
            bss = exp[1]
            auc = exp[2]
            hr = int(exp_name.split('_')[3][0])
            
            if hr == 2 and ('nn1024_drop0.1' in exp_name or 'rf' in exp_name) and panels[n] in exp_name and 'True' not in exp_name:
                p = ax1.plot(probbin, rel_values, marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0, markerfacecolor='white', \
                              linewidth=2.5, label=exp_name)

                ax1.text(0.01,0.75-0.05*inc,bss,fontsize=6,color=p[0].get_color())
                ax1.text(0.16,0.75-0.05*inc,auc,fontsize=6,color=p[0].get_color())
                inc += 1

        # legend parameters
        leg = ax1.legend(loc=0, fontsize=4, borderaxespad=0.75, borderpad=0.5, numpoints=1, fancybox=True)
        leg.get_frame().set_lw(0.25)

    plt.savefig('reliability.pdf')

plot_reliability_grid()
#plot_2d_hist()
