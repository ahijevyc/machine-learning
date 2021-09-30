#!/usr/bin/env python

import numpy as np
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap,BoundaryNorm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr
import os
import pandas as pd

def readNCLcm(name):
    '''Read in NCL colormap for use in matplotlib'''
    rgb, appending = [], False
    fh = open('/glade/u/apps/ch/opt/ncl/6.4.0/intel/16.0.3/lib/ncarg/colormaps/%s.rgb'%name, 'r')
    for line in fh.read().splitlines():
        if appending: rgb.append(map(float,line.split()))
        if ''.join(line.split()) in ['#rgb',';RGB']: appending = True
    maxrgb = max([ x for y in rgb for x in y ])
    if maxrgb > 1: rgb = [ [ x/255.0 for x in a ] for a in rgb ]
    return rgb

def computeshr01(row):
    return np.sqrt(row['USHR1-potential_mean']**2 + row['VSHR1-potential_mean']**2)

def computeshr06(row):
    return np.sqrt(row['USHR6-potential_mean']**2 + row['VSHR6-potential_mean']**2)

def computeSTP(row):
    lclterm = ((2000.0-row['MLLCL-potential_mean'])/1000.0)
    lclterm = np.where(row['MLLCL-potential_mean']<1000, 1.0, lclterm)
    lclterm = np.where(row['MLLCL-potential_mean']>2000, 0.0, lclterm)

    shrterm = (row['shr06']/20.0)
    shrterm = np.where(row['shr06'] > 30, 1.5, shrterm)
    shrterm = np.where(row['shr06'] < 12.5, 0.0, shrterm)

    stp = (row['SBCAPE-potential_mean']/1500.0) * lclterm * (row['SRH01-potential_mean']/150.0) * shrterm
    return stp

def read_csv_files(r):
    # read in all CSV files for 1km forecasts
    tdate = sdate
    all_files = []
    while tdate <= edate:
        yyyymmdd = tdate.strftime('%Y%m%d')
        if r == '1km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_1km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        #elif r == '3km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        elif r == '3km': csv_file = '/glade/work/sobash/NSC_objects/track_data_ncarstorm_3km_csv/track_step_NCARSTORM_d01_%s-0000_13.csv'%(yyyymmdd)
        if os.path.exists(csv_file): all_files.append(csv_file)
        tdate += dateinc
    print 'Reading %s files'%(len(all_files))
    df = pd.concat((pd.read_csv(f) for f in all_files))

    # compute various diagnostic quantities
    #df['shr01'] = df.apply(computeshr01, axis=1)
    #df['shr06'] = df.apply(computeshr06, axis=1)
    #df['stp'] = df.apply(computeSTP, axis=1)   
    #df['ratio'] = df['RVORT1_MAX_max'] / df['RVORT5_MAX_max']
 
    return df, len(all_files)

sdate = dt.datetime(2010,10,1,0,0,0)
edate = dt.datetime(2017,10,1,0,0,0)
dateinc = dt.timedelta(days=1)

#for f in ['UP_HELI_MAX_max', 'UP_HELI_MAX01_max', 'UP_HELI_MAX03_max', 'RVORT1_MAX_max', 'RVORT5_MAX_max']:
#    print df[f].quantile([0.5,0.75,0.9,0.95,1.0])

### PLOT 2D HISTOGRAM OF STORM NUMBERS
cmap = ListedColormap(['#ffffff', '#eeeeee', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1])
#norm = BoundaryNorm([0,1,5,10,20,30,40,50,60,70,80,90,100,125,150,200,300,400,500], cmap.N)
#norm = BoundaryNorm(np.arange(0,0.05,0.001), cmap.N)
#h, x, y, p = plt.hist2d(df1['RVORT1_MAX_max'], df3['RVORT1_MAX_max'], bins=[np.arange(0,0.05,0.001),np.arange(0,0.05,0.001)], cmap=cmap, normed=True)
#h, x, y, p = plt.hist2d(df1['RVORT1_MAX_max'], df3['RVORT1_MAX_max'], bins=10, cmap=cmap, normed=True)

#df, numfcsts = read_csv_files('1km')
df, numfcsts = read_csv_files('3km')

print df[df['UP_HELI_MAX01_max'] > 14.362][['UP_HELI_MAX01_max', 'Centroid_Lat' ,'Centroid_Lon', 'Run_Date', 'Forecast_Hour']]

# filter here
#df = df[df['UP_HELI_MAX_max'] > 300]

import seaborn as sns
sns.set_style("white", {"axes.linewidth":0.5})

xticks, yticks = [0,0.01,0.02,0.03,0.04,0.05], [0,0.01,0.02,0.03,0.04,0.05]
xticks, yticks = [0,0.01,0.02], [0,0.01,0.02]
xticks, yticks = np.arange(0,2,0.1), range(0,800,25)
xfieldname, yfieldname = 'eccentricity', 'area'
#area,eccentricity,major_axis_length,minor_axis_length,orientation

xmax, ymax = xticks[-1], yticks[-1]

#xfield, yfield = df['RVORT1_MAX_max'], df['RVORT5_MAX_max']
xfield, yfield = df[xfieldname], df[yfieldname]

g = sns.jointplot(xfield, yfield, kind='hex', space=0, size=6, ratio=8, xlim=(0,xmax), ylim=(0,ymax), gridsize=60, mincnt=1, cmap=cmap, extent=(0,xmax,0,ymax), norm=LogNorm(vmin=1,vmax=10000))
#g = sns.JointGrid(x=xfield, y=yfield, space=0, size=6, ratio=8, xlim=(0,xmax), ylim=(0,ymax))
#map = g.ax_joint(plt.hexbin, xfield, yfield, gridsize=60, mincnt=1, cmap=cmap, extent=(0,xmax,0,ymax), norm=LogNorm(vmin=1,vmax=10000))
#g.plot_marginals(sns.distplot, kde=False, color=".5")
#sns.regplot(df1['RVORT1_MAX_max'], df1['RVORT5_MAX_max'], ax=g.ax_joint, scatter=False, color='red', line_kws={'linewidth':1.5})

#g = sns.JointGrid(x=xfield, y=yfield, xlim=(0,xmax), ylim=(0,ymax))
#g = g.plot_joint(sns.hexbin, cmap="Purples_d")

g.ax_joint.set_xlabel(xfieldname)
g.ax_joint.set_ylabel(yfieldname)
g.ax_joint.set_xticks(xticks)
g.ax_joint.set_yticks(yticks)

#cax = g.fig.add_axes([0.05,0.80,0.5,0.02])
#cb = plt.colorbar(g, cax=cax, orientation='horizontal')
#cb.ax.tick_params(axis='x',labeltop='on',labelbottom='off',top='on',pad=0)
  
g.ax_joint.plot([0,xmax], [0,ymax], color='k', linewidth=0.5)
g.ax_joint.grid(linewidth=0.5)

g.savefig('hexbin.png')

def plot_histo():
  mask1 = (uhmax>75) & (rvort>0.015)
  mask1 = (uhratio>1)
  print mask1.sum()

  ### PLOT HISTOGRAM
  #print np.percentile(var2[mask1], 25), np.median(var2[mask1]), np.percentile(var2[mask1], 75)
  #print np.percentile(var2[mask2], 25), np.median(var2[mask2]), np.percentile(var2[mask2], 75)
  #h, x, y, p = plt.hist2d(var1[mask], var2[mask], bins=[50,75], cmap=cmap, norm=LogNorm())
  #h, x, y, p = plt.hist2d(var1, var2, bins=[np.arange(0,0.025,0.001),np.arange(0,2.01,0.025)], cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #h, x, y, p = plt.hist2d(var1, var2, bins=[np.arange(0,0.025,0.001),np.arange(0,60,2)], cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #h, x, y, p = plt.hist2d(lcl_all[mask1], shr01_all[mask1], bins=[np.arange(0,3001,100),np.arange(0,41)], cmap=cmap, norm=LogNorm(vmin=1,vmax=1000))
  h, x, y, p = plt.hist2d(lcl_all[mask1], shr01_all[mask1], bins=[np.arange(0,3001,100),np.arange(0,41)], cmap=cmap, normed=True)
  #h, x, y, p = plt.hist2d(var1[mask1], var2[mask1], bins=[np.arange(0,3001,200),np.arange(0,80,2)], cmap=cmap, normed=True)
  #h, x, y, p = plt.hist2d(uhmax, var2, bins=[np.arange(0,500,10),np.arange(0,7000,100)], cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #h, x, y, p = plt.hist2d(var1[mask1], var2[mask1], bins=[np.arange(0,0.025,0.001),np.arange(0,600,10)], cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #h, x, y, p = plt.hist2d(var1, var2, bins=[np.arange(0,500,10),np.arange(0,500,10)], cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #plt.xlim((0,0.025))
  #plt.ylim((-700,0))
  plt.colorbar(pad=0.01)
  plt.savefig('histo.pdf', bbox_inches='tight')

def plot_hexbin():
  import seaborn as sns
  cmap = ListedColormap(['#eeeeee', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1])
 
  sns.set_style("white", {"axes.linewidth":0.5})

  # UH ratio vs RVORT
  #g = sns.JointGrid(x=rvort, y=uhratio, space=0, size=6, ratio=8, xlim=(0,0.03), ylim=(0,2.3))
  #map = g.ax_joint.hexbin(rvort, uhratio, gridsize=40, mincnt=1, cmap=cmap, norm=LogNorm(vmin=1,vmax=10000))
  #sns.regplot(rvort, uhratio, ax=g.ax_joint, scatter=False, color='red', line_kws={'linewidth':1.5}) 
  
  g = sns.JointGrid(x=uhmax, y=uh03max, space=0, size=6, ratio=8, xlim=(0,500), ylim=(0,500))
  map = g.ax_joint.hexbin(uhmax, uh03max, gridsize=60, mincnt=1, cmap=cmap, extent=(0,500,0,500), norm=LogNorm(vmin=1,vmax=10000))
  sns.regplot(uhmax, uh03max, ax=g.ax_joint, scatter=False, color='red', line_kws={'linewidth':1.5}) 
  
  #g = sns.JointGrid(x=rvort, y=uh03max, space=0, size=6, ratio=8, xlim=(0,0.03), ylim=(0,500))
  #map = g.ax_joint.hexbin(rvort, uh03max, gridsize=60, mincnt=1, cmap=cmap, extent=(0,0.03,0,500), norm=LogNorm(vmin=1,vmax=10000))
  #sns.regplot(rvort, uh03max, ax=g.ax_joint, scatter=False, color='red', line_kws={'linewidth':1.5}) 

  # plot diagonal line and grid
  g.ax_joint.plot([0,500], [0,500], color='k', linewidth=0.5)
  #g.ax_joint.plot([0,0.03], [1.0, 1.0], color='k', linewidth=0.5)
  #g.ax_joint.plot([0.015,0.015], [0,2.3], color='k', linewidth=0.5)
  g.ax_joint.grid(linewidth=0.5)

  g.plot_marginals(sns.distplot, kde=False, color='k')
  g.ax_marg_x.axvline(uhmax.mean(), color='k')
  #g.ax_marg_y.axhline(uhratio.mean(), color='k')
  g.ax_marg_y.axhline(uh03max.mean(), color='k')

  # compute pearson r2
  r = pearsonr(uhmax, uh03max)
  g.ax_joint.text(0.03,0.95,"r_sq = %.2f"%round(r[0]**2,2), color='red', transform=g.ax_joint.transAxes)
 
  #g.ax_joint.set_xlabel('RVORT1 (s-1)')
  #g.ax_joint.set_ylabel('UH Ratio (UH03/UH25)')
  g.ax_joint.set_xlabel('UH25 (m2/s2)')
  g.ax_joint.set_ylabel('UH03 (m2/s2)')

  cax = g.fig.add_axes([0.34,0.09,0.5,0.02])
  cb = plt.colorbar(map, cax=cax, orientation='horizontal')
  cb.ax.tick_params(axis='x',labeltop='on',labelbottom='off',top='on',pad=0)

  g.savefig('hexbin.png')

def plot_hexbin2():
  cmap = ListedColormap(['#eeeeee', '#cccccc', '#aaaaaa']+readNCLcm('precip2_17lev')[3:-1])
  mask1 = (rvort<0.005) & (uhmax>75)
  mask2 = (rvort>0.015) & (uhmax>75)
  mask3 = (rvort<0.005) & (uhmax>150)
  mask4 = (rvort>0.015) & (uhmax>150)
  print 'hexbin2', mask1.sum(), mask2.sum(), mask3.sum(), mask4.sum()

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)
  map1 = ax1.hexbin(lcl_all[mask1], shr01_all[mask1], gridsize=30, extent=(0,3000,0,30), mincnt=1, cmap=cmap, norm=LogNorm(vmin=1,vmax=100))
  map2 = ax2.hexbin(lcl_all[mask2], shr01_all[mask2], gridsize=30, extent=(0,3000,0,30), mincnt=1, cmap=cmap, norm=LogNorm(vmin=1,vmax=100))
  map3 = ax3.hexbin(lcl_all[mask3], shr01_all[mask3], gridsize=30, extent=(0,3000,0,30), mincnt=1, cmap=cmap, norm=LogNorm(vmin=1,vmax=100))
  map4 = ax4.hexbin(lcl_all[mask4], shr01_all[mask4], gridsize=30, extent=(0,3000,0,30), mincnt=1, cmap=cmap, norm=LogNorm(vmin=1,vmax=100))
  
  #pts = map1.get_offsets()
  #pts2 = map2.get_offsets()
  #counts = map1.get_array()
 
  ax1.grid(linewidth=0.5, color='grey')
  ax2.grid(linewidth=0.5, color='grey')
  ax3.grid(linewidth=0.5, color='grey')
  ax4.grid(linewidth=0.5, color='grey')

  ax1.set_xlim((0,3000))
  ax1.set_ylim((0,30))
  ax1.set_xticks(range(0,3000,500))
  ax3.set_xlabel('SBLCL (m AGL)')
  ax4.set_xlabel('SBLCL (m AGL)')
  ax1.set_ylabel('SHR01 (m/s)')
  ax3.set_ylabel('SHR01 (m/s)') 
  
  cax = fig.add_axes([0.21,0.02,0.6,0.02])
  cb = plt.colorbar(map1, cax=cax, orientation='horizontal')
  cb.ax.tick_params(axis='x',labeltop='off',labelbottom='on',top='on',pad=5)
  cb.set_ticks([1,2,3,4,5,7,10,20,30,40,50,70,100]) 
  cb.set_ticklabels([1,2,3,4,5,7,10,20,30,40,50,70,100]) 
  fig.subplots_adjust(wspace=0.03, hspace=0.03)

  plt.savefig('hexbin2.pdf', bbox_inches='tight') 

def plot_scatter():
  mask1 = (rvort<0.005) & (uhmax>150)
  mask2 = (rvort>0.015) & (uhmax>150)
  print mask1.sum(), mask2.sum()
  
  fig, ax1 = plt.subplots(figsize=(8,6))
  ax1.scatter(lcl_all[mask1], shr01_all[mask1], s=5, color='blue')
  ax1.scatter(lcl_all[mask2], shr01_all[mask2], s=5, color='red')

  ### LDA ###
  from sklearn.lda import LDA

  no = np.array([lcl_all[mask1], shr01_all[mask1]]).T
  yes = np.array([lcl_all[mask2], shr01_all[mask2]]).T
  clf = LDA()
  X = yes.tolist() + no.tolist()
  y1 = np.ones((yes.shape[0])) * 2
  y2 = np.ones((no.shape[0]))
  y = y1.tolist() + y2.tolist()

  y_pred = clf.fit(X, y).predict(X)
  nx, ny = 200, 100
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
  Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
  Z = Z[:, 1].reshape(xx.shape)
  plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

  ax1.grid(linewidth=0.5, color='grey')

  ax1.set_xlim((0,3000))
  ax1.set_ylim((0,40))
  ax1.set_xticks(range(0,3000,500))
  ax1.set_xlabel('SBLCL (m AGL)')
  ax1.set_ylabel('SHR01 (m/s)')

  plt.savefig('scatter.pdf', bbox_inches='tight')

def plot_violin():
  mask1 = (rvort > 0.015)
  mask2 = (uhratio > 1)
  f1, f2 = sbcape_all, sbcin_all

  import seaborn as sns
  sns.set_style("whitegrid")
  sns.violinplot(data=[rvort, uhratio])
  plt.savefig('violin.png')

### PLOT BOXPLOTS OF STORM PROPERTIES
def plot_boxplot():
  #mask1 = (rvort > 0.015)
  #mask2 = (uhratio > 1)
  print mask1.sum(), mask2.sum()
  mask1sum, mask2sum, mask3sum = mask1.sum(), mask2.sum(), mask3.sum()

  #f1, f2 = srh01_all, srh03_all
  #f1, f2 = shr01_all, shr06_all
  f1, f2 = stp_all, stp_all
  plot_f2 = False

  print 'number of objects', stp_all.shape
  print 'number of UH>75 objects', (uhmax>75).sum()

  fig, ax = plt.subplots(figsize=(9,6))
  ax.yaxis.grid(True, color='lightgrey', alpha=0.5, linestyle='solid')

  bp = ax.boxplot([f1[mask1], f1[mask2], f1[mask3]], \
              widths=0.1, positions=[1,1.4,1.8], \
              labels=['weak\nRVORT1\nN=%d'%mask1sum, 'mod\nRVORT1\nN=%d'%mask2sum, 'strong\nRVORT1\nN=%d'%mask3sum], \
              whis=[10,90], whiskerprops={'color':'k', 'linestyle':'solid', 'linewidth':0.5}, \
              boxprops={'color':'k', 'linewidth':0.5}, medianprops={'color':'black', 'linewidth':0.75}, showfliers=False, patch_artist=True)
  #bp = ax.boxplot([f1[mask1], f1[mask2], f1[mask3], f2[mask1], f2[mask2], f2[mask3]], \
  #            widths=0.1, positions=[1,1.4,1.8,2.5,2.9,3.3], \

  # subsets for UH
  #bp = ax.boxplot([f1[rvort<0.005], f1[rvort>0.015], f1[uhmax<150], f1[uhmax>150]], \
  #            widths=0.1, positions=[1,1.4,2.0,2.4], \
  #            labels=['UH25\nN=%d'%mask1sum, 'UH03\nN=%d'%mask1sum, 'UH25\nN=%d'%mask2sum, 'UH03\nN=%d'%mask2sum], \
  #            whis=[10,90], whiskerprops={'color':'k', 'linestyle':'solid', 'linewidth':0.5}, \
  #            boxprops={'color':'k', 'linewidth':0.5}, medianprops={'color':'black', 'linewidth':0.75}, showfliers=False, patch_artist=True)

  # subsets for sbcape, area, duration
  #bp = ax.boxplot([f1[mask1], f1[mask2]], \
  #            widths=0.1, positions=[1,1.4], \
  #            labels=['strong\nRVORT1\nN=%d'%mask1.sum(), 'UHratio>1\nN=%d'%mask2.sum()], \
  #            whis=[10,90], whiskerprops={'color':'k', 'linestyle':'solid', 'linewidth':0.5}, \
  #            boxprops={'color':'k', 'linewidth':0.5}, medianprops={'color':'black', 'linewidth':0.75}, showfliers=False, patch_artist=True)
  
  # subsets for sbcape, area, duration
  #bp = ax.boxplot([srh01_all[mask1], srh01_all_max[mask1]], \
  #            widths=0.1, positions=[1,1.4], \
  #            labels=['SRHEL01', 'SRHEL01_MAX'], \
  #            whis=[10,90], whiskerprops={'color':'k', 'linestyle':'solid', 'linewidth':0.5}, \
  #            boxprops={'color':'k', 'linewidth':0.5}, medianprops={'color':'black', 'linewidth':0.75}, showfliers=False, patch_artist=True)

  ax.yaxis.grid(True, color='#dddddd', linewidth=0.75, linestyle='solid', zorder=-1)
  ax.set_axisbelow(True)

  for patch in bp['boxes']: patch.set_facecolor('lightgrey')

  fontdict = {'fontsize':10, 'fontweight':'bold'}
  
  for perc in [75,50,25]:
      #ax.text(1+0.06,   np.percentile(f1[mask1], perc), '%.0f'%np.percentile(f1[mask1], perc), verticalalignment='center', fontdict=fontdict)
      #ax.text(1.4+0.06, np.percentile(f2[mask1], perc), '%.0f'%np.percentile(f2[mask1], perc), verticalalignment='center', fontdict=fontdict)
      
      if perc==25: move=0.05
      else: move=0.0
      ax.text(1+0.06,   np.percentile(f1[mask1], perc)-move, '%.1f'%np.percentile(f1[mask1], perc), verticalalignment='center', fontdict=fontdict)
      ax.text(1.4+0.06, np.percentile(f1[mask2], perc), '%.1f'%np.percentile(f1[mask2], perc), verticalalignment='center', fontdict=fontdict)
      ax.text(1.8+0.06, np.percentile(f1[mask3], perc), '%.1f'%np.percentile(f1[mask3], perc), verticalalignment='center', fontdict=fontdict)
      #ax.text(1.4+0.06, np.percentile(srh01_all_max[mask1], perc), '%.0f'%np.percentile(srh01_all_max[mask1], perc), verticalalignment='center', fontdict=fontdict)

      if plot_f2:
        ax.text(2.0+0.06, np.percentile(f1[mask2], perc), '%.0f'%np.percentile(f1[mask2], perc), verticalalignment='center', fontdict=fontdict)
        ax.text(2.4+0.06, np.percentile(f2[mask2], perc), '%.0f'%np.percentile(f2[mask2], perc), verticalalignment='center', fontdict=fontdict)
        #ax.text(2.5+0.06, np.percentile(f2[mask1], perc), '%.0f'%np.percentile(f2[mask1], perc), verticalalignment='center', fontdict=fontdict)
        #ax.text(2.9+0.06, np.percentile(f2[mask2], perc), '%.0f'%np.percentile(f2[mask2], perc), verticalalignment='center', fontdict=fontdict)
        #ax.text(3.3+0.06, np.percentile(f2[mask3], perc), '%.0f'%np.percentile(f2[mask3], perc), verticalalignment='center', fontdict=fontdict)
      
  #ax.text(1.4, 575, 'SRHEL01', horizontalalignment='center', verticalalignment='center', fontdict={'fontsize':13, 'fontweight':'bold'})
  #ax.text(2.9, 575, 'SRHEL03', horizontalalignment='center', verticalalignment='center', fontdict={'fontsize':13, 'fontweight':'bold'})
  
  #plt.ylabel('Storm-relative helicity (m2/s2)')
  #plt.ylabel('Shear (m/s)')
  #plt.ylim((0,2000))
  plt.savefig('boxplot.pdf')

#plot_boxplot()
#plot_brooks_plot()
#plot_hexbin()
#plot_scatter()
#plot_hexbin2()
#plot_violin()
#plot_histo()
