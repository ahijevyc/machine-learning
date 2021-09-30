#!/usr/bin/env python

import numpy as np
#import cPickle as pickle
import pickle

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

# bootstrap a reliability curve
def bootstrap_rel(fcst_yes, obs_yes, bins=21, alpha=0.95, B=10000):
    n = fcst_yes.shape[0] #number of days
    idx = np.random.randint(0, n, (B,n))

    fcst_yes_draw = fcst_yes[idx,:]
    obs_yes_draw  = obs_yes[idx,:]

    # sum over number of days 
    fcst_yes_draw_sum = np.sum(fcst_yes_draw, axis=1)
    obs_yes_draw_sum  = np.sum(obs_yes_draw, axis=1)

    rel = obs_yes_draw_sum/fcst_yes_draw_sum

    cis = []
    for k in range(bins):
        cis.append(createCI(rel[:,k], B, 1-alpha))

    return np.array(cis)

# bootstrap one set of FSS values
def bootstrap_fss(fss1=None, fss2=None, alpha=0.95, B=10000):
    fbs, fbsworst = fss1

    n   = fbs.size
    idx = np.random.randint(0, n, (B,n))
   
    fbs_draw       = fbs[idx]
    fbs_worst_draw = fbsworst[idx]

    fbs_sum        = np.sum(fbs_draw, axis=1)
    fbs_worst_sum  = np.sum(fbs_worst_draw, axis=1)

    fss = (1 - (fbs_sum/fbs_worst_sum))
    
    if fss2 is not None:
        fbs, fbsworst = fss2
        fbs_draw       = fbs[idx]
        fbs_worst_draw = fbsworst[idx]

        fbs_sum        = np.sum(fbs_draw, axis=1)
        fbs_worst_sum  = np.sum(fbs_worst_draw, axis=1)

        fss2 = (1 - (fbs_sum/fbs_worst_sum))
        stat = fss2 - fss
    else:
        stat = fss

    return createCI(stat,B,1-alpha)

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

def bootstrap_auc_null_distribution(auc1=None, auc2=None, alpha=0.99, B=10000):
    hits1, miss1, fals1, cneg1 = auc1
    hits2, miss2, fals2, cneg2 = auc2

    n = hits1.shape[0] #number of days
    
    idx = np.random.randint(2, size=(B,n), dtype='bool')
    idx = np.array([idx, np.logical_not(idx)])

    hits_combined = np.array([hits1, hits2])
    miss_combined = np.array([miss1, miss2])
    fals_combined = np.array([fals1, fals2])
    cneg_combined = np.array([cneg1, cneg2])

    aucs_a, aucs_b = [], []
    for i in range(B):
        ### CONSTRUCT RANDOM CONTINGENCY TABLES FROM EITHER FORECAST 1 or 2, DO THIS B TIMES
        # pick randomly either from forecast 1 or forecast 2, B times
        hits_a = hits_combined[idx[:,i,:],:]
        miss_a = miss_combined[idx[:,i,:],:]
        fals_a = fals_combined[idx[:,i,:],:]
        cneg_a = cneg_combined[idx[:,i,:],:]
 
        # compute contingency table by summing over forecasts
        hits_a_sum = np.sum(hits_a, axis=0)
        miss_a_sum = np.sum(miss_a, axis=0)
        fals_a_sum = np.sum(fals_a, axis=0)
        cneg_a_sum = np.sum(cneg_a, axis=0)
   
        # compute pod, pofd using those contingency tables
        pod  = hits_a_sum / (hits_a_sum + miss_a_sum)
        pofd = fals_a_sum / (cneg_a_sum + fals_a_sum)
        pod_a, pofd_a = np.nan_to_num(pod), np.nan_to_num(pofd)
 
        auc = 0
        for j in range(0,pod_a.shape[0]-1):
            auc += ((pod_a[j]+pod_a[j+1])/2.0)*(pofd_a[j]-pofd_a[j+1])
        aucs_a.append(auc)

        ### CONSTRUCT RANDOM CONTINGENCY TABLES FROM EITHER FORECAST 1 or 2, DO THIS B TIMES
        idx = np.logical_not(idx)
        hits_b = hits_combined[idx[:,i,:],:]
        miss_b = miss_combined[idx[:,i,:],:]
        fals_b = fals_combined[idx[:,i,:],:]
        cneg_b = cneg_combined[idx[:,i,:],:]
    
        # compute contingency table by summing over forecasts
        hits_b_sum = np.sum(hits_b, axis=0)
        miss_b_sum = np.sum(miss_b, axis=0)
        fals_b_sum = np.sum(fals_b, axis=0)
        cneg_b_sum = np.sum(cneg_b, axis=0)
 
        # compute pod, pofd using those contingency tables
        pod  = hits_b_sum / (hits_b_sum + miss_b_sum)
        pofd = fals_b_sum / (cneg_b_sum + fals_b_sum)
        pod_b, pofd_b = np.nan_to_num(pod), np.nan_to_num(pofd) 
        
        auc = 0
        for j in range(0,pod_b.shape[0]-1):
            auc += ((pod_b[j]+pod_b[j+1])/2.0)*(pofd_b[j]-pofd_b[j+1])
        aucs_b.append(auc)

    # null distribution of AUC differences
    auc_diffs = np.array(aucs_b) - np.array(aucs_a)
    
    return createCI(auc_diffs, B, 1-alpha)

def bootstrap_auc(auc1=None, auc2=None, alpha=0.99, B=10000):
    hits, miss, fals, cneg = auc1
    n = hits.shape[0]
    idx = np.random.randint(0, n, (B,n))

    # get B random samples of n days
    hits_draw = hits[idx,:]
    miss_draw = miss[idx,:]
    fals_draw = fals[idx,:]
    cneg_draw = cneg[idx,:]

    # compute contingency table by summing those elements
    hits_sum = np.sum(hits_draw, axis=1)
    miss_sum = np.sum(miss_draw, axis=1)
    fals_sum = np.sum(fals_draw, axis=1)
    cneg_sum = np.sum(cneg_draw, axis=1)
   
    # compute pod, pofd using those contingency tables
    pod  = hits_sum / (hits_sum + miss_sum)
    pofd = fals_sum / (cneg_sum + fals_sum)
    pod, pofd = np.nan_to_num(pod), np.nan_to_num(pofd)

    # for each sample, compute an AUC 
    aucs = []
    for i in range(0,B):
        auc = 0
        for j in range(0,pod.shape[1]-1):
            auc += ((pod[i,j]+pod[i,j+1])/2.0)*(pofd[i,j]-pofd[i,j+1])
        aucs.append(auc)

    if auc2 is not None:
        hits, miss, fals, cneg = auc2

        hits_draw = hits[idx,:]
        miss_draw = miss[idx,:]
        fals_draw = fals[idx,:]
        cneg_draw = cneg[idx,:]

        hits_sum = np.sum(hits_draw, axis=1)
        miss_sum = np.sum(miss_draw, axis=1)
        fals_sum = np.sum(fals_draw, axis=1)
        cneg_sum = np.sum(cneg_draw, axis=1)

        pod  = hits_sum / (hits_sum + miss_sum)
        pofd = fals_sum / (cneg_sum + fals_sum)
        pod, pofd = np.nan_to_num(pod), np.nan_to_num(pofd)

        aucs2 = []
        for i in range(0,B):
            auc = 0
            for j in range(0,pod.shape[1]-1):
                auc += ((pod[i,j]+pod[i,j+1])/2.0)*(pofd[i,j]-pofd[i,j+1])
            aucs2.append(auc)

        stat = np.array(aucs2) - np.array(aucs)

    else:
        stat = aucs
 
    return createCI(np.array(stat), B, 1-alpha)

def bootstrap_ets(ets1=None, ets2=None, alpha=0.99, B=10000):
    hits, miss, fals, cneg = ets1
    n = hits.shape[0]
    idx = np.random.randint(0, n, (B,n))

    # get B random samples of n days
    hits_draw = hits[idx]
    miss_draw = miss[idx]
    fals_draw = fals[idx]
    cneg_draw = cneg[idx]

    # compute contingency table by summing those elements
    hits_sum = np.sum(hits_draw, axis=1)
    miss_sum = np.sum(miss_draw, axis=1)
    fals_sum = np.sum(fals_draw, axis=1)
    cneg_sum = np.sum(cneg_draw, axis=1)

    hits_random = (hits_sum + miss_sum)*(hits_sum + fals_sum) / (hits_sum + miss_sum + fals_sum + cneg_sum)
    ets_all = (hits_sum-hits_random) / (hits_sum + fals_sum + miss_sum - hits_random)

    if ets2 is not None:
        hits, miss, fals, cneg = ets2

        hits_draw = hits[idx]
        miss_draw = miss[idx]
        fals_draw = fals[idx]
        cneg_draw = cneg[idx]

        hits_sum = np.sum(hits_draw, axis=1)
        miss_sum = np.sum(miss_draw, axis=1)
        fals_sum = np.sum(fals_draw, axis=1)
        cneg_sum = np.sum(cneg_draw, axis=1)
        
        hits_random = (hits_sum + miss_sum)*(hits_sum + fals_sum) / (hits_sum + miss_sum + fals_sum + cneg_sum)
        ets_all2 = (hits_sum-hits_random)/ (hits_sum + fals_sum + miss_sum - hits_random )
        
        stat = np.array(ets_all2) - np.array(ets_all)

    else:
        stat = ets_all

    return createCI(np.array(stat), B, 1-alpha)


if __name__ == '__main__':
  ### BOOTSTRAP RELIABILITY ###
  #fcst_sums = pickle.load(open('fcst_bin_sums_daily_day1_obsall_NCAR2013_00z_UP_HELI_MAX.pk', 'r'))
  #obs_sums = pickle.load(open('obs_bin_sums_daily_day1_obsall_NCAR2013_00z_UP_HELI_MAX.pk', 'r'))
  #for i in range(10): print bootstrap_rel(fcst_sums[:,i,8,3], obs_sums[:,i,8,3])
  fbs, fbsworst = {}, {}

  fbs['rvort'] = pickle.load(open('fbs_day1_NCAR2015_RVORT1_MAX_obstorn_00z.pk', 'r'))
  fbs['uh']    = pickle.load(open('fbs_day1_NCAR2015_UP_HELI_MAX_obstorn_00z.pk', 'r'))
  fbs['uh03']  = pickle.load(open('fbs_day1_NCAR2015_UP_HELI_MAX03_obstorn_00z.pk', 'r'))
  fbsworst['rvort']   = pickle.load(open('fbsworst_day1_NCAR2015_RVORT1_MAX_obstorn_00z.pk', 'r'))
  fbsworst['uh']   = pickle.load(open('fbsworst_day1_NCAR2015_UP_HELI_MAX_obstorn_00z.pk', 'r'))
  fbsworst['uh03']   = pickle.load(open('fbsworst_day1_NCAR2015_UP_HELI_MAX03_obstorn_00z.pk', 'r'))

  fbs['rvort'] = pickle.load(open('fbs_day1_NCAR2015_RVORT1_MAX_obsall_00z.pk', 'r'))
  fbs['uh']    = pickle.load(open('fbs_day1_NCAR2015_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbs['uh03']  = pickle.load(open('fbs_day1_NCAR2015_UP_HELI_MAX03_obsall_00z.pk', 'r'))
  fbsworst['rvort']   = pickle.load(open('fbsworst_day1_NCAR2015_RVORT1_MAX_obsall_00z.pk', 'r'))
  fbsworst['uh']   = pickle.load(open('fbsworst_day1_NCAR2015_UP_HELI_MAX_obsall_00z.pk', 'r'))

  fbs['ncar3det']  = pickle.load(open('fbs_day1_NCAR3kmdet_WSPD10MAX_obswind_00z.pk', 'r'))
  fbs['ncar1det']  = pickle.load(open('fbs_day1_NCAR1kmdet_WSPD10MAX_obswind_00z.pk', 'r'))
  fbsworst['ncar3det']   = pickle.load(open('fbsworst_day1_NCAR3kmdet_WSPD10MAX_obswind_00z.pk', 'r'))
  fbsworst['ncar1det']   = pickle.load(open('fbsworst_day1_NCAR1kmdet_WSPD10MAX_obswind_00z.pk', 'r'))

  ### MEAN FSS DIFFERENCE BOOTSTRAP ###
  fbs['gfs']             = pickle.load(open('fbs_day1_GFS_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbs['gfs12']           = pickle.load(open('fbs_day1_GFS_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbs['gfs12-day2']      = pickle.load(open('fbs_day2_GFS_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbs['ncardet']         = pickle.load(open('fbs_day1_NCAR3kmdet_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbs['ncar']            = pickle.load(open('fbs_day1_NCAR2013_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbs['ncar12']          = pickle.load(open('fbs_day1_NCAR2013_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbs['ncar12-day2']     = pickle.load(open('fbs_day2_NCAR2013_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbsworst['gfs']        = pickle.load(open('fbsworst_day1_GFS_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbsworst['gfs12']      = pickle.load(open('fbsworst_day1_GFS_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbsworst['gfs12-day2'] = pickle.load(open('fbsworst_day2_GFS_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbsworst['ncardet']    = pickle.load(open('fbsworst_day1_NCAR3kmdet_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbsworst['ncar']       = pickle.load(open('fbsworst_day1_NCAR2013_UP_HELI_MAX_obsall_00z.pk', 'r'))
  fbsworst['ncar12']     = pickle.load(open('fbsworst_day1_NCAR2013_UP_HELI_MAX_obsall_12z.pk', 'r'))
  fbsworst['ncar12-day2']= pickle.load(open('fbsworst_day2_NCAR2013_UP_HELI_MAX_obsall_12z.pk', 'r'))

  #print 'ci_low, ci_high, bootstrap_mean, bootstrap_median'
  # [daily sums, sigma, thresh, window/hr]
  mod2 = ('ncar1det', 6)
  mod1 = ('ncar3det', 3)
  bs_diff = []
  for i in range(10): bs_diff.append(bootstrap_fss((fbs[mod1[0]][:,i,mod1[1],0,0],fbsworst[mod1[0]][:,i,mod1[1],0,0]), (fbs[mod2[0]][:,i,mod2[1],0,0],fbsworst[mod2[0]][:,i,mod2[1],0,0])))
  print(np.array(bs_diff)[:,0:3])
