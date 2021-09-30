#!/usr/bin/env python

import datetime
import pickle
import numpy as np
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.ndimage.filters import gaussian_filter

def bss(obs, preds):
    bs = np.mean((preds - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - (bs/climo)

def plot_2d_hist(predx, predy):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

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

def print_scores(fcst, obs, rptclass, probthresh=0.5):
    # print scores for this set of forecasts
    # histogram of probability values
    bins = np.arange(0,1.1,0.1)
    hist, bins = np.histogram(fcst, bins=bins)

    cm = metrics.confusion_matrix(obs, fcst>probthresh)
    hits = cm[1,1]
    false_alarms = cm[0,1]
    misses = cm[1,0]
    correct_neg = cm[0,0]
    hits_random = (hits + misses)*(hits + false_alarms) / float(hits + misses + false_alarms + correct_neg)

    ets = (hits-hits_random)/float(hits + false_alarms + misses - hits_random)
    bias = (hits+false_alarms)/float(hits+misses)
    pod = hits/float(hits+misses)
    far = false_alarms/float(hits+false_alarms)
    pofd = false_alarms/float(correct_neg + false_alarms)

    # reliability curves
    true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=10)
    bss_val = bss(obs, fcst)
    auc = metrics.roc_auc_score(obs, fcst)
    
    #for i in range(true_prob.size): print(true_prob[i], fcst_prob[i])

    return (hist, fcst.size, bss_val, auc, bias, pod, far, ets, true_prob, fcst_prob)

def make_gridded_forecast(predictions, labels, dates, fhr):
    ### reconstruct into grid by day (mask makes things more complex than a simple reshape)
    unique_forecasts, unique_fhr = np.unique(dates), np.unique(fhr)
    num_dates, num_fhr = len(unique_forecasts), len(unique_fhr)

    gridded_predictions = np.zeros((num_dates,num_fhr,65*93), dtype='f')
    gridded_labels      = np.zeros((num_dates,num_fhr,65*93), dtype='f')

    thismask = mask.flatten()

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

def smooth_gridded_forecast(predictions_gridded):
    smoothed_predictions = []
    dim = predictions_gridded.shape
    for k,s in enumerate(smooth_sigma):
        if len(dim) == 4: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,0,s,s]))
        if len(dim) == 3: smoothed_predictions.append(gaussian_filter(predictions_gridded, sigma=[0,s,s]))

    # return only predictions for US points
    return np.array(smoothed_predictions)

def verify_env():
  #### VERIFY PREDICTIONS BASED ON CAPE/SHEAR
  uh_thresh = 60
  cape_thresh = [0,100,500,1000,2000,3000,4000,5000]
  shear_thresh = [0,10,20,30,40,50]
  for c in range(len(cape_thresh)-1):
    for s in range(len(shear_thresh)-1):
        envmask = (cape_all >= cape_thresh[c]) & (cape_all < cape_thresh[c+1]) & (shear_all >= shear_thresh[s]) & (shear_all < shear_thresh[s+1]) & (fhr_all > 12)

        # extract labels and UH points using this mask
        predictions, labels, uh120 = predictions_all[envmask,:], labels_all[envmask,:], uh120_all[envmask]

        # verify deterministic forecasts over masked points to select appropriate UH threshold for smoothing
        bias2 = 0
        while bias2 < 0.95 or bias2 > 1.05:
            hist2, numpts2, bss_val2, auc2, bias2, pod2, far2, ets2, tp2, fp2 = print_scores((uh120>uh_thresh).astype(int), labels[:,i], classes[i])
            if bias2 > 1.05: uh_thresh += 2
            if bias2 < 0.95: uh_thresh -= 2

        # make gridded UH forecast and labels
        predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>uh_thresh).astype(np.int32), labels_all[:,i], dates_all, fhr_all)
        # make smoothed gridded UH forecast
        predictions_gridded_uh_smoothed = smooth_gridded_forecast(predictions_gridded_uh)
        # extract only CONUS points and only other masked points
        uh120_smoothed = predictions_gridded_uh_smoothed[:,:,:,mask].reshape((len(smooth_sigma),-1))[:,envmask]
        
        hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(uh120_smoothed[8,:], labels[:,i], classes[i])

        #predictions, labels, uh120 = predictions_all[envmask,:], labels_all[envmask,:], uh120_all[envmask]
        #hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions[:,i], labels[:,i], classes[i])

        print(cape_thresh[c], shear_thresh[s], numpts, "%.2f"%bss_val, "%.2f"%auc, "%.2f"%bias, "%.2f"%pod, "%.2f"%far, "%.2f"%ets)

classes = { 0:'all', 1:'wind', 2:'hailone', 3:'torn', 4:'sighail', 5:'sigwind'}    
i = 0
print(classes[i])
numclasses = 6
mask  = pickle.load(open('/glade/u/home/sobash/2013RT/usamask.pk', 'rb'))
smooth_sigma = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]

# read in gefs forecasts
#predictions_mem, uh120_mem = [], []
#for k in range(1,10):
#    print(k)
#    predictions, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120, dates_all =  pickle.load(open('predictions_nn_120km_2hr_gefs_mem%d_all'%k, 'rb'))
#    predictions_mem.append(predictions)
#    uh120_mem.append(uh120)
#predictions_all_nn = np.array(predictions_mem).mean(axis=0)
#uh120_all = np.array(uh120_mem).mean(axis=0)
    
predictions_all_nn, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open('predictions_nn_120km_2hr_all', 'rb'))
#predictions_all_rf, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, dates_all =  pickle.load(open('predictions_rf_120km_2hr_n100_all', 'rb'))


#average_prediction = (predictions_all_rf + predictions_all_nn)/2.0
#predictions_all = average_prediction
predictions_all = predictions_all_nn
print('Verifying %d forecast points'%predictions_all.shape[0])

# make gridded forecasts for machine learning and UH forecasts
predictions_gridded, labels_gridded    = make_gridded_forecast(predictions_all[:,i], labels_all[:,i], dates_all, fhr_all)
predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>60).astype(np.int32), labels_all[:,i], dates_all, fhr_all)

# make smoothed UH predictions
predictions_gridded_uh_smoothed = smooth_gridded_forecast(predictions_gridded_uh)

### MAKE AND VERIFY PREDICTIONS FOR 24-HOUR AGGREGATE FORECASTS
# verify 24-hr max ML forecast
predictions_gridded_ml_max = np.amax(predictions_gridded, axis=1)
predictions_gridded_uh_max = np.amax(predictions_gridded_uh, axis=1)

predictions_gridded_uh_max_smoothed = smooth_gridded_forecast(predictions_gridded_uh_max) #smoothed 24-hour UH max forecast
predictions_gridded_ml_max_smoothed = smooth_gridded_forecast(predictions_gridded_ml_max) #smoothed 24-hour ML forecasts
labels_gridded_max = np.amax(labels_gridded, axis=1)

#### mask and flatten (only verify over CONUS points)
mask = mask.reshape((65,93))
predictions_gridded = predictions_gridded[:,:,mask]
predictions_gridded_uh = predictions_gridded_uh[:,:,mask]
predictions_gridded_uh_smoothed = predictions_gridded_uh_smoothed[:,:,:,mask]
predictions_gridded_ml_max = predictions_gridded_ml_max[:,mask]
predictions_gridded_uh_max = predictions_gridded_uh_max[:,mask]
predictions_gridded_uh_max_smoothed = predictions_gridded_uh_max_smoothed[:,:,mask]
predictions_gridded_ml_max_smoothed = predictions_gridded_ml_max_smoothed[:,:,mask]
labels_gridded_max = labels_gridded_max[:,mask]

print(predictions_gridded_ml_max.shape, labels_gridded.shape, predictions_gridded_uh_max.shape)

#plot_2d_hist(predictions_all[:,i], predictions_gridded_uh_smoothed[8,:].flatten())
#plot_2d_hist(predictions_all[:,i], predictions_all_rf[:,i])


#verify_env()

'''
### VERIFY PREDICTIONS FOR 24-HOUR FORECASTS
print('24-hour max ML forecast verification')
hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions_gridded_ml_max.flatten(), labels_gridded_max.flatten(), classes[i], probthresh=0.3)
print(numpts, "BSS: %.2f"%bss_val, "AUC: %.2f"%auc, "BIAS: %.2f"%bias, "POD: %.2f"%pod, "FAR: %.2f"%far, "ETS: %.2f"%ets)

print('24 hour max UH forecast verification')
for n in range(len(smooth_sigma)):
    hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions_gridded_uh_max_smoothed[n,:].flatten(), labels_gridded_max.flatten(), classes[i], probthresh=0.3)
    print(smooth_sigma[n]*80, numpts, "BSS: %.2f"%bss_val, "AUC: %.2f"%auc, "BIAS: %.2f"%bias, "POD: %.2f"%pod, "FAR: %.2f"%far, "ETS: %.2f"%ets)
'''

### VERIFY PREDICTIONS FOR 4-HOUR FORECASTS
print('Verifying 4-hr ML predictions')
hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions_all[:,i], labels_all[:,i], classes[i])
print(numpts, "%.2f"%bss_val, "%.2f"%auc, "%.2f"%bias, "%.2f"%pod, "%.2f"%far, "%.2f"%ets)
for h in hist: print(h)
for t,f in list(zip(tp, fp)): print(t,f)
#for p in np.arange(0,1.1,0.1):
#    hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions_all[:,i], labels_all[:,i], classes[i], probthresh=p)
#    print(p, numpts, "%.2f"%bss_val, "%.2f"%auc, "%.2f"%bias, "%.2f"%pod, "%.2f"%far, "%.2f"%ets)
#    #print(tp, fp)



'''
# verify smoothed UH predictions
print('Verifying smoothed 4-hr UH predictions')
for n in range(len(smooth_sigma)):
    hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions_gridded_uh_smoothed[n,:].flatten(), labels_all[:,i], classes[i])
    print(smooth_sigma[n]*80, numpts, "%.2f"%bss_val, "%.2f"%auc, "%.2f"%bias, "%.2f"%ets)
    for h in hist: print(h)
    #print(tp, fp)

#verify binary UH predictions
print('Verifying binary 4-hr UH predictions')
for uh in np.arange(10,101,10):
    hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores((uh120_all>uh).astype(np.int), labels_all[:,i], classes[i])
    print(uh, numpts, "%.2f"%bss_val, "%.2f"%auc, "%.2f"%bias, "%.2f"%pod, "%.2f"%far, "%.2f"%ets)
    #print(tp, fp)
'''

# initial values for thresholds
uh_thresh = 60
#uh_thresh = 200
if i == 3: uh_thresh = 175
if i == 4: uh_thresh = 175
    
prob_thresh = 0.3
if i== 4: prob_thresh = 0.1

#months_all = np.array([ int(d[5:7]) for d in dates_all ])

# verify hourly binary predictions
all_uh120_smoothed, all_labels = [], []
for j in range(1,37):
#for j in range(10,13):
    thismask = (fhr_all==j)
    #thismask = (months_all==j)

    predictions, labels, uh120 = predictions_all[thismask,:], labels_all[thismask,:], uh120_all[thismask]

    print(labels.dtype, predictions.dtype) 
    hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions[:,i], labels[:,i], classes[i], probthresh=prob_thresh)

    '''
    # for computing bias=1 for UH and binary ML probs
    bias = 0
    while bias < 0.94 or bias > 1.06:
        hist, numpts, bss_val, auc, bias, pod, far, ets, tp, fp = print_scores(predictions[:,i], labels[:,i], classes[i], probthresh=prob_thresh)
        if bias > 1.05: prob_thresh += 0.01
        if bias < 0.95: prob_thresh -= 0.01


    # compute threshold that produces a bias of 1 for this forecast hour
    bias2 = 0
    while bias2 < 0.94 or bias2 > 1.05:
        hist2, numpts2, bss_val2, auc2, bias2, pod2, far2, ets2, tp2, fp2 = print_scores((uh120>uh_thresh).astype(int), labels[:,i], classes[i])
        if bias2 > 1.05: uh_thresh += 2
        if bias2 < 0.95: uh_thresh -= 2

    # make smoothed predictions for bias=1 UH threshold
    predictions_gridded_uh, labels_gridded = make_gridded_forecast((uh120_all>uh_thresh).astype(np.int32), labels_all[:,i], dates_all, fhr_all)
    predictions_gridded_uh_smoothed        = smooth_gridded_forecast(predictions_gridded_uh)
    predictions_gridded_uh_smoothed        = predictions_gridded_uh_smoothed[:,:,:,mask]
    uh120_smoothed                         = predictions_gridded_uh_smoothed.reshape((len(smooth_sigma),-1))[:,thismask]
    
    # verify smoothed UH predictions
    hist2, numpts2, bss_val2, auc2, bias2, pod2, far2, ets2, tp2, fp2 = print_scores(uh120_smoothed[8,:], labels[:,i], classes[i])
    all_uh120_smoothed.append(uh120_smoothed[8,:])
    all_labels.append(labels[:,i])
    '''

    #print(j, numpts, "BSS-ML: %.3f"%bss_val, "AUC-ML: %.3f"%auc, "BSS-UH: %.3f"%bss_val2, "AUC-UH: %.3f"%auc2, 'uh_thresh: %.2f'%uh_thresh, 'prob_thresh: %.2f'%prob_thresh)
    print(j, numpts, "BSS-ML: %.3f"%bss_val, "AUC-ML: %.3f"%auc)
    #print(j, numpts, "BSS: %.2f"%bss_val, "AUC: %.2f"%auc, "BIAS: %.2f"%bias, "ETS: %.2f"%ets, "BIAS-UH: %.2f"%bias2, "ETS-UH: %.3f"%ets2, uh_thresh)

# verify aggregate scores (with time-varying UH threshold)
hist2, numpts2, bss_val2, auc2, bias2, pod2, far2, ets2, tp2, fp2 = print_scores(np.array(all_uh120_smoothed).flatten(), np.array(all_labels).flatten(), classes[i])
for h in hist2: print(h)
print(tp2, fp2)
