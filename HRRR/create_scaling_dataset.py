#!/usr/bin/env python

import numpy as np
import datetime as dt
import sys, os, pickle, time

from ml_functions import read_csv_files, normalize_multivariate_data, log, get_features

dataset = 'HRRR'
scaling_dataset = 'HRRR'
sdate   = dt.datetime(2019,4,1,0,0,0)
edate   = dt.datetime(2019,6,30,0,0,0)

##################################

features = get_features(subset='all')

print(features)
log('Number of features %d'%len(features))

log('Reading Data')
# read data and reassign data types to float32 to save memory
type_dict = {}
for f in features: type_dict[f]='float32'
df, numfcsts = read_csv_files(sdate, edate, dataset)

log('Computing normalization')            
norm_in_data, scaling_values = normalize_multivariate_data(df[features].values.astype(np.float32), features, scaling_values=None)

log('Output scaling values')
pickle.dump(scaling_values, open('scaling_values_all_%s.pk'%(scaling_dataset), 'wb'))

log('Created scaling dataset for %d features based on %d forecasts'%(len(features), numfcsts))
