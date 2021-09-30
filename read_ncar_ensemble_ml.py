import pandas as pd
import pickle

trainyr = "20152016"
trainmem = 1
predictyr = 2017
for predictmem in range(2,11):
    print(f"predict.{predictyr}.mem{predictmem}")
    labels_predictions_file = f"/glade/work/ahijevyc/NSC_objects/predictions_nn_40km_1hr_basic_neighborhood.0latlon_hash_buckets.2015043000-2017122900.train.{trainyr}.mem{trainmem}.predict.{predictyr}.mem{predictmem}.pk"
    print(labels_predictions_file)
    predictions_all, labels_all, fhr_all, cape_all, shear_all, uh_all, uh120_all, uh01_120_all, date_all = pickle.load(open(labels_predictions_file, 'rb'))
    print(predictions_all.shape)
    print(date_all)
