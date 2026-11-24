import pandas as pd
import glob
import sys
import ml_functions
m = 0
ys=["2015", "2016", "2017"]
test_year = sys.argv[1]
    
train_years = ys.copy()
train_years.remove(test_year)

files = []
for y in train_years:
    files.extend(glob.glob(f"*_mem{m}_*_{y}*.par"))
files = sorted(files)
df = pd.concat(pd.read_parquet(f) for f in files)
df.info()

features = df.select_dtypes('number').columns
print("normalize", len(features),"numeric features")
_, scales = ml_functions.normalize_multivariate_data(df, features, nonormalize=True)
train_years_str = "".join(train_years)
scales.to_pickle(f'scaling_values_{train_years_str}_NCARENS.pk')
