import logging
import numpy as np
import pandas as pd
import pdb
import os
import sys
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

ifile = f"/glade/campaign/mmm/parc/ahijevyc/ENTLN/flash.txt"
ifile = sys.argv[1]
base, ext = os.path.splitext(ifile)
ofile = base + ".par"
dtypes = dict(time=np.int32, millisecond=np.int16, lat=np.float32, lon=np.float32, amp=np.float32, incloud=np.int8)
# Ignore last column (f or p). They should all be f now.
# delim_whitespace=True because columns go from being separated by a single space to being separated by multiple spaces mid-2022.
logging.info(f"read {ifile}")
df = pd.read_csv(ifile, delim_whitespace=True, names=["time","millisecond","lat","lon","amp","incloud"],
        usecols=[0,1,2,3,4,5], memory_map=True, low_memory=False ) #map file object onto memory and access directly. can improve performance without I/O overhead

# ensure all times fall in the time range [start,end) where start is from the first row; end is from the last.
start = df.iloc[0]["time"]
end   = df.iloc[-1]["time"]
logging.info("assert start and end times are between 2018 and now")
assert pd.to_datetime("20180101").timestamp() <= start < pd.Timestamp.utcnow().timestamp()
assert pd.to_datetime("20180101").timestamp() <= end < pd.Timestamp.utcnow().timestamp()

past = df["time"] <= start
future = df["time"] > end
badloc = (df["lat"] < -90) | (df["lat"] > 90) | (df["lon"] < -180) | (df["lon"] > 180)
outofbounds = (df["lat"] < 11) | (df["lat"] > 62) | (df["lon"] < -150) | (df["lon"] > -50)
badms = (df["millisecond"] < 0) | (df["millisecond"] > 999)
badincloud = (df["incloud"] != 0) &  (df["incloud"] != 1)
corrupt =  past | future | badloc | badms | badincloud

logging.warning(f"{past.sum()} ({past.sum()/len(df):%}) flashes before {start}")
logging.warning(f"{future.sum()} ({future.sum()/len(df):%}) flashes after {end}")
logging.warning(f"{badloc.sum()} ({badloc.sum()/len(df):%}) bad locations")
logging.warning(f"{badms.sum()} ({badms.sum()/len(df):%}) bad milliseconds")
logging.warning(f"{badincloud.sum()} ({badincloud.sum()/len(df):%}) bad incloud values")
logging.warning(f"{corrupt.sum()} ({corrupt.sum()/len(df):%}) any corrupt metadata")
logging.warning(f"{outofbounds.sum()} ({outofbounds.sum()/len(df):%}) flashes out-of-bounds")


df = df[~past & ~future & ~outofbounds & ~badms & ~badincloud]

df["time"] = pd.to_datetime(df["time"], unit="s") + pd.to_timedelta(df["millisecond"], unit="millisecond")
df = df.drop(columns="millisecond")
df.to_parquet(ofile)
