import urllib.request
import time, os, math, shutil, multiprocessing
import datetime as dt
import sys

def log(msg):
    print( time.ctime(time.time()), msg )

def download_date(thisdate):
    log('downloading %s'%thisdate)
    yyyymmdd = thisdate.strftime('%Y%m%d')
    hh = thisdate.strftime('%H')
    out_path = "/glade/scratch/ahijevyc/HRRR/%s%s/"%(yyyymmdd,hh)

    try: os.mkdir(out_path)
    except OSError as error: print(error)

    for fhr in range(0,nfhr):
        url = "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.%s/conus/hrrr.t%sz.wrfsfcf%02d.grib2"%(yyyymmdd,hh,fhr)
        out_file_name = url.split('/')[-1]
        
        with urllib.request.urlopen(url) as response, open(out_path+out_file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

# download this range of forecast initializations
sdate = dt.datetime(2020,12,17,6,0,0)
edate = dt.datetime(2020,12,17,7,0,0)
timeinc = dt.timedelta(hours=1)
tdate = sdate
date_list = []
while tdate <= edate:
    date_list.append(tdate)
    tdate += timeinc
print('downloading from %s to %s: %d dates'%(sdate, edate, len(date_list)))
   
nfhr      = 49
nfhr      = 18*1

for d in date_list:
    download_date(d)

sys.exit(0)
# use multiprocess to run different initializations in parallel
nprocs    = 6
chunksize = int(math.ceil(len(date_list) / float(nprocs)))
pool      = multiprocessing.Pool(processes=nprocs)
data      = pool.map(download_date, date_list, chunksize)
pool.close()

