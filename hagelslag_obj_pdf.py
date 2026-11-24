import argparse
import cartopy
import datetime
import glob
import G211
from itertools import combinations
import matplotlib.pyplot as plt
from metpy.units import units
import numpy as np
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
import pickle
import pdb
import re
from hwtmode.data import uvmagnitude
import seaborn as sns
from ahijevyc import spc
import sqlite3
import sys

# group 7 labels into 3 broad categories
def combine_storm_mode_labels(cldf):
    #return cldf
    cldf.loc[cldf['label']=="D2", "label"] = "D1" 
    cldf.loc[cldf['label']=="S3", "label"] = "S1" 
    cldf.loc[cldf['label']=="S2", "label"] = "Q1" 
    cldf.loc[cldf['label']=="Q2", "label"] = "Q1" 
    return cldf

dist_thresh = 80.
dpi = 170

# Originally copied and pasted from random_forest_preprocess_gridded import get_closest_report_distances on Jul 16, 2021
def get_closest_report_distances(object_lats, object_lons, object_times, report_types=['hailone', 'wind', 'torn', 'sighail', 'sigwind']):
    # read storm reports from database
    # get observations before forecast hour 0 and after forecast hour 36 for longer verification windows
    gmt2cst = datetime.timedelta(hours=6)
    sdate, edate = object_times.min()-datetime.timedelta(hours=4) - gmt2cst, object_times.max()+datetime.timedelta(hours=4) - gmt2cst
  
    conn = sqlite3.connect('/glade/u/home/sobash/2013RT/REPORTS/reports_v20221101.db')
    c = conn.cursor()

    df = pd.DataFrame()
    for type in report_types:
        if (type=='nonsigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag < 65 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='nonsighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size < 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sigwind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' AND mag >= 65 AND mag <= 999 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='sighail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 2.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='wind'):c.execute("SELECT slat, slon, datetime FROM reports_wind WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hail'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='hailone'):c.execute("SELECT slat, slon, datetime FROM reports_hail WHERE datetime > '%s' AND datetime <= '%s' AND size >= 1.00 ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' ORDER BY datetime asc" % (sdate,edate))
        elif (type=='torn-one-track'):c.execute("SELECT slat, slon, datetime FROM reports_torn WHERE datetime > '%s' AND datetime <= '%s' AND sg == 1 ORDER BY datetime asc" % (sdate,edate))
        rpts = c.fetchall()
        print(len(rpts), type, 'reports')

        if len(rpts) > 0:
            report_lats, report_lons, report_times = list(zip(*rpts))
            report_times = [ pd.to_datetime(t) + datetime.timedelta(hours=6) for t in report_times ] #convert to UTC
            report_times = np.array(report_times)

        for time_tolerance in [1]:
            print(f"loop thru storms & find reports within {time_tolerance}h and distance tolerances")
            # Initialize numpy array with -1s.
            all_distances = np.full(len(object_lats), -1, dtype=np.int16) # With int(), these end up as int64. Should we use less memory with int32 or int16 (-32,768 to 32,767)?
            for i in range(len(object_lats)):
                if i % 10000 == 9999:
                    print( '%d/%d'%(i+1,len(object_lats)))
                if len(rpts) > 0:
                    # find all reports w/in time_tolerance hour(s) of this grid point
                    # should second <= be a < so not overlapping? probably doesnt change much...
                    report_mask = ( report_times >= object_times[i]-datetime.timedelta(hours=time_tolerance) ) & ( report_times <= object_times[i]+datetime.timedelta(hours=time_tolerance) ) 
                    rpt_window_lons, rpt_window_lats = np.array(report_lons)[report_mask], np.array(report_lats)[report_mask]

                    # see if any remain after filtering, if so compute distances from storm centroid
                    if report_mask.sum() > 0:
                        t, bearing = spc.gdist_bearing( object_lons[i]*units.deg , object_lats[i]*units.deg, rpt_window_lons*units.deg, rpt_window_lats*units.deg )
                        all_distances[i] = min(t.to("km").m) # .m is magnitude, not meters. 

            df['%s_rptdist_%dhr'%(type,time_tolerance)] = all_distances
    return df

# =============Arguments===================
parser = argparse.ArgumentParser(description = "pairplot, boxplot, PDF of object attribute", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--attributes", nargs='*', type=str, default=["area","SBCAPE-potential_mean","UP_HELI_MAX_mean"], help="attributes of object (column of csv file)")
parser.add_argument("--confmin", type=int, default=1, help="minimum confidence")
parser.add_argument("--clobber", action="store_true", help="clobber old csv file")
parser.add_argument("-d", "--debug", action="store_true", help="debug info")
parser.add_argument("--freezetime", default="2021-07-08 18:00", help="ignore label created at this time and later")
parser.add_argument("--labelpick", type=str, choices=["first","last","all"], default="all", help="how to handle multiple labels on same storm")
parser.add_argument("--groupby", type=str, default="segmentation", help="group by this")
parser.add_argument("--idir", type=str, choices=["/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv",
        "/glade/p/cisl/aiml/hwt_mode/HWT_2021/WRF_Objects/track_data_wrf_3km_csv_refl", 
        "/glade/p/cisl/aiml/hwt_mode/HWT_2021/HRRR_Objects/track_data_hrrr_3km_csv_refl"], 
        default="/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv", help="path to track_step_*.csv files") 
parser.add_argument("--odir", type=str, default="/glade/scratch/ahijevyc/temp/HWT_mode_output", help="save pngs here")

# Assign arguments to simple-named variables
args = parser.parse_args()
attributes    = args.attributes
clobber       = args.clobber
confmin       = args.confmin
debug         = args.debug
freezetime    = pd.to_datetime(args.freezetime)
groupby       = args.groupby
idir          = args.idir
labelpick     = args.labelpick
odir          = args.odir
segmentation  = "hyst"

if debug: print(args)
atts_and_expertlabels_file = f"/glade/scratch/ahijevyc/temp/HWT_mode_output/atts_and_expertlabels_hyst.csv"
atts_and_cnnlabels_file = f"/glade/scratch/ahijevyc/temp/HWT_mode_output/atts_and_cnnlabels_hyst.csv"
if not (os.path.exists(atts_and_expertlabels_file) and os.path.exists(atts_and_cnnlabels_file)) or clobber:

    assert labelpick == "all", f"Rewrite csv file with labelpick={labelpick}. Are you sure?"

    assert confmin == 1, f"Rewrite csv file with confmin={confmin}. Are you sure?"

    searchstr = idir+"/track_step_*_*_2*.csv"
    cfiles = glob.glob(searchstr)
    nfiles = len(cfiles)
    assert nfiles, f"no files match {searchstr}"

    print("reading",nfiles,"csv files matching "+searchstr) 
    df = [pd.read_csv(cfile,parse_dates=["Valid_Date","Run_Date"]).set_index("Step_ID") for cfile in cfiles]
    dfs = pd.concat(df)
    dfs['segmentation'] = segmentation

    # Derive vector magnitude, don't drop scalar components
    dfs = uvmagnitude(dfs,drop=False)

    if dfs.columns.isin(attributes).sum() < len(attributes):
        print("one or more requested attributes not found in DataFrame")
        print(attributes)
        print("some choices", dfs.columns[::3])
        pdb.set_trace()
        sys.exit(1)

    mode_class_dashboard_tables = pd.read_html('https://www2.mmm.ucar.edu/projects/ncar_ensemble/classifymode/showtable.php', 
           index_col="imgstring", header=0, parse_dates=["labeltime"])
    cldf = mode_class_dashboard_tables[0] # first, and only, table on webpage
    cldf = cldf.sort_values("labeltime")
    cldf["seconds to classify"] = cldf["labeltime"].diff().astype('timedelta64[s]')
    #cldf = cldf[cldf["seconds to classify"] < 150]
    for index, data in cldf.iterrows():
        if index in dfs.index:
            pass
            #if debug: print(index,"good")
        else:
            print(index,"missing")
            pdb.set_trace()

    Forecast_Hours = cldf.index.str[27:29].astype(int).values + cldf.index.str[37:39].astype(int).values
    assert Forecast_Hours.min() == 16 # At one time I thought expert labels started at fhr=14, but they start at fhr=16.

    # Labeltime filter
    nlate = (cldf.labeltime >= freezetime).sum()
    print(f"dropping {nlate}/{len(df)} labeltimes {freezetime} or later")
    cldf = cldf[cldf.labeltime < freezetime]

    # Multiple labels filter
    if labelpick == "first":
        cldf = cldf.groupby("Step_ID").first()
    elif labelpick == "last":
        cldf = cldf.groupby("Step_ID").last()
    elif labelpick == "all":
        pass
    # Drop low confidence.
    cldf = cldf[cldf.conf >= confmin]

    cldf = cldf.join(dfs)
    if os.path.exists(atts_and_expertlabels_file):
        os.rename(atts_and_expertlabels_file,atts_and_expertlabels_file+"_1")
    cldf.to_csv(atts_and_expertlabels_file, index_label=dfs.index.name) # index names differ ("imgstring" and "Step_ID")
    print("Created", os.path.realpath(atts_and_expertlabels_file))

    # Ryan Sobash's convolutional neural network probabilities
    # Jul 1, 2021 email: pandas dataframe containing predictions from one of the latest versions of the CNN for all storms between 6/16/2013 and 12/31/2013. 
    rsdf = pickle.load(open("/glade/work/sobash/HWT_mode/predictions_df_v4.pkl", "rb"))
    #rsdf["Forecast_Hour"] = ((rsdf.time - rsdf.run_date).dt.total_seconds()/3600).astype(int) # not needed because Forecast_Hour is merged from dfs.If you add it here they will overlap and be renamed by _x and _y.
    # Considered renaming fancier strings with list of components, but these simple classes are referred to again below.
    rsdf["label"] = "Q1"
    rsdf.loc[rsdf.supercell_prob > rsdf.qlcs_prob, "label"] = "S1"
    rsdf.loc[(rsdf.disorganized_prob > rsdf.supercell_prob ) & ( rsdf.disorganized_prob > rsdf.qlcs_prob), "label"] = "D1"
    rsdf["conf"] = 5
    dfs["shared_id"] = dfs.index.str[13:26] + dfs.Valid_Date.dt.strftime("%Y%m%d%H") + dfs.Track_ID.str[-3:]
    rsdf["shared_id"] = rsdf.run_date.dt.strftime("%Y%m%d-%H%M") + rsdf["time"].dt.strftime("%Y%m%d%H")+rsdf.track_id.astype(str).str.zfill(3)
    rsdf = rsdf.merge(dfs, on="shared_id")
    rsdf.to_csv(atts_and_cnnlabels_file)
    print("Created", os.path.realpath(atts_and_cnnlabels_file))

else:
    print("Reading cldf", os.path.realpath(atts_and_expertlabels_file))
    cldf = pd.read_csv(atts_and_expertlabels_file, parse_dates=["labeltime", "Run_Date", "Valid_Date"])
    print("Reading rsdf", os.path.realpath(atts_and_cnnlabels_file))
    rsdf = pd.read_csv(atts_and_cnnlabels_file, parse_dates=["Run_Date", "Valid_Date"])



add_rpts = True
if add_rpts:
    quickfile = f"{odir}/features_with_rpts.csv"
    if os.path.exists(quickfile):
        cldf = pd.read_csv(quickfile, parse_dates=["labeltime"])
        print(f"read quickfile {quickfile}")
    else:
        rpts = get_closest_report_distances(cldf["Centroid_Lat"], cldf["Centroid_Lon"], cldf["Valid_Date"])
        cldf = cldf.join(rpts)
        cldf.to_csv(quickfile)
        print(f"created quickfile {quickfile}")
        #TODO: why does it give TypeError: '<' not supported between instances of 'str' and 'int'
        # when using cldf, but when csv quickfile is read, there is no problem?

    # If you comment this out here, comment it out in other place.
    cldf = combine_storm_mode_labels(cldf) # combine labels into broad categories
    for att in [x+"_rptdist_1hr" for x in ["torn", "hailone", "sighail", "wind","sigwind"]]:
        print(att)
        tmp = cldf.replace(-1,np.nan)[["label",att]]
        counts = tmp.groupby(["label"]).size()
        print(tmp.groupby("label").describe())
        ax = tmp.groupby("label")[att].apply(lambda c: (c<=dist_thresh).sum()/len(c)).plot.bar()
        for i,n in enumerate(counts):
            ax.annotate(xy=(i, 0), text=n, ha='center', va='bottom', xycoords=("data", ax.transAxes))
        ax.grid()
        ax.set_title(f"chance of {att} within {dist_thresh} km")
        ofile = f"{odir}/{att}.ge{confmin}confidence.label.hitrate{dist_thresh}km.png"
        ax.figure.savefig(ofile)
        print("Made", os.path.realpath(ofile))
        plt.close()

        g = sns.boxplot(x="label", y=att, data=tmp)
        g.axes.set_ylim((0,2500))
        for label,n in counts.items():
            i = [x.get_text() for x in g.axes.get_xticklabels()].index(label)
            g.annotate(xy=(i, 0), text=n, ha='center', va='bottom', xycoords=("data", g.transAxes))
        g.grid()
        ofile = f"{odir}/{att}.ge{confmin}confidence.label.boxplot.png"
        g.figure.savefig(ofile)
        print("Made", os.path.realpath(ofile))
        plt.close()

if debug:
    pdb.set_trace()

# Drop low confidence.
nq = (cldf.conf < confmin).sum()
print(f"dropping {nq}/{len(cldf)} confidence < {confmin}")
cldf = cldf[cldf.conf >= confmin]

# Labeltime filter
nlate = (cldf.labeltime >= freezetime).sum()
print(f"dropping {nlate}/{len(cldf)} labeltimes {freezetime} or later")
cldf = cldf[cldf.labeltime < freezetime]

cldf = combine_storm_mode_labels(cldf) # Don't comment out; change function--it's used other place(s)

clip_shear = (0,60)
clip_cape  = (0,5500)
clip_srh   = (0, 1250)
clip_area = (0, 3200)
lim = {
        "area" : clip_area,
        "eccentricity" : (0,1),
        "MLCAPE-potential_max"  : clip_cape,
        "MLCAPE-potential_mean" : clip_cape,
        "MUCAPE-potential_max"  : clip_cape,
        "MUCAPE-potential_mean" : clip_cape,
        "RVORT1_MAX_max"        : (0,0.02),
        "RVORT1_MAX_mean"       : (0,0.02),
        "RVORT5_MAX_max"        : (0,0.02),
        "RVORT5_MAX_mean"       : (0,0.02),
        "SBCAPE-potential_max"  : clip_cape,
        "SBCAPE-potential_mean" : clip_cape,
        "SHR1-potential_max"    : (0,60),
        "SHR1-potential_mean"   : (0,35),
        "SHR6-potential_max"    : (0,75),
        "SHR6-potential_mean"   : clip_shear,
        "SRH01-potential_max" : clip_srh,
        "SRH01-potential_mean" : clip_srh,
        "SRH03-potential_max" : clip_srh,
        "SRH03-potential_mean" : clip_srh,
        "UP_HELI_MAX03_max"  : (0, 500),
        "UP_HELI_MAX03_mean" : (0, 100),
        "UP_HELI_MAX_max"  : (0, 500),
        "UP_HELI_MAX_mean" : (0, 100),
        "UP_HELI_MIN_min"  : (-300, 0),
        "W_DN_MAX_min"  : (-22,0),
        "W_DN_MAX_max"  : (-10,0.5),
        "W_UP_MAX_max"  : (0,50),
        "W_UP_MAX_mean" : (0,25),
        }

for a in attributes:
    if a not in lim:
        print(f"please add lim entry for {a}")
        sys.exit(1)

def add_fineprint(text=None, **kwargs):
    annotate_dict = {"xy":(3,2), "xycoords":'figure pixels', "fontsize":5}
    annotate_dict.update(kwargs)
    if not text:
        text = f"\ncreated {datetime.datetime.now(tz=None)}"
    fineprint_obj = plt.annotate(text=text, **annotate_dict)
    return fineprint_obj

fineprint = "\ncreated "+str(datetime.datetime.now(tz=None)).split('.')[0]
fineprint += f"\n{cldf.Valid_Date.min()}-{cldf.Valid_Date.max()}  segmentation {segmentation}"
fineprint += f"\nlabel time < {freezetime}  confidence >= {confmin}"



cat_type = CategoricalDtype(categories=["Q1", "Q2", "S1", "S2", "S3", "D1", "D2"], ordered=True)
for df, prefix, conf, lon, lat in [(rsdf, "cnn.", 0, "centroid_lon", "centroid_lat"), (cldf, "", confmin, "Centroid_Lon", "Centroid_Lat")]:
    df["label"] = df["label"].astype(cat_type)
    df['label'] = df['label'].cat.remove_unused_categories()

    # filter forecast hours print msg about how many are dropped 
    if (df["Forecast_Hour"] < 16).sum():
        print(f"dropping {(df['Forecast_Hour'] < 16).sum()}/{len(df)} fhr < 16")
        df = df[df["Forecast_Hour"] >= 16]
    if (df["Forecast_Hour"] > 34).sum():
        print(f"dropping {(df['Forecast_Hour'] > 34).sum()}/{len(df)} fhr > 34")
        df = df[df["Forecast_Hour"] <= 34]

    # filter UH print how many are dropped
    lowuh = (df.UP_HELI_MIN_min > -25) & (df.UP_HELI_MAX_max < 25)
    if lowuh.sum():
        print(f"dropping {lowuh.sum()}/{len(df)} low uh")
        df = df[(df.UP_HELI_MIN_min <= -25) | (df.UP_HELI_MAX_max >= 25)]


    g = df["label"].value_counts().plot(kind='bar', grid=True)
    g.bar_label(g.containers[0])
    g.set_title(f"Run_Date range:{df.Run_Date.min()}-{df.Run_Date.max()}")
    fp = add_fineprint(fineprint)
    ofile = f"{odir}/{prefix}label_counts.png"
    g.figure.savefig(ofile)
    print(f"made {os.path.realpath(ofile)}")
    plt.close()

    ax = plt.axes(projection=G211.g211)
    sns.kdeplot(
            ax = ax,
            common_norm=False,
            cut=0,
            data=df,
            x=lon,
            y=lat,
            hue="label",
            fill=False,
            thresh=0.25,
            levels=4,
            linestyles=["dotted","solid","solid"],
            linewidths=3,
            transform=cartopy.crs.PlateCarree()
            )
    ax.add_feature(cartopy.feature.COASTLINE.with_scale('50m'), linewidth=0.25)
    ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'), linewidth=0.25)
    ax.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.05)
    ax.add_feature(cartopy.feature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25, facecolor='k', alpha=0.05)
    ax.set_title(f"Run_Date range:{df.Run_Date.min()}-{df.Run_Date.max()}")
    fp = add_fineprint(fineprint)
    ofile = f"{odir}/{prefix}kdeplot.png"
    ax.figure.savefig(ofile,dpi=dpi)
    print(f"made {os.path.realpath(ofile)}")
    plt.close()


    g = sns.pairplot(df[attributes+["label"]], hue="label")
    fp = add_fineprint(fineprint)
    ofile = odir + "/" + prefix + "_".join(attributes) + f".ge{confmin}confidence.pairplot.png"
    g.savefig(ofile)
    print("Made", os.path.realpath(ofile))
    plt.close()


    counts = df.groupby(["label"]).size()

    for att in attributes:
        g = sns.boxplot(x="label", y=att, data=df[attributes+["label"]])
        g.axes.set_ylim(lim[att])
        for i,n in enumerate(counts):
            g.annotate(xy=(i, 0), text=n, ha='center', va='bottom', xycoords=("data", g.transAxes))
        fp = add_fineprint(fineprint)
        ofile = odir + "/" + prefix + att + f".ge{confmin}confidence.label.boxplot.png"
        g.figure.savefig(ofile)
        print("Made", os.path.realpath(ofile))
        plt.close()


    if len(attributes) >= 2:
        for att1, att2 in combinations(attributes,2):
            g = sns.displot(df, x=att1, y=att2, col="label", kind="kde", col_wrap=3, clip=(lim[att1],lim[att2]), common_norm=False, fill=True)
            for col_key, ax in g.axes_dict.items():
                ax.set_title(f"{col_key}  (n={counts[col_key]})") 
            ofile = f"{odir}/{prefix}{att1}_{att2}.ge{confmin}confidence.label.displot.png"
            fp = add_fineprint(fineprint)
            g.savefig(ofile, dpi=96)
            print("Made", os.path.realpath(ofile))
            plt.close()

if debug:
    pdb.set_trace()

for k,v in cldf.groupby(groupby):
    # How many objects have a forecast hour between 16 and 34 hours?
    df = v[(v.Forecast_Hour >= 16) & (v.Forecast_Hour <= 34)]
    df = df[(df.area >= 150) | (df.UP_HELI_MAX_max >= 150) | (df.UP_HELI_MIN_min <= -100)]
    #df.to_csv("worthwhile_area_UH."+k+".csv")
    print('nice area and UH', k, len(df))

sys.exit(0)

# Combine attribute columns into one column
cldf_long = pd.melt(cldf, groupby, var_name="attribute")
# plot 3 different colors shaded in a single panel. one for each groupby approach. 
g = sns.FacetGrid(cldf_long, hue=groupby, col="attribute", sharex=False, sharey=True, col_wrap=3)
g.map(plt.hist, "value", alpha=.4, log=True)
g = g.set(ylim=(1,5e5))
g.add_legend()
ofile = odir + "/" + groupby+".png"
g.savefig(ofile)
print(os.path.realpath(ofile))

