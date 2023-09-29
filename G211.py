import cartopy.crs as ccrs
import numpy as np
import os
import pickle
import xarray

g211 = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))

# TODO: Depreciate width and height in favor of nlon and nlat
nlon = 93
nlat = 65
width = nlon
height = nlat

ll_lon = -133.459
ll_lat=12.19
ur_lon=-49.38641
ur_lat=57.2894

# lower left and upper right projection coordinates
(llx, lly, llz), (urx ,ury, urz) = g211.transform_points(
    ccrs.PlateCarree(),
    np.array([ll_lon, ur_lon]),
    np.array([ll_lat, ur_lat])
    )

# gridded projection coordinates xv, yv
xs = np.linspace( llx, urx, nlon)
ys = np.linspace( lly, ury, nlat)
xv, yv = np.meshgrid(xs,ys)
# gridded lat/lon coordinates lat, lon
ll3 = ccrs.PlateCarree().transform_points(g211,xv,yv)
lon = ll3[:,:,0]
lat = ll3[:,:,1]

mask = pickle.load(open('/glade/u/home/ahijevyc/HRRR/usamask_mod.pk', 'rb'))
mask = xarray.DataArray(mask.reshape((nlat,nlon)), 
        coords=dict(y=range(nlat), x=range(nlon)), dims=["y","x"])

# TODO: clean up this kludge
class x2():
    """
    Half spacing compared to G211
    """
    global ll_lon, ll_lat, ur_lon, ur_lat
    def __init__(self):
        nlon = 93 * 2 - 1 
        nlat = 65 * 2 - 1 # subtract 1 to line up with 80 km grid (started Sep 19 2023)

        xs = np.linspace( llx, urx, nlon)
        ys = np.linspace( lly, ury, nlat)
        self.xs = xs
        self.ys = ys
        xv, yv = np.meshgrid(xs,ys)
        self.xv = xv
        self.yv = yv
        ll3 = ccrs.PlateCarree().transform_points(g211,xv,yv)
        lon = ll3[:,:,0]
        lat = ll3[:,:,1]

        self.nlon = nlon
        self.nlat = nlat
        self.width = nlon
        self.height = nlat
        self.lon = lon
        self.lat = lat
