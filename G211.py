import cartopy
import numpy as np
import pickle
import xarray

g211 = cartopy.crs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))
width=93
height=65

ll_lon = -133.459
ll_lat=12.19
ur_lon=-49.38641
ur_lat=57.2894

# lower left and upper right projection coordinates
(llx, lly, llz), (urx ,ury, urz) = g211.transform_points(
    cartopy.crs.PlateCarree(),
    np.array([ll_lon, ur_lon]),
    np.array([ll_lat, ur_lat])
    )

# gridded projection coordinates xv, yv
xs = np.linspace( llx, urx, width)
ys = np.linspace( lly, ury, height)
xv, yv = np.meshgrid(xs,ys)
# gridded lat/lon coordinates lat, lon
ll3 = cartopy.crs.PlateCarree().transform_points(g211,xv,yv)
lon = ll3[:,:,0]
lat = ll3[:,:,1]

mask = pickle.load(open('/glade/u/home/ahijevyc/HRRR/usamask_mod.pk', 'rb'))
mask = xarray.DataArray(mask.reshape((height,width)), 
        coords=dict(y=range(height), x=range(width)), dims=["y","x"])

# TODO: clean up this kludge
def x2():
    """
    Half spacing compared to G211
    """
    global ll_lon, ll_lat, ur_lon, ur_lat
    width=93 * 2
    height=65 * 2

    xs = np.linspace( llx, urx, width)
    ys = np.linspace( lly, ury, height)
    xv, yv = np.meshgrid(xs,ys)
    ll3 = cartopy.crs.PlateCarree().transform_points(g211,xv,yv)
    lon = ll3[:,:,0]
    lat = ll3[:,:,1]
    return lon, lat

