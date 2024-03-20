import cartopy.crs as ccrs
import geopandas
import numpy as np
import os
import pandas as pd
import pdb
import pickle
from shapely.geometry import Polygon
import xarray

g211 = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))

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

# depreciated width and height in favor of nlon and nlat
nlon = 93
nlat = 65


def getgdf(nlon: int, nlat: int, lon, lat):
    """ 
    Given nlon, nlat, lon, and lat
    Return Geopandas DataFrame of grid points
    """
    x, y = np.meshgrid(range(nlon), range(nlat))
    # assign 80-km grid points to geopandas dataframe
    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel()})
    geometry = geopandas.points_from_xy(x=lon.ravel(), y=lat.ravel())
    crs = ccrs.PlateCarree()
    # Avoid pyproj.exceptions.CRSError: Invalid CRS input: <cartopy.crs.PlateCarree object
    crs = "EPSG:4326" # had to switch after casper upgrade Oct 2023
    grid = geopandas.GeoDataFrame(
        df.set_index(["y", "x"]), geometry=geometry, crs=crs
    )
    return grid


# gridded projection coordinates xv, yv
xs = np.linspace( llx, urx, nlon)
ys = np.linspace( lly, ury, nlat)
xv, yv = np.meshgrid(xs,ys)
# gridded lat/lon coordinates lat, lon
ll3 = ccrs.PlateCarree().transform_points(g211,xv,yv)
lon = ll3[:,:,0]
lat = ll3[:,:,1]

def getmask(grid: geopandas.GeoDataFrame, nlon: int, nlat: int):
    """
    Given a grid Geopandas DataFrame, nlon, and nlat,
    Return DataArray of True over CONUS False elsewhere
    """
    gdf = geopandas.read_file("/glade/work/ahijevyc/share/shapeFiles/ne_110m_admin_0_countries")
    usa = gdf[gdf["NAME"] == "United States of America"]

    # get poly from gdf instead of depreciated geopandas.datasets

    #poly = geopandas.GeoDataFrame.from_file(
    #    geopandas.datasets.get_path("naturalearth_lowres")
    #)
    #usa_old = poly[poly.iso_a3 == "USA"]
    
    # lat/lon box around CONUS (no AK or HI)
    lat_point_list = [51, 51, 20, 20, 51]
    lon_point_list = [-130, -60, -60, -130, -130]
    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    polygon = geopandas.GeoDataFrame(crs=usa.crs, geometry=[polygon_geom])
    conus = geopandas.overlay(usa, polygon, how="intersection")

    conus_mask = np.array(
        [g.within(conus.geometry.values[0]) for g in grid.geometry]
    )
    conus_mask = xarray.DataArray(
        conus_mask.reshape(nlat, nlon),
        dims=["y","x"],
        coords={"y": range(nlat), "x": range(nlon)},
    )
    return conus_mask

grid = getgdf(nlon, nlat, lon, lat)

# TODO: use getmask() instead of pickle file. results are different.
mask = pickle.load(open('/glade/work/ahijevyc/NSC_objects/HRRR/usamask_mod.pk', 'rb'))
mask = xarray.DataArray(mask.reshape((nlat,nlon)), 
        coords=dict(y=range(nlat), x=range(nlon)), dims=["y","x"])



class x2:
    """
    Half grid spacing of G211
    """
    global llx, lly, urx, ury
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
        self.lon = lon
        self.lat = lat
        self.grid = getgdf(nlon, nlat, lon, lat)
        self.mask = getmask(self.grid, nlon, nlat)
