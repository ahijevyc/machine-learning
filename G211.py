import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import xarray as xr
import warnings
from shapely.geometry import Polygon
from typing import Tuple

G211 = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))

def get_projection_bounds() -> Tuple[float, float, float, float]:
    """Calculates llx, lly, urx, ury for the G211 grid."""
    ll_lon, ll_lat = -133.459, 12.19
    ur_lon, ur_lat = -49.38641, 57.2894
    pts = G211.transform_points(
        ccrs.PlateCarree(),
        np.array([ll_lon, ur_lon]),
        np.array([ll_lat, ur_lat])
    )
    return pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]

# Pre-calculate bounds for legacy support
LLX, LLY, URX, URY = get_projection_bounds()

# depreciated width and height in favor of nlon and nlat
nlon = 93
nlat = 65


def getgdf(nlon: int, nlat: int, lon: np.ndarray, lat: np.ndarray) -> gpd.GeoDataFrame:
    """ 
    Given nlon, nlat, lon, and lat
    Return Geopandas DataFrame of grid points
    """
    x, y = np.meshgrid(range(nlon), range(nlat))
    # assign 80-km grid points to geopandas dataframe
    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel()})
    geometry = gpd.points_from_xy(x=lon.ravel(), y=lat.ravel())
    # Avoid pyproj.exceptions.CRSError: Invalid CRS input: <cartopy.crs.PlateCarree object
    crs = "EPSG:4326" # had to switch after casper upgrade Oct 2023
    grid = gpd.GeoDataFrame(
        df.set_index(["y", "x"]), geometry=geometry, crs=crs
    )
    return grid


def getmask(grid: gpd.GeoDataFrame, nlon: int, nlat: int) -> xr.DataArray:
    """
    Given a grid Geopandas DataFrame, nlon, and nlat,
    Return DataArray of True over CONUS False elsewhere
    """
    world = gpd.read_file("/glade/work/ahijevyc/share/shapeFiles/ne_110m_admin_0_countries")
    usa = world[world["NAME"] == "United States of America"]

    # get poly from gdf instead of depreciated geopandas.datasets

    #poly = gpd.GeoDataFrame.from_file(
    #    gpd.datasets.get_path("naturalearth_lowres")
    #)
    #usa_old = poly[poly.iso_a3 == "USA"]
    
    # lat/lon box around CONUS (no AK or HI)
    lat_point_list = [51, 51, 20, 20, 51]
    lon_point_list = [-130, -60, -60, -130, -130]
    clip_poly = Polygon(zip(lon_point_list, lat_point_list))
    clip_gdf = gpd.GeoDataFrame(crs="EPSG:4326", geometry=[clip_poly])
    conus_geom = gpd.overlay(usa, clip_gdf, how="intersection")

    points_in_conus = gpd.sjoin(grid, conus_geom, how="inner", predicate="within")
    mask_flat = np.zeros(len(grid), dtype=bool)
    # Set True where index exists in joined dataframe
    mask_flat[grid.index.isin(points_in_conus.index)] = True

    conus_mask = xr.DataArray(
        mask_flat.reshape(nlat, nlon),
        dims=["y","x"],
        coords={"y": np.arange(nlat), "x": np.arange(nlon)},
    )
    return conus_mask


# TODO: use getmask() instead of pickle file. results are different.
mask = pickle.load(open('/glade/work/ahijevyc/NSC_objects/HRRR/usamask_mod.pk', 'rb'))
mask = xr.DataArray(mask.reshape((nlat,nlon)), 
        coords=dict(y=range(nlat), x=range(nlon)), dims=["y","x"])


class GridManager:
    """
    Handles grid generation and masking for G211 variants.
    factor=1: 80km grid (93x65)
    factor=2: 40km grid (185x129)
    """
    def __init__(self, factor: int = 1):
        LLX, LLY, URX, URY = get_projection_bounds()
        
        # Calculate dimensions (Factor 1 = 80km, Factor 2 = 40km)
        self.nlon = 93 * factor - (factor - 1)
        self.nlat = 65 * factor - (factor - 1)

        self.xs = np.linspace(LLX, URX, self.nlon)
        self.ys = np.linspace(LLY, URY, self.nlat)
        self.xv, self.yv = np.meshgrid(self.xs, self.ys)
        
        # Transform back to Lat/Lon for GeoDataFrame
        ll3 = ccrs.PlateCarree().transform_points(G211, self.xv, self.yv)
        self.lon = ll3[:, :, 0]
        self.lat = ll3[:, :, 1]

        self.grid = getgdf(self.nlon, self.nlat, self.lon, self.lat)
        
        self.mask = getmask(self.grid, self.nlon, self.nlat)

class x2(GridManager):
    """
    DEPRECIATED: Legacy wrapper for GridManager(factor=2).
    Half grid spacing of G211
    """
    def __init__(self):
        warnings.warn(
            "class 'x2' is deprecated. Please use 'GridManager(factor=2)' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(factor=2)

