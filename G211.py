import cartopy
import numpy as np


g211 = cartopy.crs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))
width=93
height=65

llcrnrlon = -133.459
llcrnrlat=12.19
urcrnrlon=-49.38641
urcrnrlat=57.2894
lons = np.array([llcrnrlon, urcrnrlon])
lats = np.array([llcrnrlat, urcrnrlat])

projected_corners = g211.transform_points(
    cartopy.crs.PlateCarree(), lons, lats)

xs = np.linspace( projected_corners[0, 0], projected_corners[1, 0], width)
ys = np.linspace( projected_corners[0, 1], projected_corners[1, 1], height)
xv, yv = np.meshgrid(xs,ys)
llz = cartopy.crs.PlateCarree().transform_points(g211,xv,yv)
lon = llz[:,:,0]
lat = llz[:,:,1]

def x2():
    """
    Half spacing (40km) compared to G211 (80km)
    """
    global width, height, projected_corners
    width=width*2
    height=height*2

    xs = np.linspace( projected_corners[0, 0], projected_corners[1, 0], width)
    ys = np.linspace( projected_corners[0, 1], projected_corners[1, 1], height)
    xv, yv = np.meshgrid(xs,ys)
    llz = cartopy.crs.PlateCarree().transform_points(g211,xv,yv)
    lon = llz[:,:,0]
    lat = llz[:,:,1]
    return lon, lat

