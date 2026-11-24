"""
get lightning observations
"""
import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import geopandas
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import spatial
import xarray

from ahijevyc import G211
from ahijevyc import nclcmaps
from ml_functions import load_df

test = nclcmaps.colors["MPL_Greys"][6:-25:] + nclcmaps.colors["MPL_Reds"]
cmap = ListedColormap(test, "GreysReds")


def get_obsgdf(args, valid_start, obsvar, rptdist):
    """
    read lightning observations
    convert to geopandas DataFrame (obsgdf)
    """
    twin = args.twin
    f = f"{obsvar}_{rptdist}km_{twin}hr"
    df = load_df(args)
    df = df[df.valid_time == valid_start + pd.Timedelta(hours=twin / 2)]
    df = df[[f, "lon", "lat", "x", "y"]].rename(columns={f: obsvar})

    obsgdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    obsgdf = obsgdf.set_crs(ccrs.PlateCarree())

    return obsgdf

def get_obs(valid_start:pd.Timestamp, valid_end:pd.Timestamp, obsvar:str, twin:float, rptdist: float):
    assert obsvar in ["cg", "ic", "cg.ic", "flashes"], f"unexpected obsvar {obsvar}" 
    if obsvar in ["cg", "ic", "cg.ic"]:
        # Earth Networks (previously WeatherBug) Total Lightning Network (ENTLN)
        # Add up lighting counts in 30-minute blocks spanning [valid_start, valid_end).
        # cgds lightning counts are indexed by time_coverage_start. time_coverage_start
        # is the same thing as valid_start. Therefore, we want the index slice to be
        # slice(valid_start, valid_end - pd.Timedelta(minutes=30)). Because slice start and end are
        # inclusive, the slice end must be 30 minutes before valid_end.
        # Don't expect to find all 30-min blocks within valid time range.
        # If there is no lightning, that 30-min block will be missing.
        # So select with a time slice, not a list of 30-min blocks.
        wbugtimes = slice(valid_start, valid_end - pd.Timedelta(minutes=30))
        expected_times = twin * 2  # 30-min blocks in twin
        cgds = xarray.open_dataset(Path("wbug_lightning") / f"flash.{rptdist}km_30min.nc")
        found_times = cgds.sel(time_coverage_start=wbugtimes).time_coverage_start.size
        if found_times == 0:
            logging.warning(f"no wbug for [{valid_start},{valid_end})")
            return None
        if found_times < expected_times:
            logging.warning(
                f"found {found_times}/{expected_times} wbug for [{valid_start},{valid_end})"
            )
            # If you wish to fill missing blocks with the average of non-missing blocks
            # in the time window, don't return None now. In other words, comment next line.
            return None
        ds = (
            cgds.sel(time_coverage_start=wbugtimes).mean(
                dim="time_coverage_start")
            * (valid_end - valid_start)
            / pd.Timedelta(minutes=30)
        )
        ds["cg.ic"] = ds.cg + ds.ic
    elif obsvar == "flashes":
        # {twin}-hour blocks spanning valid time range
        # Blocks are indexed by their time coverage center,
        # so the start index is 1/2 twin hours after valid_start
        # and the end index is 1/2 twin hour prior to valid_end.
        glmtimes = pd.date_range(
            start=valid_start + pd.Timedelta(hours=twin / 2),
            end=valid_end - pd.Timedelta(hours=twin / 2),
            freq=f"{twin}H",
        )
        ifiles = [
            os.path.join(
                "GLM",
                t.strftime("%Y"),
                t.strftime("%Y%m%d_%H%M") + f".glm_{rptdist}km_{twin}hr.nc",
            )
            for t in glmtimes
        ]
        ifiles = [x for x in ifiles if os.path.exists(x)]
        if len(ifiles) == 0:
            logging.warning(f"no GLM for {glmtimes}")
            return None
        ds = xarray.open_mfdataset(ifiles).sum(dim="time")
    else:
        logging.error(f"unexpected obsvar {obsvar}")

    if rptdist == 20:
        # in the case of rptdist=20 return pts on coarser grid
        # find nearest indices of 40-km grid.
        lats = G211.x2().lat.ravel()
        lons = G211.x2().lon.ravel()
        x = G211.lon.ravel()
        y = G211.lat.ravel()
        tree = spatial.KDTree(list(zip(lons, lats)))
        dist, indices = tree.query(list(zip(x, y)))
        ltg_sum_coarse = ds.stack(pt=("y", "x")).isel(pt=indices)

        # replace 40km grid coordinates with 80km grid coordinates
        c = ltg_sum_coarse.coords
        c.update(G211.mask.stack(pt=("y", "x")).coords)
        ds = ltg_sum_coarse.assign_coords(c).unstack(dim="pt")

    assert obsvar in ds, f"{obsvar} not in obsgdf. did you assign the correct dataset?"

    return ds


def ztfs(x, how: str="nearest"):
    """
    zero, ten, forty, seventy
    If how="floor", round x DOWN to nearest probability level of
    SPC Thunderstorm Outlook.
    If how="nearest", round x to nearest (down OR up).
    """
    assert x >= 0
    assert x <= 1
    eps = 1e-12
    if how == "floor":
        if x < 0.1:
            return 0
        if x < 0.4:
            return 0.1 + eps # avoid floating point 0.1->0.099999999
        if x < 0.7:
            return 0.4
        return 0.7
    else:
        assert how == "nearest"
        if x < 0.05:
            return 0
        if x < 0.25:
            return 0.1 + eps
        if x < 0.55:
            return 0.4
        return 0.7
