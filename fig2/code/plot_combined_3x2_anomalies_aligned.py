#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_combined_3x2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-combined-3x2")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cmaps

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Paths
# =============================================================================
PATH_Z = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/Z_1000_50hpa_1982_2025_JJA_detrended.nc"
PATH_W = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/W_1982_2025_1000_200hpa_detrended_signal.nc"
PATH_TCC = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/TCC_1982_2025_summer_detrended.nc"
PATH_QNET = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/Qnet_new/Qnet_1982_2025_summer_detrended.nc"
PATH_MLD = "/gpfs/hpc/home/syxz/5250101004/files/Grayson/data/NCEP/obml/dbss_obml_1982_2025.nc"
PATH_SLP = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/slp_1982-2025.nc"
PATH_10UV = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/10uv_1982_2025.nc"
SHP_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/WORLD_SHP/continent.shp"

OUT_PNG = Path(__file__).resolve().parent / "combined_3x2_anomalies_2025_summer_aligned_new.png"


# =============================================================================
# Config
# =============================================================================
TARGET_YEAR = 2025
CLIM_START = 1982
CLIM_END = 2024
G_ACCEL = 9.80665

LON_MIN, LON_MAX = 90.0, 180.0
LAT_MIN, LAT_MAX = 0.0, 80.0
SECTION_LON_MIN, SECTION_LON_MAX = 100.0, 200.0
SECTION_LAT_MIN, SECTION_LAT_MAX = 27.0, 48.0
BOX_LONS = [127.0, 164.0, 164.0, 127.0, 127.0]
BOX_LATS = [27.0, 27.0, 48.0, 48.0, 27.0]

TITLE_SIZE = 18
TICK_SIZE = 17
CBAR_TICK_SIZE = 10

_LAND_FEATURE = None


# =============================================================================
# Helpers
# =============================================================================
def coord_slice(coord, lower, upper):
    values = np.asarray(coord)
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def seasonal_yearly_mean(da, time_dim):
    da_jja = da.where(da[time_dim].dt.month.isin([6, 7, 8]), drop=True)
    return da_jja.groupby(f"{time_dim}.year").mean(time_dim)


def target_minus_clim(yearly_da):
    clim = yearly_da.sel(year=slice(CLIM_START, CLIM_END))
    anomaly = yearly_da.sel(year=TARGET_YEAR) - clim.mean("year")
    std = clim.std("year")
    return anomaly.load(), std.load()


def add_land(ax):
    global _LAND_FEATURE
    try:
        if _LAND_FEATURE is None:
            reader = shpreader.Reader(SHP_PATH)
            _LAND_FEATURE = ShapelyFeature(
                reader.geometries(),
                ccrs.PlateCarree(),
                edgecolor="black",
                facecolor="none",
                linewidth=0.7,
                zorder=3,
            )
        ax.add_feature(_LAND_FEATURE)
    except Exception:
        ax.coastlines(linewidth=0.7)


def format_map_axis(ax):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(LON_MIN, LON_MAX + 1, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(LAT_MIN, LAT_MAX + 1, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=TICK_SIZE)


def add_target_box(ax):
    ax.plot(
        BOX_LONS,
        BOX_LATS,
        color="black",
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )


def draw_significance_hatching(ax, x, y, anom, std, transform=None):
    sig = (np.abs(anom) > std).astype(float)
    kwargs = {"levels": [0.5, 1.5], "hatches": ["..."], "colors": "none", "zorder": 4}
    if transform is not None:
        kwargs["transform"] = transform
    ax.contourf(x, y, sig, **kwargs)


def format_pressure_axis(ax, upper_level):
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(1000, upper_level)
    tick_candidates = [1000, 850, 700, 500, 300, 200, 100, 50]
    ticks = [tick for tick in tick_candidates if upper_level <= tick <= 1000]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(labelsize=TICK_SIZE)


# =============================================================================
# Panels
# =============================================================================
def plot_panel_a(fig, gs):
    ax = fig.add_subplot(gs[0, 0])
    with xr.open_dataset(PATH_Z) as ds:
        da = ds["z_detrended"] / G_ACCEL
        da = da.sel(
            latitude=coord_slice(ds["latitude"], SECTION_LAT_MIN, SECTION_LAT_MAX),
            longitude=coord_slice(ds["longitude"], SECTION_LON_MIN, SECTION_LON_MAX),
            pressure_level=coord_slice(ds["pressure_level"], 50.0, 1000.0),
        )
        da = da.mean("latitude")
        yearly = seasonal_yearly_mean(da, "valid_time")
        anom, std = target_minus_clim(yearly)

    levels = np.linspace(-60, 60, 25)
    cf = ax.contourf(
        anom.longitude,
        anom.pressure_level,
        anom.values,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
    )
    draw_significance_hatching(ax, anom.longitude, anom.pressure_level, anom.values, std.values)

    ax.axvline(x=129, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.axvline(x=164, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.set_xticks(np.arange(100, 201, 30))
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    format_pressure_axis(ax, 50)
    ax.set_title(
        "(a) 2025 JJA Z Anomaly @ 27-48°N",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8, ticks=[-60, -40, -20, 0, 20, 40, 60])
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


def plot_panel_b(fig, gs):
    ax = fig.add_subplot(gs[0, 1])
    with xr.open_dataset(PATH_W) as ds:
        da = ds["w_detrended"].sel(
            latitude=coord_slice(ds["latitude"], SECTION_LAT_MIN, SECTION_LAT_MAX),
            longitude=coord_slice(ds["longitude"], SECTION_LON_MIN, SECTION_LON_MAX),
            pressure_level=coord_slice(ds["pressure_level"], 200.0, 1000.0),
        )
        da = da.mean("latitude")
        yearly = seasonal_yearly_mean(da, "valid_time")
        anom, std = target_minus_clim(yearly)
        anom = (anom * 100.0).load()
        std = (std * 100.0).load()

    vmax = np.ceil(np.nanpercentile(np.abs(anom.values), 98) * 10.0) / 10.0
    vmax = max(vmax, 0.2)
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(
        anom.longitude,
        anom.pressure_level,
        anom.values,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
    )
    draw_significance_hatching(ax, anom.longitude, anom.pressure_level, anom.values, std.values)

    ax.axvline(x=129, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.axvline(x=164, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.set_xticks(np.arange(100, 201, 30))
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    format_pressure_axis(ax, 200)
    ax.set_title(
        "(b) 2025 JJA W Anomaly @ 27-48°N",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


def plot_panel_c(fig, gs, proj):
    ax = fig.add_subplot(gs[1, 0], projection=proj)
    with xr.open_dataset(PATH_TCC) as ds:
        da = ds["tcc_detrended"].sel(
            latitude=coord_slice(ds["latitude"], LAT_MIN, LAT_MAX),
            longitude=coord_slice(ds["longitude"], LON_MIN, LON_MAX),
        )
        yearly = seasonal_yearly_mean(da, "valid_time")
        anom, std = target_minus_clim(yearly)
        anom = (anom * 100.0).load()
        std = (std * 100.0).load()

    cf = ax.contourf(
        anom.longitude,
        anom.latitude,
        anom.values,
        levels=np.arange(-15, 16, 2),
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    draw_significance_hatching(
        ax,
        anom.longitude,
        anom.latitude,
        anom.values,
        std.values,
        transform=ccrs.PlateCarree(),
    )
    add_land(ax)
    format_map_axis(ax)
    add_target_box(ax)
    ax.set_title(
        "(c) 2025 JJA TCC Anomaly",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


def plot_panel_d(fig, gs, proj):
    ax = fig.add_subplot(gs[1, 1], projection=proj)
    with xr.open_dataset(PATH_QNET) as ds:
        da = ds["avg_snswrf_detrended"].sel(
            latitude=coord_slice(ds["latitude"], LAT_MIN, LAT_MAX),
            longitude=coord_slice(ds["longitude"], LON_MIN, LON_MAX),
        )
        yearly = seasonal_yearly_mean(da, "valid_time")
        anom, std = target_minus_clim(yearly)

    vmax = np.ceil(np.nanpercentile(np.abs(anom.values), 99) / 5.0) * 5.0
    vmax = max(vmax, 10.0)
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(
        anom.longitude,
        anom.latitude,
        anom.values,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    draw_significance_hatching(
        ax,
        anom.longitude,
        anom.latitude,
        anom.values,
        std.values,
        transform=ccrs.PlateCarree(),
    )
    add_land(ax)
    format_map_axis(ax)
    add_target_box(ax)
    ax.set_title(
        "(d) 2025 JJA Surface Net Shortwave Anomaly",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


def plot_panel_e(fig, gs, proj):
    ax = fig.add_subplot(gs[2, 0], projection=proj)
    with xr.open_dataset(PATH_MLD) as ds:
        da = ds["dbss_obml"].sel(
            lat=coord_slice(ds["lat"], LAT_MIN, LAT_MAX),
            lon=coord_slice(ds["lon"], LON_MIN, LON_MAX),
        )
        yearly = seasonal_yearly_mean(da, "time")
        anom, std = target_minus_clim(yearly)

    vmax = int(np.ceil(np.nanpercentile(np.abs(anom.values), 98) / 5.0) * 5.0)
    vmax = max(vmax, 20)
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(
        anom.lon,
        anom.lat,
        anom.values,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    draw_significance_hatching(
        ax,
        anom.lon,
        anom.lat,
        anom.values,
        std.values,
        transform=ccrs.PlateCarree(),
    )
    add_land(ax)
    format_map_axis(ax)
    add_target_box(ax)
    ax.set_title(
        "(e) 2025 JJA GODAS Mixed Layer Depth Anomaly",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8, pad=0.03)
    cbar.set_label("m", fontsize=11)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


def plot_panel_f(fig, gs, proj):
    ax = fig.add_subplot(gs[2, 1], projection=proj)

    with xr.open_dataset(PATH_SLP) as ds_slp, xr.open_dataset(PATH_10UV) as ds_uv:
        msl = (ds_slp["msl"] / 100.0).sel(
            latitude=coord_slice(ds_slp["latitude"], LAT_MIN, LAT_MAX),
            longitude=coord_slice(ds_slp["longitude"], LON_MIN, LON_MAX),
        )
        u10 = ds_uv["u10"].sel(
            latitude=coord_slice(ds_uv["latitude"], LAT_MIN, LAT_MAX),
            longitude=coord_slice(ds_uv["longitude"], LON_MIN, LON_MAX),
        )
        v10 = ds_uv["v10"].sel(
            latitude=coord_slice(ds_uv["latitude"], LAT_MIN, LAT_MAX),
            longitude=coord_slice(ds_uv["longitude"], LON_MIN, LON_MAX),
        )

        msl_yearly = seasonal_yearly_mean(msl, "valid_time")
        u10_yearly = seasonal_yearly_mean(u10, "valid_time")
        v10_yearly = seasonal_yearly_mean(v10, "valid_time")
        ws10_yearly = np.hypot(u10_yearly, v10_yearly)

        msl_anom, _ = target_minus_clim(msl_yearly)
        u10_anom, _ = target_minus_clim(u10_yearly)
        v10_anom, _ = target_minus_clim(v10_yearly)
        ws10_anom, _ = target_minus_clim(ws10_yearly)

    vmax = np.ceil(np.nanpercentile(np.abs(ws10_anom.values), 98) * 10.0) / 10.0
    vmax = max(vmax, 0.5)
    speed_levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(
        ws10_anom.longitude,
        ws10_anom.latitude,
        ws10_anom.values,
        levels=speed_levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    slp_vmax = np.ceil(np.nanpercentile(np.abs(msl_anom.values), 98) * 2.0) / 2.0
    slp_vmax = max(slp_vmax, 1.0)
    slp_step = 0.5 if slp_vmax <= 3.0 else 1.0
    slp_levels = np.arange(-slp_vmax, slp_vmax + slp_step * 0.5, slp_step)
    slp_levels = slp_levels[np.abs(slp_levels) > 1.0e-8]
    linestyles = ["dashed" if level < 0 else "solid" for level in slp_levels]
    cs = ax.contour(
        msl_anom.longitude,
        msl_anom.latitude,
        msl_anom.values,
        levels=slp_levels,
        colors="black",
        linewidths=0.9,
        linestyles=linestyles,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.contour(
        msl_anom.longitude,
        msl_anom.latitude,
        msl_anom.values,
        levels=[0.0],
        colors="black",
        linewidths=1.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    if len(cs.levels) > 0:
        label_levels = cs.levels[::2]
        fmt = "%.1f" if slp_step < 1.0 else "%d"
        ax.clabel(cs, levels=label_levels, inline=True, fmt=fmt, fontsize=8)

    stride = 12
    lon2d, lat2d = np.meshgrid(u10_anom.longitude.values, u10_anom.latitude.values)
    u_plot = u10_anom.values[::stride, ::stride]
    v_plot = v10_anom.values[::stride, ::stride]
    q = ax.quiver(
        lon2d[::stride, ::stride],
        lat2d[::stride, ::stride],
        u_plot,
        v_plot,
        transform=ccrs.PlateCarree(),
        color="black",
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=3.8,
        zorder=6,
    )

    vec_mag = np.hypot(u_plot, v_plot)
    finite_vec = vec_mag[np.isfinite(vec_mag)]
    ref_vec = 1.0
    if finite_vec.size > 0:
        ref_vec = float(np.round(np.nanpercentile(finite_vec, 90) * 2.0) / 2.0)
        ref_vec = max(ref_vec, 0.5)
    ax.quiverkey(
        q,
        X=0.90,
        Y=1.04,
        U=ref_vec,
        label=rf"{ref_vec:.1f} m s$^{{-1}}$",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 9},
    )

    add_land(ax)
    format_map_axis(ax)
    add_target_box(ax)
    ax.set_title(
        "(f) 2025 JJA SLP and 10m Wind Anomalies",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink=0.8, pad=0.03)
    cbar.set_label(r"m s$^{-1}$", fontsize=11)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


# =============================================================================
# Main
# =============================================================================
def main():
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(18, 21), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    plot_panel_a(fig, gs)
    plot_panel_b(fig, gs)
    plot_panel_c(fig, gs, proj)
    plot_panel_d(fig, gs, proj)
    plot_panel_e(fig, gs, proj)
    plot_panel_f(fig, gs, proj)

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
