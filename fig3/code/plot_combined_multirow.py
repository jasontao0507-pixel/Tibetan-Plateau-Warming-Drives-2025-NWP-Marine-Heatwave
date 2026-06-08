#!/usr/bin/env python3
"""
Combined multi-row figure for Summer 2025.

Panels:
  (a) 200-hPa meridional wind anomaly
  (b) T2m anomaly map
  (c) Temperature anomaly cross-section along 36N
  (d) Geopotential-height anomaly cross-section along 36N
  (e) Eddy Z200 anomaly and Plumb wave-activity flux
  (f) Standardized T2m index time series

This version follows the layout/content of:
  /gpfs/hpc/home/syxz/5250101004/JasonTao/scripts/202604/20260405/new/plot_combined_multirow.py

Requested changes:
  1. Panel (e) uses the plotting result/style of:
     /gpfs/hpc/home/syxz/5250101004/JasonTao/scripts/202603/20260329/plot_plumb_z200eddy_2025_summer_40E_180E_20_60N.py
  2. Panels (a) and (e) use 40-180E, 20-60N.
  3. Code organization is rewritten in a clearer style, similar to:
     /gpfs/hpc/home/syxz/5250101004/JasonTao/scripts/202604/20260405/plot_fig4_tibet_36N.py
  4. Output is saved in the current directory.
"""

from pathlib import Path
import os
import warnings

import numpy as np
import xarray as xr
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_plot_combined_multirow")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter, minimum_filter
from scipy.interpolate import PchipInterpolator
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================== Global style ========================
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


# ======================== Output ========================
CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_FIG = CURRENT_DIR / "combined_multirow_figure_new.png"


# ======================== File paths ========================
UVT_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/uvt_1_1000hpa_1982_2025_summer_detrended.nc"
Z_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/Z_1000_50hpa_1982_2025_JJA_detrended.nc"
T2M_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/t2m_1982_202601_detrended.nc"
ORO_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/dixing/geo.nc"
WORLD_SHP = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/WORLD_SHP/continent.shp"
TP_SHP = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/TP_SHP/TPBoundary_new(2021)/TPBoundary_new(2021).shp"
TP3000_SHP = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/TP_SHP/TPBoundary_3000m/TPBoundary_3000m.shp"
DSST_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/dSST_detrended_signal.nc"


# ======================== Constants ========================
TARGET_YEAR = 2025
CLIM_START = 1982
CLIM_END = 2024

GRAVITY = 9.80665
OMEGA = 7.292e-5
EARTH_RADIUS = 6.371e6
P0_HPA = 1000.0

LEVEL_200 = 200.0

MAP_LON_MIN = 40.0
MAP_LON_MAX = 180.0
MAP_LAT_MIN = 25.0
MAP_LAT_MAX = 55.0

T2M_LON_MIN = 60.0
T2M_LON_MAX = 110.0
T2M_LAT_MIN = 17.0
T2M_LAT_MAX = 50.0
BOX_LON_MIN = 70.0
BOX_LON_MAX = 78.0
BOX_LAT_MIN = 33.0
BOX_LAT_MAX = 40.0

LAT_CROSS = 36.0
XS_LON_MIN = 50.0
XS_LON_MAX = 90.0

PANEL_E_COMP_LAT_MIN = 10.0
PANEL_E_COMP_LAT_MAX = 80.0
PANEL_E_LEV_TOP = 100.0
PANEL_E_LEV_BOTTOM = 300.0
PANEL_E_VECTOR_STD_FACTOR = 0.5

# Region for panel (g) SST regression  —  same as plot_combined_3x2_anomalies.py panel c
REG_LON_MIN = 110.0
REG_LON_MAX = 180.0
REG_LAT_MIN = 20.0
REG_LAT_MAX = 70.0


# ======================== Figure layout ========================
FIG_SIZE = (20, 26)

AX_A = [0.07, 0.795, 0.89, 0.22]
CAX_A = [0.17, 0.8, 0.71, 0.008]

AX_B = [0.07, 0.547, 0.29, 0.27]
CAX_B = [0.086, 0.58, 0.24, 0.01]

AX_C = [0.425, 0.547, 0.20, 0.21]
CAX_C = [0.635, 0.557, 0.012, 0.18]

AX_D = [0.735, 0.547, 0.20, 0.21]
CAX_D = [0.945, 0.557, 0.012, 0.18]

AX_E = [0.07, 0.315, 0.89, 0.22]
CAX_E = [0.15, 0.32, 0.71, 0.008]

AX_F = [0.07, 0.11, 0.5, 0.16]

AX_G  = [0.65, 0.05, 0.33, 0.28]
CAX_G = [0.99, 0.12, 0.012, 0.12]


# ======================== Helpers ========================
def add_shapefile(ax, path, edgecolor="black", facecolor="none", linewidth=0.8, linestyle="-"):
    """Draw a shapefile on a cartopy axis."""
    if not os.path.exists(path):
        return
    try:
        for geom in Reader(path).geometries():
            ax.add_geometries(
                [geom],
                ccrs.PlateCarree(),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
    except Exception:
        return


def coord_name(ds, *candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of {candidates} found in dataset.")


def select_lat_slice(lat_values, lat_min, lat_max):
    if float(lat_values[0]) > float(lat_values[-1]):
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def subset_lat_lon(ds, lat_name="latitude", lon_name="longitude",
                   lat_min=MAP_LAT_MIN, lat_max=MAP_LAT_MAX,
                   lon_min=MAP_LON_MIN, lon_max=MAP_LON_MAX):
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    if float(lat_values[0]) > float(lat_values[-1]):
        lat_mask = (lat_values <= lat_max) & (lat_values >= lat_min)
    else:
        lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)
    lon_mask = (lon_values >= lon_min) & (lon_values <= lon_max)

    return ds.isel({lat_name: np.where(lat_mask)[0], lon_name: np.where(lon_mask)[0]})


def summer_year_mean(da, year, time_name="valid_time"):
    summer = da.sel({time_name: da[time_name].dt.month.isin([6, 7, 8])})
    summer_year = summer.groupby(f"{time_name}.year").mean(time_name)
    return summer_year.sel(year=year)


def summer_climatology(da, start_year=CLIM_START, end_year=CLIM_END, time_name="valid_time"):
    summer = da.sel({time_name: da[time_name].dt.month.isin([6, 7, 8])})
    summer_year = summer.groupby(f"{time_name}.year").mean(time_name)
    return summer_year.sel(year=slice(start_year, end_year)).mean("year")


def ddlon_rad(da):
    return da.differentiate("longitude") * (180.0 / np.pi)


def topo_to_pressure(height_m):
    height_m = np.maximum(height_m, 0)
    return 1013.25 * np.exp(-height_m / 7500.0)


def round_up_nice(value):
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / 10.0 ** exponent
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * 10.0 ** exponent)


def set_map_ticks(ax, lon_min, lon_max, lat_min, lat_max, lon_step=20, lat_step=10):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(lon_min, lon_max + 0.1, lon_step), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min, lat_max + 0.1, lat_step), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)


def find_anomaly_centers(field, lat, lon):
    smoothed = gaussian_filter(field, sigma=6)
    extrema_window = (25, 25)
    edge_margin = 10
    max_mask = smoothed == maximum_filter(smoothed, size=extrema_window, mode="nearest")
    min_mask = smoothed == minimum_filter(smoothed, size=extrema_window, mode="nearest")

    def collect(mask, sign):
        points = []
        for iy, ix in np.argwhere(mask):
            if iy < edge_margin or iy >= field.shape[0] - edge_margin:
                continue
            if ix < edge_margin or ix >= field.shape[1] - edge_margin:
                continue
            value = field[iy, ix]
            if sign > 0 and value <= 0:
                continue
            if sign < 0 and value >= 0:
                continue
            points.append((abs(float(smoothed[iy, ix])), float(lat[iy]), float(lon[ix])))
        points.sort(reverse=True)
        return points[:2]

    centers = collect(max_mask, 1) + collect(min_mask, -1)
    centers.sort(key=lambda item: item[2])
    return [(lat0, lon0) for _, lat0, lon0 in centers]


# ======================== Data loaders ========================
def load_v200_panel_a():
    print("Loading panel (a): V200 anomaly...")
    ds = xr.open_dataset(UVT_PATH, chunks="auto")
    v200 = ds["v_detrended"].sel(pressure_level=LEVEL_200)
    yearly = v200.groupby("valid_time.year").mean("valid_time")
    climatology = yearly.sel(year=slice(CLIM_START, CLIM_END))
    anomaly = (yearly.sel(year=TARGET_YEAR) - climatology.mean("year")).compute()
    std = climatology.std("year").compute()
    ds.close()

    out = xr.Dataset({"v_anomaly": anomaly, "v_std": std})
    return subset_lat_lon(out)


def load_jet_axis_panel_a():
    """Climatological (1982-2024) summer 200-hPa westerly jet axis.

    At every longitude the jet axis is the latitude where the climatological
    JJA-mean zonal wind U200 is maximum. Returns (lon, jet_lat) spanning the
    full panel-(a) longitude range, lightly smoothed for a clean curve.
    """
    print("Loading panel (a): climatological U200 jet axis...")
    ds = xr.open_dataset(UVT_PATH, chunks="auto")
    u200 = ds["u_detrended"].sel(pressure_level=LEVEL_200)
    yearly = u200.groupby("valid_time.year").mean("valid_time")
    clim = yearly.sel(year=slice(CLIM_START, CLIM_END)).mean("year")
    clim = clim.sortby("latitude").sel(
        latitude=slice(MAP_LAT_MIN, MAP_LAT_MAX),
        longitude=slice(MAP_LON_MIN, MAP_LON_MAX),
    ).transpose("latitude", "longitude").compute()
    ds.close()

    lats = clim["latitude"].values
    lons = clim["longitude"].values
    jet_lat = lats[np.argmax(clim.values, axis=0)].astype(float)
    jet_lat = gaussian_filter1d(jet_lat, sigma=8, mode="nearest")
    return lons, jet_lat


def load_plumb_panel_e():
    print("Loading panel (e): Plumb wave-activity flux...")
    dsz = xr.open_dataset(Z_PATH, chunks="auto")
    dsu = xr.open_dataset(UVT_PATH, chunks="auto")

    common_levels = np.intersect1d(dsz["pressure_level"].values, dsu["pressure_level"].values)
    actual_top = max(float(common_levels.min()), PANEL_E_LEV_TOP)
    lev_sel = common_levels[(common_levels >= actual_top) & (common_levels <= PANEL_E_LEV_BOTTOM)]

    lat_slice = select_lat_slice(dsz["latitude"].values, PANEL_E_COMP_LAT_MIN, PANEL_E_COMP_LAT_MAX)

    phi_raw = dsz["z_detrended"].sel(pressure_level=lev_sel, latitude=lat_slice)
    u_raw = dsu["u_detrended"].sel(pressure_level=lev_sel, latitude=lat_slice)
    v_raw = dsu["v_detrended"].sel(pressure_level=lev_sel, latitude=lat_slice)

    phi_anom = summer_year_mean(phi_raw, TARGET_YEAR) - summer_climatology(phi_raw)
    u_anom = summer_year_mean(u_raw, TARGET_YEAR) - summer_climatology(u_raw)
    v_anom = summer_year_mean(v_raw, TARGET_YEAR) - summer_climatology(v_raw)

    phi_prime = phi_anom - phi_anom.mean("longitude")
    u_prime = u_anom - u_anom.mean("longitude")
    v_prime = v_anom - v_anom.mean("longitude")

    lat_rad = np.deg2rad(phi_prime["latitude"].astype(np.float64))
    cosphi = np.cos(lat_rad)
    sin2phi = np.sin(2.0 * lat_rad)
    sin2phi = xr.where(np.abs(sin2phi) < 1e-10, np.nan, sin2phi)
    coef_lon = 1.0 / (2.0 * OMEGA * EARTH_RADIUS * sin2phi)
    p_ratio = phi_prime["pressure_level"] / P0_HPA

    d_vphi = ddlon_rad(v_prime * phi_prime)
    d_uphi = ddlon_rad(u_prime * phi_prime)

    fs_x = p_ratio * cosphi * (v_prime * v_prime - coef_lon * d_vphi)
    fs_y = p_ratio * cosphi * (-u_prime * v_prime + coef_lon * d_uphi)

    z200_raw = dsz["z_detrended"].sel(pressure_level=LEVEL_200, latitude=lat_slice) / GRAVITY
    z200_2025 = summer_year_mean(z200_raw, TARGET_YEAR)
    z200_clim = summer_climatology(z200_raw)
    z200_anom = z200_2025 - z200_clim
    z200_eddy = (z200_anom - z200_anom.mean("longitude")).compute()

    out = xr.Dataset(
        {
            "Fsx": fs_x.sel(pressure_level=LEVEL_200).compute(),
            "Fsy": fs_y.sel(pressure_level=LEVEL_200).compute(),
            "z200_eddy": z200_eddy,
        }
    )

    dsz.close()
    dsu.close()
    return subset_lat_lon(out)


def load_t2m_panel_b():
    print("Loading panel (b): T2m anomaly map...")
    ds = xr.open_dataset(T2M_PATH)
    t2m = ds["t2m_detrended"]
    summer = t2m.sel(valid_time=t2m.valid_time.dt.month.isin([6, 7, 8]))
    yearly = summer.groupby("valid_time.year").mean("valid_time")
    climatology = yearly.sel(year=slice(CLIM_START, CLIM_END)).mean("year")
    std = yearly.sel(year=slice(CLIM_START, CLIM_END)).std("year")
    anomaly = yearly.sel(year=TARGET_YEAR) - climatology
    stippling = np.abs(anomaly) > 1.5 * std
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    ds.close()
    return anomaly.values, stippling.values, lon, lat


def load_t2m_index_panel_f():
    print("Loading panel (f): T2m index...")
    ds = xr.open_dataset(T2M_PATH)
    t2m = ds["t2m_detrended"]
    summer = t2m.sel(valid_time=t2m.valid_time.dt.month.isin([6, 7, 8]))
    yearly = summer.groupby("valid_time.year").mean("valid_time")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_mask = (lat >= BOX_LAT_MIN) & (lat <= BOX_LAT_MAX)
    lon_mask = (lon >= BOX_LON_MIN) & (lon <= BOX_LON_MAX)

    region = yearly.values[:, lat_mask, :][:, :, lon_mask]
    weights = np.cos(np.deg2rad(lat[lat_mask]))
    lat_weighted = np.average(region, axis=1, weights=weights)
    box_mean = np.nanmean(lat_weighted, axis=1)

    years = yearly["year"].values
    mask = (years >= 1982) & (years <= TARGET_YEAR)
    years = years[mask]
    values = box_mean[mask]
    index = (values - np.nanmean(values)) / np.nanstd(values)

    ds.close()
    return years, index


def load_cross_section(dataset_path, variable_candidates, title_name, p_top, divide_by_gravity=False):
    print(f"Loading {title_name} cross-section at {LAT_CROSS:.0f}N...")
    ds = xr.open_dataset(dataset_path)

    time_dim = coord_name(ds, "valid_time", "time")
    lat_dim = coord_name(ds, "latitude", "lat")
    lon_dim = coord_name(ds, "longitude", "lon")
    lev_dim = coord_name(ds, "pressure_level", "level")

    var_name = None
    for candidate in variable_candidates:
        if candidate in ds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        raise KeyError(f"Cannot find variables {variable_candidates} in {dataset_path}")

    da = ds[var_name].sel({lat_dim: LAT_CROSS}, method="nearest")
    lev_values = ds[lev_dim].values
    if float(lev_values[0]) > float(lev_values[-1]):
        pressure_slice = slice(1000.0, p_top)
    else:
        pressure_slice = slice(p_top, 1000.0)

    da = da.sel({lon_dim: slice(XS_LON_MIN, XS_LON_MAX), lev_dim: pressure_slice})
    summer = da.sel({time_dim: da[time_dim].dt.month.isin([6, 7, 8])})
    yearly = summer.groupby(f"{time_dim}.year").mean(time_dim)
    climatology = yearly.sel(year=slice(CLIM_START, CLIM_END)).mean("year")
    std = yearly.sel(year=slice(CLIM_START, CLIM_END)).std("year")
    anomaly = yearly.sel(year=TARGET_YEAR) - climatology

    if divide_by_gravity:
        anomaly = anomaly / GRAVITY
        std = std / GRAVITY

    anomaly = anomaly.compute()
    std = std.compute()
    stippling = (np.abs(anomaly) > std).compute()

    lon = anomaly[lon_dim].values
    plev = anomaly[lev_dim].values
    ds.close()
    return anomaly.values, stippling.values, lon, plev


def load_topography_profile():
    print("Loading topography profile...")
    ds = xr.open_dataset(ORO_PATH)
    lon_dim = coord_name(ds, "longitude", "lon")
    lat_dim = coord_name(ds, "latitude", "lat")

    topo = ds["z"].sel({lat_dim: LAT_CROSS}, method="nearest")
    if "time" in topo.dims:
        topo = topo.mean("time")
    topo = topo.sel({lon_dim: slice(XS_LON_MIN, XS_LON_MAX)})

    height_m = topo.values / GRAVITY
    pressure = topo_to_pressure(height_m)

    ds.close()
    return topo[lon_dim].values, pressure


# ======================== Panel drawers ========================
def draw_panel_a(fig, ax, cax, panel_ds, jet_lon, jet_lat):
    lon = panel_ds["v_anomaly"].longitude.values
    lat = panel_ds["v_anomaly"].latitude.values
    anomaly = panel_ds["v_anomaly"].values
    std = panel_ds["v_std"].values
    stippling = np.abs(anomaly) > std

    levels = np.linspace(-4.0, 4.0, 21)
    cf = ax.contourf(
        lon,
        lat,
        anomaly,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    add_shapefile(ax, WORLD_SHP, edgecolor="dimgray", linewidth=0.6)
    add_shapefile(ax, TP_SHP, edgecolor="black", linewidth=1.2)

    lon2d, lat2d = np.meshgrid(lon, lat)
    skip = 5
    ax.scatter(
        lon2d[::skip, ::skip][stippling[::skip, ::skip]],
        lat2d[::skip, ::skip][stippling[::skip, ::skip]],
        color="black",
        marker=".",
        s=0.8,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Climatological (1982-2024) summer 200-hPa jet axis: black dashed line
    # running across the whole panel (latitude of maximum mean U200 at each lon).
    ax.plot(
        jet_lon,
        jet_lat,
        color="black",
        linestyle=(0, (5, 2.5)),
        linewidth=5.0,
        solid_capstyle="round",
        dash_capstyle="round",
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    centers = find_anomaly_centers(anomaly, lat, lon)
    if centers:
        center_lats, center_lons = zip(*centers)
        center_lons = np.asarray(center_lons, dtype=float)
        center_lats = np.asarray(center_lats, dtype=float)

        # Bold green centers with a white halo so they pop over both warm and cool shading.
        ax.scatter(
            center_lons,
            center_lats,
            s=210,
            color="white",
            transform=ccrs.PlateCarree(),
            zorder=9,
        )
        ax.scatter(
            center_lons,
            center_lats,
            s=150,
            color="limegreen",
            edgecolors="black",
            linewidths=1.6,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

    set_map_ticks(ax, MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX)
    ax.set_title(
        f"(a) Meridional wind anomaly at 200 hPa ({TARGET_YEAR} JJA)",
        loc="left",
        fontweight="bold",
        fontsize=23,
        pad=5,
    )

    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=np.arange(-4, 5, 1))
    cbar.ax.tick_params(labelsize=18)


def draw_panel_b(fig, ax, cax, anomaly, stippling, lon, lat):
    if float(lat[0]) > float(lat[-1]):
        lat = lat[::-1]
        anomaly = anomaly[::-1, :]
        stippling = stippling[::-1, :]

    lat_mask = (lat >= T2M_LAT_MIN) & (lat <= T2M_LAT_MAX)
    lon_mask = (lon >= T2M_LON_MIN) & (lon <= T2M_LON_MAX)
    lat_sub = lat[lat_mask]
    lon_sub = lon[lon_mask]
    anomaly_sub = anomaly[np.ix_(lat_mask, lon_mask)]
    stippling_sub = stippling[np.ix_(lat_mask, lon_mask)]
    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

    cf = ax.contourf(
        lon2d,
        lat2d,
        anomaly_sub,
        levels=np.linspace(-2.0, 2.0, 21),
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    add_shapefile(ax, WORLD_SHP, edgecolor="dimgray", linewidth=0.6)
    add_shapefile(ax, TP_SHP, edgecolor="black", linewidth=1.2)

    skip = 3
    ax.scatter(
        lon2d[::skip, ::skip][stippling_sub[::skip, ::skip]],
        lat2d[::skip, ::skip][stippling_sub[::skip, ::skip]],
        color="black",
        marker=".",
        s=2.4,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    box = mpatches.Rectangle(
        (BOX_LON_MIN, BOX_LAT_MIN),
        BOX_LON_MAX - BOX_LON_MIN,
        BOX_LAT_MAX - BOX_LAT_MIN,
        linewidth=3.0,
        edgecolor="green",
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=8,
    )
    ax.add_patch(box)

    set_map_ticks(ax, T2M_LON_MIN, T2M_LON_MAX, T2M_LAT_MIN, T2M_LAT_MAX, lon_step=10, lat_step=10)
    ax.set_title("(b) T2m anomaly", loc="left", fontweight="bold", fontsize=23,pad=5)

    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.tick_params(labelsize=17)


def draw_cross_section_panel(fig, ax, cax, anomaly, stippling, lon, plev,
                             topo_lon, topo_pressure, p_top, title, cbar_fmt):
    vmax = round_up_nice(max(float(np.nanpercentile(np.abs(anomaly), 98)), 0.5))
    levels = np.linspace(-vmax, vmax, 21)

    cf = ax.contourf(
        lon,
        plev,
        anomaly,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
    )
    ax.contour(lon, plev, anomaly, levels=levels[::2], colors="black", linewidths=0.35)

    lon2d, plev2d = np.meshgrid(lon, plev)
    sx = max(1, len(lon) // 40)
    sy = max(1, len(plev) // 15)
    ax.scatter(
        lon2d[::sy, ::sx][stippling[::sy, ::sx]],
        plev2d[::sy, ::sx][stippling[::sy, ::sx]],
        color="black",
        marker=".",
        s=1.6,
        alpha=0.65,
        zorder=3,
    )

    ax.fill_between(topo_lon, topo_pressure, 1100, color="dimgray", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(1000, p_top)
    tick_candidates = [1000, 850, 700, 500, 400, 300, 200, 100, 50]
    y_ticks = [tick for tick in tick_candidates if p_top <= tick <= 1000]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlim(XS_LON_MIN, XS_LON_MAX)
    ax.set_xticks(np.arange(XS_LON_MIN, XS_LON_MAX + 1, 10))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.tick_params(labelsize=18)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=23,pad=5)

    cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(cbar_fmt))
    cbar.ax.tick_params(labelsize=18)


def draw_panel_e(fig, ax, cax, panel_ds):
    lon = panel_ds["longitude"].values
    lat = panel_ds["latitude"].values
    fsx = panel_ds["Fsx"].values
    fsy = panel_ds["Fsy"].values
    z200_eddy = panel_ds["z200_eddy"].values

    cf = ax.contourf(
        lon,
        lat,
        z200_eddy,
        levels=np.linspace(-60.0, 60.0, 25),
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    add_shapefile(ax, WORLD_SHP, edgecolor="dimgray", linewidth=0.6)
    add_shapefile(ax, TP3000_SHP, edgecolor="black", linewidth=1.1)

    lat_step = 20
    lon_step = 20
    lon_q = lon[::lon_step]
    lat_q = lat[::lat_step]
    fsx_q = fsx[::lat_step, ::lon_step]
    fsy_q = fsy[::lat_step, ::lon_step]

    magnitude = np.sqrt(fsx_q ** 2 + fsy_q ** 2)
    threshold = PANEL_E_VECTOR_STD_FACTOR * float(np.nanstd(magnitude))
    mask = np.isfinite(magnitude) & (magnitude > threshold)

    fsx_plot = np.where(mask, fsx_q, np.nan)
    fsy_plot = np.where(mask, fsy_q, np.nan)

    ref_arrow = 5.0
    quiver = ax.quiver(
        lon_q,
        lat_q,
        fsx_plot,
        fsy_plot,
        transform=ccrs.PlateCarree(),
        color="black",
        pivot="mid",
        scale=80.0,
        width=0.004,
        headwidth=4,
        headlength=5,
        zorder=7,
    )
    qk = ax.quiverkey(
        quiver,
        0.92,
        1.03,
        ref_arrow,
        f"{ref_arrow:.0f} m$^2$·s$^{{-2}}$",
        coordinates="axes",
        labelpos="E",
    )
    qk.text.set_fontsize(18)

    set_map_ticks(ax, MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX)
    ax.set_title(
        f"(e) Z200  and wave-activity flux anomaly",
        loc="left",
        fontweight="bold",
        fontsize=25,
        pad=5,
    )

    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=np.arange(-60, 61, 20))
    cbar.ax.tick_params(labelsize=18)


def draw_panel_f(ax, years, index):
    bar_colors = ["#d73027" if value >= 0 else "#4575b4" for value in index]
    bars = ax.bar(years, index, color=bar_colors, alpha=0.55, width=0.7, zorder=2)

    target_idx = np.where(years == TARGET_YEAR)[0]
    if len(target_idx):
        bars[target_idx[0]].set_edgecolor("black")
        bars[target_idx[0]].set_linewidth(2.0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlim(1981, 2026)
    tick_years = [1982, 1987, 1992, 1997, 2002, 2007, 2012, 2017, 2021, 2025]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(year) for year in tick_years], fontsize=17)

    for label in ax.get_xticklabels():
        if label.get_text() == str(TARGET_YEAR):
            label.set_color("black")
            label.set_fontweight("bold")

    ax.tick_params(axis="y", labelsize=17)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(
        f"(f) T2m index ({BOX_LON_MIN:.0f}°-{BOX_LON_MAX:.0f}°E, {BOX_LAT_MIN:.0f}°-{BOX_LAT_MAX:.0f}°N)",
        loc="left",
        fontweight="bold",
        fontsize = 23,
        pad=5,
    )


def load_sst_regression_panel_g(years, index):
    """Regress JJA summer-mean SST onto the standardised T2m index (1982-2025).

    Returns
    -------
    slope      : 2-D ndarray  [°C per std of index]
    sig_mask   : 2-D bool     True where p < 0.05
    lon, lat   : 1-D ndarrays
    """
    print("Loading panel (g): SST regression...")
    ds = xr.open_dataset(DSST_PATH, chunks={"time": 366})

    # subset to region of interest first (keeps memory small)
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    lat_slice = slice(REG_LAT_MIN, REG_LAT_MAX) if lat_vals[0] < lat_vals[-1] \
        else slice(REG_LAT_MAX, REG_LAT_MIN)
    lon_slice = slice(REG_LON_MIN, REG_LON_MAX)
    sst = ds["sst_detrended"].sel(lat=lat_slice, lon=lon_slice)

    # compute JJA annual means
    sst_jja = sst.sel(time=sst["time"].dt.month.isin([6, 7, 8]))
    sst_yearly = sst_jja.groupby("time.year").mean("time")

    # align with the t2m index years
    year_min, year_max = int(years[0]), int(years[-1])
    sst_yearly = sst_yearly.sel(year=slice(year_min, year_max)).compute()
    ds.close()

    # make sure year arrays match
    sst_years = sst_yearly["year"].values
    common_years = np.intersect1d(years, sst_years)
    idx_t2m = np.isin(years, common_years)
    idx_sst = np.isin(sst_years, common_years)

    x = index[idx_t2m].astype(np.float64)          # shape (n,)
    y = sst_yearly.values[idx_sst]                  # shape (n, nlat, nlon)

    n = len(x)
    x_mean = x.mean()
    x_c = x - x_mean
    ss_x = float((x_c ** 2).sum())

    y_2d = y.reshape(n, -1).astype(np.float64)      # (n, nlat*nlon)
    nan_col = np.any(~np.isfinite(y_2d), axis=0)

    slope_1d = np.full(y_2d.shape[1], np.nan)
    sig_1d   = np.zeros(y_2d.shape[1], dtype=bool)

    valid = ~nan_col
    if valid.any():
        y_v = y_2d[:, valid]
        y_mean = y_v.mean(axis=0)
        y_c = y_v - y_mean
        slope_v = (x_c @ y_c) / ss_x          # (nvalid,)

        # residuals and standard error of slope
        yhat   = x_c[:, None] * slope_v[None, :]
        resid  = y_c - yhat
        mse    = (resid ** 2).sum(axis=0) / (n - 2)
        se     = np.sqrt(mse / ss_x)
        t_stat = slope_v / np.where(se > 0, se, np.nan)
        p_val  = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)

        slope_1d[valid] = slope_v
        sig_1d[valid]   = p_val < 0.05

    nlat = sst_yearly.sizes["lat"]
    nlon = sst_yearly.sizes["lon"]
    slope   = slope_1d.reshape(nlat, nlon)
    sig_mask = sig_1d.reshape(nlat, nlon)

    lat_out = sst_yearly["lat"].values
    lon_out = sst_yearly["lon"].values

    # ensure ascending lat for plotting
    if lat_out[0] > lat_out[-1]:
        lat_out  = lat_out[::-1]
        slope    = slope[::-1, :]
        sig_mask = sig_mask[::-1, :]

    return slope, sig_mask, lon_out, lat_out


def draw_panel_g(fig, ax, cax, slope, sig_mask, lon, lat, map_proj):
    """Regression of JJA SST onto T2m index — geographic framework from
    plot_combined_3x2_anomalies.py panel (c)."""

    vmax = np.nanpercentile(np.abs(slope[np.isfinite(slope)]), 98) if np.any(np.isfinite(slope)) else 0.5
    vmax = max(float(np.ceil(vmax * 10) / 10), 0.2)
    levels = np.linspace(-vmax, vmax, 21)

    cf = ax.contourf(
        lon,
        lat,
        slope,
        levels=levels,
        cmap=cmaps.BlueWhiteOrangeRed,
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # significance hatching (dots) where p < 0.05
    sig_float = sig_mask.astype(float)
    ax.contourf(
        lon,
        lat,
        sig_float,
        levels=[0.5, 1.5],
        hatches=["..."],
        colors="none",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # land outlines
    add_shapefile(ax, WORLD_SHP, edgecolor="black", linewidth=0.7)

    # map extent and ticks — same region as reference panel c
    ax.set_extent([REG_LON_MIN, REG_LON_MAX, REG_LAT_MIN, REG_LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(REG_LON_MIN, REG_LON_MAX + 0.1, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(REG_LAT_MIN, REG_LAT_MAX + 0.1, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)

    ax.set_title(
        "(g) SST  onto T2m",
        loc="left",
        fontweight="bold",
        fontsize=23,
        pad=5,
    )

    cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
    cbar.set_label("°C", fontsize=12)
    cbar.ax.tick_params(labelsize=12)


# ======================== Main ========================
def main():
    print("Preparing data...")
    panel_a = load_v200_panel_a()
    jet_lon, jet_lat = load_jet_axis_panel_a()
    panel_e = load_plumb_panel_e()
    panel_b_anom, panel_b_stip, panel_b_lon, panel_b_lat = load_t2m_panel_b()
    years, t2m_index = load_t2m_index_panel_f()
    sst_slope, sst_sig, sst_lon, sst_lat = load_sst_regression_panel_g(years, t2m_index)
    panel_c_anom, panel_c_stip, panel_c_lon, panel_c_plev = load_cross_section(
        UVT_PATH,
        ["t_detrended", "t"],
        "temperature",
        p_top=200.0,
        divide_by_gravity=False,
    )
    panel_d_anom, panel_d_stip, panel_d_lon, panel_d_plev = load_cross_section(
        Z_PATH,
        ["z_detrended", "z"],
        "geopotential height",
        p_top=50.0,
        divide_by_gravity=True,
    )
    topo_lon, topo_pressure = load_topography_profile()

    print("Drawing figure...")
    fig = plt.figure(figsize=FIG_SIZE)
    map_proj = ccrs.PlateCarree(central_longitude=180)

    ax_a = fig.add_axes(AX_A, projection=map_proj)
    cax_a = fig.add_axes(CAX_A)

    ax_b = fig.add_axes(AX_B, projection=ccrs.PlateCarree())
    cax_b = fig.add_axes(CAX_B)

    ax_c = fig.add_axes(AX_C)
    cax_c = fig.add_axes(CAX_C)

    ax_d = fig.add_axes(AX_D)
    cax_d = fig.add_axes(CAX_D)

    ax_e = fig.add_axes(AX_E, projection=map_proj)
    cax_e = fig.add_axes(CAX_E)

    ax_f = fig.add_axes(AX_F)

    ax_g  = fig.add_axes(AX_G, projection=map_proj)
    cax_g = fig.add_axes(CAX_G)

    draw_panel_a(fig, ax_a, cax_a, panel_a, jet_lon, jet_lat)
    draw_panel_b(fig, ax_b, cax_b, panel_b_anom, panel_b_stip, panel_b_lon, panel_b_lat)
    draw_cross_section_panel(
        fig,
        ax_c,
        cax_c,
        panel_c_anom,
        panel_c_stip,
        panel_c_lon,
        panel_c_plev,
        topo_lon,
        topo_pressure,
        p_top=200.0,
        title=f"(c) T anomaly along {LAT_CROSS:.0f}°N",
        cbar_fmt="%.1f",
    )
    draw_cross_section_panel(
        fig,
        ax_d,
        cax_d,
        panel_d_anom,
        panel_d_stip,
        panel_d_lon,
        panel_d_plev,
        topo_lon,
        topo_pressure,
        p_top=50.0,
        title=f"(d) Z anomaly along {LAT_CROSS:.0f}°N",
        cbar_fmt="%d",
    )
    draw_panel_e(fig, ax_e, cax_e, panel_e)
    draw_panel_f(ax_f, years, t2m_index)
    draw_panel_g(fig, ax_g, cax_g, sst_slope, sst_sig, sst_lon, sst_lat, map_proj)

    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FIG}")

    panel_a.close()
    panel_e.close()


if __name__ == "__main__":
    main()
