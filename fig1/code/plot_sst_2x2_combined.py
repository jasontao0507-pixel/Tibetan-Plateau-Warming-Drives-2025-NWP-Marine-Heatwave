#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import cmaps

    MAP_CMAP = cmaps.BlueWhiteOrangeRed
except Exception:
    MAP_CMAP = "RdBu_r"

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# Paths and output
# ============================================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUTPUT_DIR, "sst_2x2_combined.png")
OUT_PANEL_C_CSV = os.path.join(OUTPUT_DIR, "sst_panel_c_raw_vs_detrended_1982_2025.csv")
OUT_PANEL_D_CSV = os.path.join(OUTPUT_DIR, "tp_heat_budget_2025_summer_terms.csv")

SHP_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/SHP/WORLD_SHP/continent.shp"

MAP_DAILY_SST_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/Ocean/sst_daily_1982_2026_detrended_signal.nc"
MAP_VAR_NAME = "sst_detrended"

PANEL_C_RAW_CANDIDATES = [
    "/gpfs/hpc/home/syxz/5250101004/Grayson/data/dSST.nc",
    "/gpfs/hpc/home/syxz/5250101004/Grayson/data/Ocean/SST/sst_daily_1982_2026.nc",
]
PANEL_C_RAW_VAR_CANDIDATES = ["sst", "sst"]

PANEL_C_DET_CANDIDATES = [
    "/gpfs/hpc/home/syxz/5250101004/Grayson/data/dSST_detrended_signal.nc",
    "/gpfs/hpc/home/syxz/5250101004/Grayson/data/Ocean/sst_daily_1982_2026_detrended_signal.nc",
]
PANEL_C_DET_VAR_CANDIDATES = ["sst_detrended", "sst_detrended"]

GODAS_DATA_DIR = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/Ocean"


# ============================================================================
# Figure config
# ============================================================================
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 13
ANNOT_FONTSIZE = 11
LEGEND_FONTSIZE = 10
TEXT_FONTSIZE = 10
PANEL_D_XTICK_FONTSIZE = 16


# ============================================================================
# Panel (a, b) config
# ============================================================================
MAP_LON_MIN = 90
MAP_LON_MAX = 180
MAP_LAT_MIN = 0
MAP_LAT_MAX = 80

BOX_LON_MIN = 130
BOX_LON_MAX = 175
BOX_LAT_MIN = 30
BOX_LAT_MAX = 45

MAP_ANOM_CLIM_START = 1982
MAP_ANOM_CLIM_END = 2023
TARGET_YEAR = 2025


# ============================================================================
# Panel (c) config
# ============================================================================
PANEL_C_LON_MIN = 120
PANEL_C_LON_MAX = 180
PANEL_C_LAT_MIN = 30
PANEL_C_LAT_MAX = 60
PANEL_C_YEAR_START = 1982
PANEL_C_YEAR_END = 2025
PANEL_C_BASELINE_START = 1982
PANEL_C_BASELINE_END = 2010


# ============================================================================
# Panel (d) config
# ============================================================================
RHO0 = 1015.0
CP = 4000.0
SEC_PER_MONTH = 30.44 * 86400

TP_LAT_S = 27.0
TP_LAT_N = 48.0
TP_LON_W = 127.0
TP_LON_E = 164.0
TP_PAD = 2.0

BOX_LON_MIN = TP_LON_W
BOX_LON_MAX = TP_LON_E
BOX_LAT_MIN = TP_LAT_S
BOX_LAT_MAX = TP_LAT_N

CLIM_START = 1982
CLIM_END = 2023


# ============================================================================
# Helpers
# ============================================================================
def _resolve_existing_path(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] Using {label}: {path}")
            return path
    raise FileNotFoundError(f"No existing file found for {label}: {candidates}")


def _lat_slice(lat_values, lat_min, lat_max):
    if float(lat_values[0]) > float(lat_values[-1]):
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def _add_land(ax):
    try:
        reader = shpreader.Reader(SHP_PATH)
        feature = ShapelyFeature(
            reader.geometries(),
            ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            linewidth=0.7,
            zorder=3,
        )
        ax.add_feature(feature)
    except Exception:
        ax.coastlines(linewidth=0.7)


def _format_map_axis(ax, draw_box=False):
    ax.set_extent([MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(MAP_LON_MIN, MAP_LON_MAX + 1, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(MAP_LAT_MIN, MAP_LAT_MAX + 1, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=TICK_FONTSIZE)
    if draw_box:
        box_lon = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
        box_lat = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
        ax.plot(
            box_lon,
            box_lat,
            color="black",
            linewidth=1.8,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )


def _seasonal_mean_by_year_map(path, var_name, lon_min, lon_max, lat_min, lat_max, year_start, year_end):
    fields = []
    with xr.open_dataset(path) as ds:
        lat_sel = _lat_slice(ds.lat.values, lat_min, lat_max)
        region = ds[var_name].sel(lat=lat_sel, lon=slice(lon_min, lon_max))
        for year in range(year_start, year_end + 1):
            da_year = region.sel(time=slice(f"{year}-06-01", f"{year}-08-31"))
            if da_year.sizes.get("time", 0) == 0:
                continue
            jja_field = da_year.mean("time").load()
            fields.append(jja_field.assign_coords(year=year).expand_dims("year"))
    if not fields:
        raise RuntimeError(f"No JJA map fields were computed from {path}")
    return xr.concat(fields, dim="year")


def _regional_jja_series(path, var_name, lon_min, lon_max, lat_min, lat_max, year_start, year_end):
    years = []
    values = []
    with xr.open_dataset(path) as ds:
        lat_sel = _lat_slice(ds.lat.values, lat_min, lat_max)
        region = ds[var_name].sel(lat=lat_sel, lon=slice(lon_min, lon_max))
        weights = np.cos(np.deg2rad(region.lat))
        for year in range(year_start, year_end + 1):
            da_year = region.sel(time=slice(f"{year}-06-01", f"{year}-08-31"))
            if da_year.sizes.get("time", 0) == 0:
                continue
            jja_mean = da_year.weighted(weights).mean(dim=("lat", "lon")).mean("time").load()
            years.append(year)
            values.append(float(jja_mean.values))
    return np.asarray(years, dtype=int), np.asarray(values, dtype=float)


def _rank_desc(values, target_idx):
    order = np.argsort(values)[::-1]
    rank = int(np.where(order == target_idx)[0][0]) + 1
    percentile = 100.0 * np.mean(values <= values[target_idx])
    return rank, percentile


def _load_yearly_subset(var_dir, file_prefix, years, var_name, lat_min, lat_max, lon_min, lon_max, level_slice=None):
    arrays = []
    for year in years:
        path = os.path.join(GODAS_DATA_DIR, var_dir, f"{file_prefix}.{year}.nc")
        print(f"    loading {path}")
        with xr.open_dataset(path) as ds:
            da = ds[var_name]
            if level_slice is not None:
                da = da.sel(level=level_slice)
            da = da.sel(
                lat=_lat_slice(da.lat.values, lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            ).load()
        arrays.append(da)
    return xr.concat(arrays, dim="time")


def _load_thflx_subset(lat_min, lat_max, lon_min, lon_max):
    path = os.path.join(GODAS_DATA_DIR, "thflx_1982_2025.nc")
    with xr.open_dataset(path) as ds:
        da = ds["thflx"].sel(
            time=slice(f"{CLIM_START}-01", f"{TARGET_YEAR}-12"),
            lat=_lat_slice(ds.lat.values, lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        ).load()
    return da


def compute_seasonal_anomaly(da, season="JJA"):
    seasonal = da.sel(time=da.time.dt.season == season)
    yearly = seasonal.groupby("time.year").mean("time")
    clim = yearly.sel(year=slice(CLIM_START, CLIM_END)).mean("year")
    return yearly - clim


def area_weighted_mean(da, lat_s, lat_n, lon_w, lon_e):
    region = da.sel(
        lat=_lat_slice(da.lat.values, lat_s, lat_n),
        lon=slice(lon_w, lon_e),
    )
    weights = np.cos(np.deg2rad(region.lat))
    return region.weighted(weights).mean(dim=["lat", "lon"])


def mld_average_temperature(pottmp, dbss):
    levels = pottmp.level.values
    bounds = np.zeros(len(levels) + 1)
    bounds[0] = 0.0
    for i in range(len(levels) - 1):
        bounds[i + 1] = 0.5 * (levels[i] + levels[i + 1])
    bounds[-1] = levels[-1] + (levels[-1] - bounds[-2])

    result = []
    for t in range(len(pottmp.time)):
        mld_t = dbss.isel(time=t)
        temp_t = pottmp.isel(time=t)
        weighted_sum = xr.zeros_like(mld_t)
        total_weight = xr.zeros_like(mld_t)

        for k in range(len(levels)):
            top_k = bounds[k]
            bot_k = bounds[k + 1]
            thickness = np.minimum(bot_k, mld_t) - top_k
            thickness = thickness.clip(min=0)
            weighted_sum = weighted_sum + temp_t.isel(level=k) * thickness
            total_weight = total_weight + thickness

        total_weight = total_weight.where(total_weight > 0, np.nan)
        result.append(weighted_sum / total_weight)

    mlt = xr.concat(result, dim="time")
    mlt["time"] = pottmp.time
    return mlt


def interp_uv_to_tgrid(ucur, vcur, t_lat, t_lon):
    u_interp = ucur.interp(lat=t_lat, lon=t_lon, method="linear")
    v_interp = vcur.interp(lat=t_lat, lon=t_lon, method="linear")
    return u_interp, v_interp


def compute_advection_term(temp_mla, ucur, vcur, dbss):
    lat_rad = np.deg2rad(temp_mla.lat)
    earth_radius = 6.371e6

    dTdx = temp_mla.differentiate("lon") / (earth_radius * np.cos(lat_rad) * np.pi / 180.0)
    dTdy = temp_mla.differentiate("lat") / (earth_radius * np.pi / 180.0)

    levels = ucur.level.values
    bounds = np.zeros(len(levels) + 1)
    bounds[0] = 0.0
    for i in range(len(levels) - 1):
        bounds[i + 1] = 0.5 * (levels[i] + levels[i + 1])
    bounds[-1] = levels[-1] + (levels[-1] - bounds[-2])

    u_mla_list = []
    v_mla_list = []
    for t in range(len(ucur.time)):
        mld_t = dbss.isel(time=t)
        u_t = ucur.isel(time=t)
        v_t = vcur.isel(time=t)
        u_sum = xr.zeros_like(mld_t)
        v_sum = xr.zeros_like(mld_t)
        total_weight = xr.zeros_like(mld_t)
        for k in range(len(levels)):
            thickness = (np.minimum(bounds[k + 1], mld_t) - bounds[k]).clip(min=0)
            u_sum = u_sum + u_t.isel(level=k) * thickness
            v_sum = v_sum + v_t.isel(level=k) * thickness
            total_weight = total_weight + thickness
        total_weight = total_weight.where(total_weight > 0, np.nan)
        u_mla_list.append(u_sum / total_weight)
        v_mla_list.append(v_sum / total_weight)

    u_mla = xr.concat(u_mla_list, dim="time")
    v_mla = xr.concat(v_mla_list, dim="time")
    u_mla["time"] = ucur.time
    v_mla["time"] = vcur.time

    return -(u_mla * dTdx + v_mla * dTdy)


# ============================================================================
# Panel calculations
# ============================================================================
def compute_map_panel_data():
    print("[1/5] Computing panels (a) and (b) map data...")
    jja_by_year = _seasonal_mean_by_year_map(
        MAP_DAILY_SST_PATH,
        MAP_VAR_NAME,
        MAP_LON_MIN,
        MAP_LON_MAX,
        MAP_LAT_MIN,
        MAP_LAT_MAX,
        MAP_ANOM_CLIM_START,
        TARGET_YEAR,
    )
    ref_years = jja_by_year.sel(year=slice(MAP_ANOM_CLIM_START, MAP_ANOM_CLIM_END))
    clim = ref_years.mean("year")
    anom_2025 = jja_by_year.sel(year=TARGET_YEAR) - clim
    std_map = jja_by_year.sel(year=slice(MAP_ANOM_CLIM_START, TARGET_YEAR)).std("year")
    return anom_2025.lon.values, anom_2025.lat.values, anom_2025.values, std_map.values


def compute_panel_c_data():
    print("[2/5] Computing panel (c) raw and detrended JJA series...")
    raw_path = _resolve_existing_path(PANEL_C_RAW_CANDIDATES, "panel (c) raw SST")
    det_path = _resolve_existing_path(PANEL_C_DET_CANDIDATES, "panel (c) detrended SST")

    raw_var = PANEL_C_RAW_VAR_CANDIDATES[PANEL_C_RAW_CANDIDATES.index(raw_path)]
    det_var = PANEL_C_DET_VAR_CANDIDATES[PANEL_C_DET_CANDIDATES.index(det_path)]

    years_det, jja_det = _regional_jja_series(
        det_path,
        det_var,
        PANEL_C_LON_MIN,
        PANEL_C_LON_MAX,
        PANEL_C_LAT_MIN,
        PANEL_C_LAT_MAX,
        PANEL_C_YEAR_START,
        PANEL_C_YEAR_END,
    )
    years_raw, jja_raw = _regional_jja_series(
        raw_path,
        raw_var,
        PANEL_C_LON_MIN,
        PANEL_C_LON_MAX,
        PANEL_C_LAT_MIN,
        PANEL_C_LAT_MAX,
        PANEL_C_YEAR_START,
        PANEL_C_YEAR_END,
    )

    common_years = np.intersect1d(years_det, years_raw)
    det_mask = np.isin(years_det, common_years)
    raw_mask = np.isin(years_raw, common_years)

    years = common_years.astype(int)
    jja_det = jja_det[det_mask]
    jja_raw = jja_raw[raw_mask]

    det_base = np.nanmean(jja_det[(years >= PANEL_C_BASELINE_START) & (years <= PANEL_C_BASELINE_END)])
    raw_base = np.nanmean(jja_raw[(years >= PANEL_C_BASELINE_START) & (years <= PANEL_C_BASELINE_END)])

    det_anom = jja_det - det_base
    raw_anom = jja_raw - raw_base

    idx_2025 = int(np.where(years == TARGET_YEAR)[0][0])
    rank_det, pct_det = _rank_desc(det_anom, idx_2025)
    rank_raw, pct_raw = _rank_desc(raw_anom, idx_2025)

    df = pd.DataFrame(
        {
            "year": years,
            "jja_mean_detrended_degC": jja_det,
            "jja_anomaly_detrended_degC": det_anom,
            "jja_mean_raw_degC": jja_raw,
            "jja_anomaly_raw_degC": raw_anom,
        }
    )
    df.to_csv(OUT_PANEL_C_CSV, index=False, float_format="%.6f")

    return {
        "years": years,
        "det_anom": det_anom,
        "raw_anom": raw_anom,
        "idx_2025": idx_2025,
        "rank_det": rank_det,
        "pct_det": pct_det,
        "rank_raw": rank_raw,
        "pct_raw": pct_raw,
    }


def compute_panel_d_budget():
    print("[3/5] Computing panel (d) TP heat budget...")
    years = list(range(CLIM_START, TARGET_YEAR + 1))
    lat_min = TP_LAT_S - TP_PAD
    lat_max = TP_LAT_N + TP_PAD
    lon_min = TP_LON_W - TP_PAD
    lon_max = TP_LON_E + TP_PAD

    print("  loading pottmp ...")
    pottmp = _load_yearly_subset(
        "pottmp",
        "pottmp",
        years,
        "pottmp",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        level_slice=slice(5, 115),
    )

    print("  loading dbss ...")
    dbss = _load_yearly_subset(
        "dbss",
        "dbss_obil",
        years,
        "dbss_obil",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    print("  loading thflx ...")
    thflx = _load_thflx_subset(lat_min, lat_max, lon_min, lon_max)

    print("  loading currents ...")
    ucur = _load_yearly_subset(
        "Ucur",
        "ucur",
        years,
        "ucur",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        level_slice=slice(5, 115),
    )
    vcur = _load_yearly_subset(
        "Vcur",
        "vcur",
        years,
        "vcur",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        level_slice=slice(5, 115),
    )

    print("  interpolating currents to T-grid ...")
    ucur_t, vcur_t = interp_uv_to_tgrid(ucur, vcur, pottmp.lat, pottmp.lon)

    print("  computing mixed-layer mean temperature ...")
    mlt = mld_average_temperature(pottmp, dbss) - 273.15
    mlt_anom = compute_seasonal_anomaly(mlt, season="JJA")
    dTdt_val = float(area_weighted_mean(mlt_anom.sel(year=TARGET_YEAR), TP_LAT_S, TP_LAT_N, TP_LON_W, TP_LON_E))

    print("  computing Qnet term ...")
    H = dbss.where(dbss > 1.0, 1.0)
    qnet_term = -thflx / (RHO0 * CP * H) * SEC_PER_MONTH * 3.0
    qnet_anom = compute_seasonal_anomaly(qnet_term, season="JJA")
    qnet_val = float(area_weighted_mean(qnet_anom.sel(year=TARGET_YEAR), TP_LAT_S, TP_LAT_N, TP_LON_W, TP_LON_E))

    print("  computing advection term ...")
    adv = compute_advection_term(mlt, ucur_t, vcur_t, dbss) * SEC_PER_MONTH * 3.0
    adv_anom = compute_seasonal_anomaly(adv, season="JJA")
    adv_val = float(area_weighted_mean(adv_anom.sel(year=TARGET_YEAR), TP_LAT_S, TP_LAT_N, TP_LON_W, TP_LON_E))

    resid_val = dTdt_val - qnet_val - adv_val

    df = pd.DataFrame(
        [
            {
                "term": "Delta_SSTa",
                "value_degC_per_season": dTdt_val,
            },
            {
                "term": "Qnet_term",
                "value_degC_per_season": qnet_val,
            },
            {
                "term": "Advection_term",
                "value_degC_per_season": adv_val,
            },
            {
                "term": "Residual",
                "value_degC_per_season": resid_val,
            },
        ]
    )
    df.to_csv(OUT_PANEL_D_CSV, index=False, float_format="%.6f")

    return {
        "values": [dTdt_val, qnet_val, adv_val, resid_val],
        "labels": [
            r"$\Delta \mathrm{SST}_a$",
            r"$\left(\frac{Q_{\mathrm{net}}}{\rho_0 C_p H}\right)_a$",
            r"$-(\mathbf{V}\cdot\nabla T)_a$",
            "Residual",
        ],
    }


# ============================================================================
# Plotting
# ============================================================================
def make_figure(lon_map, lat_map, data_map, std_map, panel_c, panel_d):
    print("[4/5] Plotting 2x2 figure...")
    fig = plt.figure(figsize=(16.5, 14.0))
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.0)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.075)

    proj = ccrs.PlateCarree(central_longitude=180)
    map_levels = np.arange(-2.0, 2.5, 0.5)
    std_levels = np.arange(0.6, 1.81, 0.1)
    std_ticks = np.arange(0.6, 1.81, 0.2)
    panel_box_aspect = (MAP_LAT_MAX - MAP_LAT_MIN) / (MAP_LON_MAX - MAP_LON_MIN)

    ax_a = fig.add_subplot(gs[0, 0], projection=proj)
    cf_a = ax_a.contourf(
        lon_map,
        lat_map,
        data_map,
        levels=map_levels,
        cmap=MAP_CMAP,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    _add_land(ax_a)
    _format_map_axis(ax_a, draw_box=True)
    ax_a.set_box_aspect(panel_box_aspect)
    ax_a.set_title(
        "(a) SSTA, 2025, JJA",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax_a.text(
        0.98,
        1.01,
        "Unit: degC",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    cax_a = inset_axes(
        ax_a,
        width="60%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
        bbox_transform=ax_a.transAxes,
        borderpad=0.0,
    )
    cbar_a = fig.colorbar(cf_a, cax=cax_a, orientation="horizontal", ticks=map_levels)
    cbar_a.ax.tick_params(labelsize=TICK_FONTSIZE)

    ax_b = fig.add_subplot(gs[0, 1], projection=proj)
    cf_b = ax_b.contourf(
        lon_map,
        lat_map,
        std_map,
        levels=std_levels,
        cmap="YlOrRd",
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    _add_land(ax_b)
    _format_map_axis(ax_b, draw_box=False)
    ax_b.set_box_aspect(panel_box_aspect)
    ax_b.set_title(
        "(b) STDA, 1982-2025",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax_b.text(
        0.98,
        1.01,
        "Unit: degC",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    cax_b = inset_axes(
        ax_b,
        width="60%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
        bbox_transform=ax_b.transAxes,
        borderpad=0.0,
    )
    cbar_b = fig.colorbar(cf_b, cax=cax_b, orientation="horizontal", ticks=std_ticks)
    cbar_b.ax.tick_params(labelsize=TICK_FONTSIZE)

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_box_aspect(panel_box_aspect)
    years = panel_c["years"]
    det_anom = panel_c["det_anom"]
    raw_anom = panel_c["raw_anom"]
    idx_2025 = panel_c["idx_2025"]

    ax_c.bar(
        years,
        det_anom,
        width=0.78,
        facecolor="none",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
    )

    ax_c.vlines(years, 0.0, raw_anom, colors="#ff69b4", linewidth=2.0, zorder=5)

    ax_c.axhline(0.0, color="black", linewidth=1.0, zorder=4)
    ax_c.set_xlim(PANEL_C_YEAR_START - 0.8, PANEL_C_YEAR_END + 0.8)
    xticks_c = list(np.arange(PANEL_C_YEAR_START, PANEL_C_YEAR_END, 6))
    if PANEL_C_YEAR_END - 1 in xticks_c:
        xticks_c.remove(PANEL_C_YEAR_END - 1)
    if PANEL_C_YEAR_END not in xticks_c:
        xticks_c.append(PANEL_C_YEAR_END)
    ax_c.set_xticks(xticks_c)
    ax_c.tick_params(labelsize=TICK_FONTSIZE)
    ax_c.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax_c.set_title(
        "(c) SSTA, Raw & De-trended",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    y_all = np.concatenate([det_anom, raw_anom])
    y_pad = max(0.10, 0.12 * (np.nanmax(y_all) - np.nanmin(y_all)))
    ax_c.set_ylim(np.nanmin(y_all) - y_pad, np.nanmax(y_all) + y_pad)
    ax_c.text(
        0.98,
        0.06,
        (
            f"2025 JJA: Raw={raw_anom[idx_2025]:.2f} " + "\N{DEGREE SIGN}C"
            + f", De-trended={det_anom[idx_2025]:.2f} " + "\N{DEGREE SIGN}C"
        ),
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=TEXT_FONTSIZE + 2,
        color="black",
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor="none", edgecolor="black", linewidth=1.4, label="Detrended SST"),
        Line2D([0], [0], color="#ff69b4", linewidth=2.0, label="Raw SST"),
    ]
    ax_c.legend(handles=legend_handles, loc="upper left", fontsize=LEGEND_FONTSIZE + 4, frameon=False)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_box_aspect(panel_box_aspect)
    x = np.arange(len(panel_d["values"]))
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]
    bars = ax_d.bar(x, panel_d["values"], color=colors, width=0.62, edgecolor="black", linewidth=0.8, zorder=3)
    ax_d.axhline(0.0, color="black", linewidth=0.8, zorder=4)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(panel_d["labels"], fontsize=PANEL_D_XTICK_FONTSIZE)
    ax_d.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_d.grid(axis="y", alpha=0.3, linestyle=":", zorder=0)
    ax_d.set_title(
        "(d) Mixed-layer heat budget",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    for spine in ax_d.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    vals = np.asarray(panel_d["values"], dtype=float)
    y_pad_d = max(0.15, 0.15 * (np.nanmax(vals) - np.nanmin(vals)))
    ax_d.set_ylim(np.nanmin(vals) - y_pad_d, np.nanmax(vals) + y_pad_d)

    for bar, val in zip(bars, vals):
        y = val if val >= 0 else val
        va = "bottom" if val >= 0 else "top"
        ax_d.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{val:+.2f}",
            ha="center",
            va=va,
            fontsize=ANNOT_FONTSIZE,
            fontweight="bold",
        )

    for tick in ax_c.get_xticklabels():
        if tick.get_text() == str(PANEL_C_YEAR_END):
            tick.set_color("red")
            tick.set_fontweight("bold")

    fig.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lon_map, lat_map, data_map, std_map = compute_map_panel_data()
    panel_c = compute_panel_c_data()
    panel_d = compute_panel_d_budget()
    make_figure(lon_map, lat_map, data_map, std_map, panel_c, panel_d)

    print("[5/5] Done.")
    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved data  : {OUT_PANEL_C_CSV}")
    print(f"Saved data  : {OUT_PANEL_D_CSV}")


if __name__ == "__main__":
    main()
