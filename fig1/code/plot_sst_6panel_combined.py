#!/usr/bin/env python3

import importlib.util
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import marineHeatWaves as mhw
import cartopy.crs as ccrs
import cmaps

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Nimbus Roman"
plt.rcParams["mathtext.it"] = "Nimbus Roman:italic"
plt.rcParams["mathtext.bf"] = "Nimbus Roman:bold"
plt.rcParams["mathtext.cal"] = "Nimbus Roman:italic"
plt.rcParams["axes.unicode_minus"] = False

ROMAN_TEXT_PROP = FontProperties(family=["Nimbus Roman"], weight="bold")
CELSIUS_TEXT_PROP = FontProperties(family=["DejaVu Serif"], weight="bold")


# ============================================================================
# Paths / outputs
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "out")

OUT_PNG = os.path.join(OUT_DIR, "sst_6panel_combined_new.png")
OUT_PANEL_D_CSV = os.path.join(OUT_DIR, "sst_panel_d_raw_vs_detrended_1982_2025.csv")
OUT_PANEL_F_CSV = os.path.join(OUT_DIR, "tp_qnet_component_contribution_2025.csv")

BASE_SCRIPT_PATH = (
    "/gpfs/hpc/home/syxz/5250101004/JasonTao/codex/202603/20260328/plot_sst_2x2_combined.py"
)
SST_DSTYLE_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/dSST_detrended_signal.nc"
ERA5_QNET_RAW_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/Qnet_new/Qnet_1982_2025_summer.nc"
ERA5_QNET_DET_PATH = "/gpfs/hpc/home/syxz/5250101004/Grayson/data/ERA5/Qnet_new/Qnet_1982_2025_summer_detrended.nc"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("plot_sst_2x2_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base module from: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()
base.OUTPUT_DIR = OUT_DIR
base.OUT_PNG = os.path.join(OUT_DIR, "sst_2x2_combined_base.png")
base.OUT_PANEL_C_CSV = OUT_PANEL_D_CSV
base.OUT_PANEL_D_CSV = os.path.join(OUT_DIR, "tp_heat_budget_2025_summer_terms.csv")


# ============================================================================
# Config
# ============================================================================
TARGET_YEAR = 2025
MHW_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 21
TICK_FONTSIZE = 21
LEGEND_FONTSIZE = 20
ANNOT_FONTSIZE = 19
TEXT_FONTSIZE = 19
COLORBAR_TICK_FONTSIZE = 18
PANEL_E_XTICK_FONTSIZE = 22
PANEL_F_XTICK_FONTSIZE = 21

SEC_PER_SEASON = base.SEC_PER_MONTH * 3.0

PLOT_MAP_LON_MIN = 90.0
PLOT_MAP_LON_MAX = 180.0
PLOT_MAP_LAT_MIN = 0.0
PLOT_MAP_LAT_MAX = 60.0

BOX_LON_MIN = base.BOX_LON_MIN
BOX_LON_MAX = 176.0
BOX_LAT_MIN = base.BOX_LAT_MIN
BOX_LAT_MAX = 55.0


# ============================================================================
# Panel (b): 2025 D-style time series
# ============================================================================
def compute_mhw_series_tp():
    print("Computing panel (b) TP-region D-style SST series...")
    with xr.open_dataset(SST_DSTYLE_PATH) as ds:
        lat_slice = base._lat_slice(ds.lat.values, base.TP_LAT_S, base.TP_LAT_N)
        sst = ds["sst_detrended"].sel(
            time=slice("1982-01-01", f"{TARGET_YEAR}-12-31"),
            lat=lat_slice,
            lon=slice(base.TP_LON_W, base.TP_LON_E),
        )
        sst_ts = sst.mean(dim=["lat", "lon"])
        sst_values = sst_ts.values.astype(float)

    sst_mean = sst_values.mean()
    sst_detrend = signal.detrend(sst_values, axis=0, type="linear")
    sst_final = sst_detrend + sst_mean

    t_arr = np.array([date.fromisoformat(str(d)[:10]).toordinal() for d in sst_ts.time.values], dtype=int)
    dates = np.array([date.fromordinal(int(tt)) for tt in t_arr], dtype=object)
    mhws, clim = mhw.detect(t_arr, sst_final)
    return t_arr, dates, sst_final, mhws, clim


def overlapping_events(mhws, start_d, end_d):
    ids = []
    for i in range(len(mhws["time_start"])):
        start_evt = mhws["date_start"][i]
        end_evt = mhws["date_end"][i]
        if end_evt < start_d or start_evt > end_d:
            continue
        ids.append(i)
    return ids


def compute_common_ylim(dates, sst_vals, clim, years):
    vals = []
    last_day = dates[-1]
    for year in years:
        win_start = date(year, 6, 1)
        win_end = min(date(year, 12, 31), last_day)
        mask = np.array([(d >= win_start) and (d <= win_end) for d in dates], dtype=bool)
        if not np.any(mask):
            continue
        vals.extend(np.asarray(sst_vals)[mask].tolist())
        vals.extend(np.asarray(clim["thresh"])[mask].tolist())
        vals.extend(np.asarray(clim["seas"])[mask].tolist())

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (10.0, 22.0)

    y_min = np.floor((np.nanmin(vals) - 0.4) * 2.0) / 2.0
    y_max = np.ceil((np.nanmax(vals) + 0.4) * 2.0) / 2.0
    if (y_max - y_min) < 4.0:
        center = 0.5 * (y_min + y_max)
        y_min = center - 2.0
        y_max = center + 2.0
    return (float(y_min), float(y_max))


def draw_year_panel(ax, year, t_arr, dates, sst_vals, mhws, clim, ylim):
    win_start = date(year, 6, 1)
    jja_end = min(date(year, 8, 31), dates[-1])
    y0, y1 = float(ylim[0]), float(ylim[1])

    idx_map = {int(v): i for i, v in enumerate(t_arr)}
    ev_ids = overlapping_events(mhws, win_start, jja_end)
    main_ev = max(ev_ids, key=lambda i: float(mhws["intensity_max"][i])) if ev_ids else None
    date_arr = np.array(dates)
    win_mask = np.array([(d >= win_start) and (d <= jja_end) for d in date_arr], dtype=bool)
    plot_end = jja_end

    for ev in ev_ids:
        t1 = idx_map[int(mhws["time_start"][ev])]
        t2 = idx_map[int(mhws["time_end"][ev])]
        if ev == main_ev:
            ax.fill_between(
                dates[t1 : t2 + 1],
                sst_vals[t1 : t2 + 1],
                clim["thresh"][t1 : t2 + 1],
                color="r",
                alpha=0.78,
            )
        else:
            ax.fill_between(
                dates[t1 : t2 + 1],
                sst_vals[t1 : t2 + 1],
                clim["thresh"][t1 : t2 + 1],
                color=(1, 0.6, 0.5),
                alpha=0.45,
            )

    ax.plot(dates, sst_vals, "k-", linewidth=1.2, label="SST")
    ax.plot(dates, clim["thresh"], "g-", linewidth=1.1, label="90th Threshold")
    ax.plot(dates, clim["seas"], "b-", linewidth=1.1, label="Climatology")

    if main_ev is not None:
        start_date = mhws["date_start"][main_ev]
        end_date = mhws["date_end"][main_ev]
        plot_end = min(max(jja_end, end_date), dates[-1])
        i_start = np.where(date_arr == start_date)[0]
        peak_dates = date_arr[win_mask]
        peak_vals = np.asarray(sst_vals)[win_mask]
        peak_date = peak_dates[int(np.nanargmax(peak_vals))]
        i_peak = np.where(date_arr == peak_date)[0]
        sep_mask = np.array([(d >= date(year, 9, 1)) and (d <= plot_end) for d in date_arr], dtype=bool)
        if np.any(sep_mask):
            sep_dates = date_arr[sep_mask]
            sep_diff = np.abs(np.asarray(sst_vals)[sep_mask] - np.asarray(clim["thresh"])[sep_mask])
            end_marker_date = sep_dates[int(np.nanargmin(sep_diff))]
        else:
            end_marker_date = end_date
        i_end = np.where(date_arr == end_marker_date)[0]

        y_start = float(sst_vals[i_start[0]]) if i_start.size > 0 else np.nan
        y_end = float(sst_vals[i_end[0]]) if i_end.size > 0 else np.nan
        y_peak = float(sst_vals[i_peak[0]]) if i_peak.size > 0 else np.nan

        if np.isfinite(y_start) and win_start <= start_date <= plot_end:
            ax.vlines(start_date, y0, y_start, color="k", linestyle="--", linewidth=1.0)
            ax.text(
                start_date,
                min(y_start + 0.25, y1 - 0.25),
                start_date.strftime("%m-%d"),
                color="k",
                fontsize=ANNOT_FONTSIZE,
                ha="right",
                va="bottom",
                fontweight="bold",
            )
        if np.isfinite(y_end) and win_start <= end_marker_date <= plot_end:
            ax.vlines(end_marker_date, y0, y_end, color="k", linestyle="--", linewidth=1.0)
            ax.text(
                end_marker_date,
                min(y_end + 0.25, y1 - 0.25),
                end_marker_date.strftime("%m-%d"),
                color="k",
                fontsize=ANNOT_FONTSIZE,
                ha="left",
                va="bottom",
                fontweight="bold",
            )
        if np.isfinite(y_peak):
            ax.vlines(peak_date, y0, y_peak, color="r", linestyle="--", linewidth=1.0)
            ax.text(
                peak_date,
                min(y_peak + 0.25, y1 - 0.25),
                peak_date.strftime("%m-%d"),
                color="r",
                fontsize=ANNOT_FONTSIZE,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xlim(win_start, plot_end)
    ax.set_ylim(y0, y1)
    ax.set_title("(b) 2025 JJA SST Time Series", loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False)


def format_map_axis(ax, draw_box):
    ax.set_extent([PLOT_MAP_LON_MIN, PLOT_MAP_LON_MAX, PLOT_MAP_LAT_MIN, PLOT_MAP_LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(PLOT_MAP_LON_MIN, PLOT_MAP_LON_MAX + 1, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(PLOT_MAP_LAT_MIN, PLOT_MAP_LAT_MAX + 1, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(base.cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(base.cticker.LatitudeFormatter())
    ax.tick_params(labelsize=TICK_FONTSIZE)
    if draw_box:
        box_lon = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
        box_lat = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
        ax.plot(
            box_lon,
            box_lat,
            color="green",
            linewidth=2.8,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )


def add_panel_d_summary(ax, raw_val, det_val):
    text_size = TEXT_FONTSIZE + 1
    pieces = [
        TextArea(
            f"2025 JJA: Raw={raw_val:.2f} ",
            textprops={"color": "black", "fontproperties": ROMAN_TEXT_PROP.copy(), "fontsize": text_size},
        ),

        TextArea(
            f", De-trended={det_val:.2f} ",
            textprops={"color": "black", "fontproperties": ROMAN_TEXT_PROP.copy(), "fontsize": text_size},
        ),

    ]
    packed = HPacker(children=pieces, align="baseline", pad=0, sep=0)
    box = AnnotationBbox(
        packed,
        (0.98, 0.06),
        xycoords=ax.transAxes,
        box_alignment=(1.0, 0.0),
        frameon=False,
        pad=0.0,
    )
    ax.add_artist(box)


# ============================================================================
# Panel (f): ERA5 Qnet decomposition
# ============================================================================
def _load_dbss_jja_region(years, lat_min, lat_max, lon_min, lon_max):
    arrays = []
    for year in years:
        path = os.path.join(base.GODAS_DATA_DIR, "dbss", f"dbss_obil.{year}.nc")
        print(f"    loading {path}")
        with xr.open_dataset(path) as ds:
            da = ds["dbss_obil"].sel(
                time=slice(f"{year}-06-01", f"{year}-08-31"),
                lat=base._lat_slice(ds.lat.values, lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            ).load()
        arrays.append(da)
    return xr.concat(arrays, dim="time")


def compute_panel_f_qnet_components():
    print("Computing panel (f) ERA5 Qnet decomposition...")
    years = list(range(base.CLIM_START, TARGET_YEAR + 1))

    with xr.open_dataset(ERA5_QNET_RAW_PATH) as ds:
        era = ds.sel(
            valid_time=slice(f"{base.CLIM_START}-06-01", f"{TARGET_YEAR}-08-31"),
            latitude=slice(base.TP_LAT_N, base.TP_LAT_S),
            longitude=slice(base.TP_LON_W, base.TP_LON_E),
        ).load()

    dbss = _load_dbss_jja_region(years, base.TP_LAT_S - 2.0, base.TP_LAT_N + 2.0, base.TP_LON_W - 2.0, base.TP_LON_E + 2.0)
    dbss = dbss.rename({"lat": "latitude", "lon": "longitude"})
    dbss_interp = dbss.interp(latitude=era.latitude, longitude=era.longitude)
    dbss_interp = dbss_interp.assign_coords(time=("time", era.valid_time.values))
    H = xr.where(
        np.isfinite(dbss_interp),
        xr.where(dbss_interp > 1.0, dbss_interp, 1.0),
        np.nan,
    ).rename({"time": "valid_time"})

    component_map = {
        "Shortwave": "avg_snswrf",
        "Longwave": "avg_snlwrf",
        "Sensible": "avg_ishf",
        "Latent": "avg_slhtf",
    }

    values = []
    rows = []
    for label, var_name in component_map.items():
        term = era[var_name] / (base.RHO0 * base.CP * H) * SEC_PER_SEASON
        yearly = term.groupby("valid_time.year").mean("valid_time")
        clim = yearly.sel(year=slice(base.CLIM_START, base.CLIM_END)).mean("year")
        anom = yearly - clim
        region_2025 = anom.sel(year=TARGET_YEAR)
        weights = np.cos(np.deg2rad(region_2025.latitude))
        val = float(region_2025.weighted(weights).mean(("latitude", "longitude")))
        values.append(val)
        rows.append({"component": label, "value_degC_per_season": val, "source_data": "raw_ERA5_Qnet"})

    pd.DataFrame(rows).to_csv(OUT_PANEL_F_CSV, index=False, float_format="%.6f")
    return {
        "labels": ["SW", "LW", "SH", "LH"],
        "values": values,
        "title": "(f) Surface heat flux contributions",
    }


# ============================================================================
# Plotting
# ============================================================================
def make_figure(map_data, panel_b_data, panel_d_data, panel_e_data, panel_f_data):
    print("Plotting 3x2 figure...")
    lon_map, lat_map, anom_map, std_map = map_data
    years_b, dates_b, sst_b, mhws_b, clim_b, ylim_b = panel_b_data

    fig = plt.figure(figsize=(18, 20.25))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.16)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.05)

    proj = ccrs.PlateCarree(central_longitude=180)
    map_levels = np.arange(-2.0, 2.5, 0.5)
    std_levels = np.arange(0.5, 1.51, 0.1)
    std_ticks = np.arange(0.5, 1.51, 0.2)
    map_panel_aspect = (PLOT_MAP_LAT_MAX - PLOT_MAP_LAT_MIN) / (PLOT_MAP_LON_MAX - PLOT_MAP_LON_MIN)

    ax_a = fig.add_subplot(gs[0, 0], projection=proj)
    cf_a = ax_a.contourf(
        lon_map,
        lat_map,
        anom_map,
        levels=map_levels,
        cmap=base.MAP_CMAP,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    base._add_land(ax_a)
    format_map_axis(ax_a, draw_box=True)
    ax_a.set_box_aspect(map_panel_aspect)
    ax_a.set_title("(a) SSTA, 2025, JJA", loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    cax_a = inset_axes(
        ax_a,
        width="58%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.13, 1.0, 1.0),
        bbox_transform=ax_a.transAxes,
        borderpad=0.0,
    )
    cbar_a = fig.colorbar(cf_a, cax=cax_a, orientation="horizontal", ticks=map_levels)
    cbar_a.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_year_panel(ax_b, TARGET_YEAR, years_b, dates_b, sst_b, mhws_b, clim_b, ylim_b)

    ax_c = fig.add_subplot(gs[1, 0], projection=proj)
    cf_c = ax_c.contourf(
        lon_map,
        lat_map,
        std_map,
        levels=std_levels,
        cmap=cmaps.sunshine_9lev,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    base._add_land(ax_c)
    format_map_axis(ax_c, draw_box=False)
    ax_c.set_box_aspect(map_panel_aspect)
    ax_c.set_title("(c) STDA, 1982-2025", loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    cax_c = inset_axes(
        ax_c,
        width="58%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.13, 1.0, 1.0),
        bbox_transform=ax_c.transAxes,
        borderpad=0.0,
    )
    cbar_c = fig.colorbar(cf_c, cax=cax_c, orientation="horizontal", ticks=std_ticks)
    cbar_c.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    ax_d = fig.add_subplot(gs[1, 1])
    years = panel_d_data["years"]
    det_anom = panel_d_data["det_anom"]
    raw_anom = panel_d_data["raw_anom"]
    idx_2025 = panel_d_data["idx_2025"]
    ax_d.bar(years, det_anom, width=0.78, facecolor="none", edgecolor="black", linewidth=1.4, zorder=3)
    ax_d.vlines(years, 0.0, raw_anom, colors="#ff69b4", linewidth=2.0, zorder=5)
    ax_d.axhline(0.0, color="black", linewidth=1.0, zorder=4)
    ax_d.set_xlim(base.PANEL_C_YEAR_START - 0.8, base.PANEL_C_YEAR_END + 0.8)
    xticks_d = list(np.arange(base.PANEL_C_YEAR_START, base.PANEL_C_YEAR_END, 6))
    if base.PANEL_C_YEAR_END - 1 in xticks_d:
        xticks_d.remove(base.PANEL_C_YEAR_END - 1)
    if base.PANEL_C_YEAR_END not in xticks_d:
        xticks_d.append(base.PANEL_C_YEAR_END)
    ax_d.set_xticks(xticks_d)
    ax_d.tick_params(labelsize=TICK_FONTSIZE)
    ax_d.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax_d.set_title("(d) SSTA, Raw & De-trended", loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    y_all = np.concatenate([det_anom, raw_anom])
    y_pad = max(0.10, 0.12 * (np.nanmax(y_all) - np.nanmin(y_all)))
    ax_d.set_ylim(np.nanmin(y_all) - y_pad, np.nanmax(y_all) + y_pad)
    add_panel_d_summary(ax_d, raw_anom[idx_2025], det_anom[idx_2025])
    legend_handles_d = [
        Patch(facecolor="none", edgecolor="black", linewidth=1.4, label="Detrended SST"),
        Line2D([0], [0], color="#ff69b4", linewidth=2.0, label="Raw SST"),
    ]
    ax_d.legend(handles=legend_handles_d, loc="upper left", fontsize=LEGEND_FONTSIZE + 2, frameon=False)

    ax_e = fig.add_subplot(gs[2, 0])
    x_e = np.arange(len(panel_e_data["values"]))
    colors_e = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]
    bars_e = ax_e.bar(x_e, panel_e_data["values"], color=colors_e, width=0.62, edgecolor="black", linewidth=0.8, zorder=3)
    ax_e.axhline(0.0, color="black", linewidth=0.8, zorder=4)
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(panel_e_data["labels"], fontsize=PANEL_E_XTICK_FONTSIZE)
    ax_e.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_e.grid(axis="y", alpha=0.3, linestyle=":", zorder=0)
    ax_e.set_title("(e) Mixed-layer heat budget", loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    for spine in ax_e.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    vals_e = np.asarray(panel_e_data["values"], dtype=float)
    y_pad_e = max(0.15, 0.15 * (np.nanmax(vals_e) - np.nanmin(vals_e)))
    ax_e.set_ylim(np.nanmin(vals_e) - y_pad_e, np.nanmax(vals_e) + y_pad_e)
    for bar, val in zip(bars_e, vals_e):
        ax_e.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=ANNOT_FONTSIZE,
            fontweight="bold",
        )

    ax_f = fig.add_subplot(gs[2, 1])
    x_f = np.arange(len(panel_f_data["values"]))
    colors_f = ["#f1a340", "#998ec3", "#80cdc1", "#ef8a62"]
    bars_f = ax_f.bar(x_f, panel_f_data["values"], color=colors_f, width=0.62, edgecolor="black", linewidth=0.8, zorder=3)
    ax_f.axhline(0.0, color="black", linewidth=0.8, zorder=4)
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(panel_f_data["labels"], fontsize=PANEL_F_XTICK_FONTSIZE)
    ax_f.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_f.grid(axis="y", alpha=0.3, linestyle=":", zorder=0)
    ax_f.set_title(panel_f_data["title"], loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")
    vals_f = np.asarray(panel_f_data["values"], dtype=float)
    y_pad_f = max(0.08, 0.18 * (np.nanmax(vals_f) - np.nanmin(vals_f)))
    ax_f.set_ylim(np.nanmin(vals_f) - y_pad_f, np.nanmax(vals_f) + y_pad_f)
    for bar, val in zip(bars_f, vals_f):
        ax_f.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=ANNOT_FONTSIZE,
            fontweight="bold",
        )

    fig.savefig(OUT_PNG, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing map panels and panel (d) bar series from the base workflow...")
    map_data = base.compute_map_panel_data()
    panel_d_data = base.compute_panel_c_data()
    panel_e_data = base.compute_panel_d_budget()

    t_arr, dates, sst_vals, mhws, clim = compute_mhw_series_tp()
    ylim = compute_common_ylim(dates, sst_vals, clim, MHW_YEARS)
    panel_b_data = (t_arr, dates, sst_vals, mhws, clim, ylim)

    panel_f_data = compute_panel_f_qnet_components()

    make_figure(map_data, panel_b_data, panel_d_data, panel_e_data, panel_f_data)

    print("Done.")
    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved data  : {OUT_PANEL_D_CSV}")
    print(f"Saved data  : {OUT_PANEL_F_CSV}")


if __name__ == "__main__":
    main()
