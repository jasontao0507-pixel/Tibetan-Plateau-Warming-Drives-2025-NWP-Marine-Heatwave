#!/usr/bin/env python3
"""
Figure 4 style plot using Tibet LBM experiment data.
6 panels: (a) horizontal heating, (b) vertical profile,
          (c) Z cross-section TP, (d) T cross-section TP,
          (e) Z cross-section extended, (f) 200hPa v-wind map
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
import cmaps
warnings.filterwarnings('ignore')

# ======================== Global font sizes ========================
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})

# ======================== File paths ========================
force_file = '/Users/jasontao/mywork/LBM/ln_solver/data/tintgr/tibet_heat.frc.t21l20.nc'
res_file   = '/Users/jasontao/mywork/LBM/ln_solver/data/tintgr/tibet_heat.t21l20.nc'
tp_shp     = '/Users/jasontao/mywork/data/SHP/TP_SHP/TPBoundary_new(2021)/TPBoundary_new(2021).shp'
cont_shp   = '/Users/jasontao/mywork/data/SHP/WORLD_SHP/continent.shp'
geo_file   = '/Users/jasontao/mywork/data/SHP/dixing/geo.nc'
output_fig = '/Users/jasontao/mywork/ex/fig4_tibet_lbm_36N.png'

# ======================== Load data ========================
ds_f = xr.open_dataset(force_file, decode_times=False)
ds_r = xr.open_dataset(res_file,   decode_times=False)
ds_g = xr.open_dataset(geo_file)

# --- Forcing ---
lon_f  = ds_f['lon'].values
lat_f  = ds_f['lat'].values
sigma  = ds_f['lev'].values
t_frc  = ds_f['t'].isel(time=0).values.copy()   # (lev, lat, lon)
t_frc[t_frc == -999] = np.nan
t_kday = t_frc * 86400.0                         # K/s -> K/day

# --- Response (steady state: mean of last 10 days) ---
lon_r = ds_r['lon'].values
lat_r = ds_r['lat'].values
plev  = ds_r['lev'].values                       # hPa
# Taking mean over the last 10 days (assuming 30 days total, index 20 to 29)
z_r   = ds_r['z'].isel(time=slice(-10, None)).mean(dim='time').values.copy()    # geopotential height (m)
t_r   = ds_r['t'].isel(time=slice(-10, None)).mean(dim='time').values.copy()    # temperature (K)
v_r   = ds_r['v'].isel(time=slice(-10, None)).mean(dim='time').values.copy()    # meridional wind (m/s)
u_r   = ds_r['u'].isel(time=slice(-10, None)).mean(dim='time').values.copy()    # zonal wind (m/s)
for a in [z_r, t_r, v_r, u_r]:
    a[a == -999] = np.nan

# --- Topography ---
geo_lon = ds_g['longitude'].values
geo_lat = ds_g['latitude'].values
topo_m  = ds_g['z'].isel(time=0).values / 9.80665   # geopotential -> meters

# ======================== Forcing center ========================
idx3d   = np.unravel_index(np.nanargmax(t_kday), t_kday.shape)
ci_lev, ci_lat, ci_lon = idx3d
ctr_lat = lat_f[ci_lat]
ctr_lon = lon_f[ci_lon]
print(f"Forcing center : {ctr_lat:.1f}°N, {ctr_lon:.1f}°E")
print(f"Max heating    : {np.nanmax(t_kday):.2f} K/day  (sigma={sigma[ci_lev]:.3f})")

# ======================== Helpers ========================
def add_shp(ax, path, ec='k', fc='none', lw=1.0, ls='-'):
    for geom in Reader(path).geometries():
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)

def topo_to_pressure(heights):
    """Convert terrain height (m) to approximate pressure (hPa)."""
    h = np.maximum(heights, 0)
    return 1013.25 * np.exp(-h / 7500.0)

def sym_levels(data, n=17):
    """Symmetric color levels around zero."""
    vmax = np.nanpercentile(np.abs(data), 98)
    if vmax <= 0 or np.isnan(vmax):
        vmax = 1.0
    return np.linspace(-vmax, vmax, n)

# ======================== Figure ========================
fig = plt.figure(figsize=(17, 20))

# ---------- axes positions [left, bottom, width, height] ----------
# Row 0
ax_a = fig.add_axes([0.07, 0.688, 0.48, 0.27], projection=ccrs.PlateCarree())
ax_b = fig.add_axes([0.58, 0.73, 0.43, 0.23])

# Row 1  (cross-sections; leave ~0.07 gap on right of each for colorbar)
ax_c = fig.add_axes([0.11, 0.41, 0.25, 0.26])
ax_d = fig.add_axes([0.45, 0.41, 0.25, 0.26])
ax_e = fig.add_axes([0.78, 0.41, 0.25, 0.26])

# Row 2
ax_f = fig.add_axes([0.11, 0.11, 0.88, 0.30], projection=ccrs.PlateCarree())


heat_map = t_kday[ci_lev]                       # level with max heating
lon2d_f, lat2d_f = np.meshgrid(lon_f, lat_f)
vmax_a = np.nanmax(heat_map)
if vmax_a <= 0:
    vmax_a = 1.0
lvls_a = np.linspace(0, vmax_a, 15)[1:]         # skip 0

cf_a = ax_a.contourf(lon2d_f, lat2d_f, heat_map, levels=lvls_a,
                      cmap='YlOrRd', extend='max', transform=ccrs.PlateCarree())
ax_a.set_extent([55, 120, 18, 55], crs=ccrs.PlateCarree())
add_shp(ax_a, tp_shp, ec='k', lw=1.8, ls='--')
add_shp(ax_a, cont_shp, ec='dimgray', lw=0.6)
ax_a.set_xticks(np.arange(60, 140, 20), crs=ccrs.PlateCarree())
ax_a.set_yticks(np.arange(20, 60, 10), crs=ccrs.PlateCarree())
ax_a.xaxis.set_major_formatter(LongitudeFormatter())
ax_a.yaxis.set_major_formatter(LatitudeFormatter())
ax_a.tick_params(labelsize=20)
cb_a = plt.colorbar(cf_a, ax=ax_a, orientation='horizontal', shrink=0.75, pad=0.12)
cb_a.set_label('', fontsize=13)
cb_a.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
cb_a.ax.tick_params(labelsize=19)
ax_a.set_title(f'(a) Diabatic heating at $\\sigma$={sigma[ci_lev]:.2f}',
               fontsize=18, fontweight='bold', loc='left')

# ==================== (b) Vertical heating profile ====================
profile = t_kday[:, ci_lat, ci_lon]
ax_b.plot(profile, sigma, 'b-o', linewidth=2, markersize=3)
ax_b.set_ylim(1.0, 0.0)
ax_b.set_xlim(left=0)
ax_b.set_xlabel('K day$^{-1}$', fontsize=13)
ax_b.set_ylabel('$\\sigma$', fontsize=14)
ax_b.grid(True, alpha=0.3)
ax_b.tick_params(labelsize=20)

# Right axis: approximate pressure
ax_b2 = ax_b.twinx()
ax_b2.set_ylim(1.0, 0.0)
sig_ticks = [0.99, 0.85, 0.7, 0.55, 0.45, 0.3, 0.2, 0.1, 0.05]
ax_b2.set_yticks(sig_ticks)
ax_b2.set_yticklabels([])
ax_b2.set_ylabel('')
ax_b.set_title('(b) Heating profile',
               fontsize=18, fontweight='bold', loc='left')

# ==================== (c) Z cross-section along 36°N (TP) ====================
jc = np.argmin(np.abs(lat_r - 36.0))
lon_sel = (lon_r >= 45) & (lon_r <= 90)
lons_c  = lon_r[lon_sel]
z_xs_c  = z_r[:, jc, :][:, lon_sel]

Lc, Pc = np.meshgrid(lons_c, plev)
lvls_c  = sym_levels(z_xs_c, 17)

cf_c = ax_c.contourf(Lc, Pc, z_xs_c, levels=lvls_c, cmap=cmaps.BlueWhiteOrangeRed, extend='both')
ax_c.contour(Lc, Pc, z_xs_c, levels=lvls_c[::2], colors='k', linewidths=0.3)
ax_c.set_yscale('log')
ax_c.set_ylim(1000, 50)
ax_c.set_yticks([1000, 700, 500, 300, 200, 100, 50])
ax_c.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_c.set_ylabel('', fontsize=13)
ax_c.set_xlabel('', fontsize=13)
ax_c.set_xlim(50, 90)
ax_c.tick_params(labelsize=19)

# Topography fill
jg = np.argmin(np.abs(geo_lat - 36.0))
tlon_sel = (geo_lon >= 50) & (geo_lon <= 90)
tp_lons  = geo_lon[tlon_sel]
tp_p     = topo_to_pressure(topo_m[jg, tlon_sel])
ax_c.fill_between(tp_lons, tp_p, 1100, color='dimgray', zorder=5)


xtk = np.arange(50, 100, 10)
ax_c.set_xticks(xtk)
xlabels = []
for x in xtk:
    if x <= 180:
        xlabels.append(f'{int(x)}°E')
    else:
        xlabels.append(f'{int(360-x)}°W')
ax_c.set_xticklabels(xlabels, fontsize=19)

cb_c = plt.colorbar(cf_c, ax=ax_c, shrink=0.82, pad=0.03)
cb_c.set_label('', fontsize=13)
cb_c.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
cb_c.ax.tick_params(labelsize=15)
ax_c.set_title('(c) Z along 36°N', fontsize=18, fontweight='bold', loc='left')

# ==================== (d) T cross-section along 36°N (TP) ====================
t_xs_c = t_r[:, jc, :][:, lon_sel]
lvls_d  = sym_levels(t_xs_c, 17)

cf_d = ax_d.contourf(Lc, Pc, t_xs_c, levels=lvls_d, cmap=cmaps.BlueWhiteOrangeRed, extend='both')
ax_d.contour(Lc, Pc, t_xs_c, levels=lvls_d[::2], colors='k', linewidths=0.3)
ax_d.set_yscale('log')
ax_d.set_ylim(1000, 200)
ax_d.set_yticks([1000, 700, 500, 300, 200])
ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax_d.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_d.set_xlabel('', fontsize=13)
ax_d.set_xlim(50, 90)
ax_d.tick_params(labelsize=19)

ax_d.fill_between(tp_lons, tp_p, 1100, color='dimgray', zorder=5)

xtk = np.arange(50, 100, 10)
ax_d.set_xticks(xtk)
xlabels = []
for x in xtk:
    if x <= 180:
        xlabels.append(f'{int(x)}°E')
    else:
        xlabels.append(f'{int(360-x)}°W')
ax_d.set_xticklabels(xlabels, fontsize=19)




cb_d = plt.colorbar(cf_d, ax=ax_d, shrink=0.82, pad=0.03)
cb_d.set_label('', fontsize=13)
cb_d.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
cb_d.ax.tick_params(labelsize=15)
ax_d.set_title('(d) T along 36°N', fontsize=18, fontweight='bold', loc='left')

# ==================== (e) Z cross-section 27-48°N extended ====================
lat_band  = (lat_r >= 27) & (lat_r <= 48)
z_xs_e    = np.nanmean(z_r[:, lat_band, :], axis=1)   # (lev, lon)
Le, Pe    = np.meshgrid(lon_r, plev)
lvls_e    = np.linspace(-12, 12, 15)

cf_e = ax_e.contourf(Le, Pe, z_xs_e, levels=lvls_e, cmap=cmaps.BlueWhiteOrangeRed, extend='both')
ax_e.contour(Le, Pe, z_xs_e, levels=lvls_e[::2], colors='k', linewidths=0.3)
ax_e.set_yscale('log')
ax_e.set_ylim(1000, 50)
ax_e.set_xlim(100, 160)                                # 100°E – 160°E
ax_e.set_yticks([1000, 700, 500, 300, 200, 100, 50])
ax_e.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_e.set_xlabel('', fontsize=13)
ax_e.tick_params(labelsize=19)


# Custom x-tick labels with °E / °W
xtk = np.arange(100, 170, 20)
ax_e.set_xticks(xtk)
xlabels = []
for x in xtk:
    if x <= 180:
        xlabels.append(f'{int(x)}°E')
    else:
        xlabels.append(f'{int(360-x)}°W')
ax_e.set_xticklabels(xlabels, fontsize=19)

# Topography fill (average 27-48N)
gband     = (geo_lat >= 27) & (geo_lat <= 48)
topo_avg  = np.nanmean(topo_m[gband, :], axis=0)
tp_p_avg  = topo_to_pressure(topo_avg)
# (topography not drawn for panel e)

cb_e = plt.colorbar(cf_e, ax=ax_e, shrink=0.82, pad=0.03)
cb_e.set_label('', fontsize=13)
cb_e.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
cb_e.ax.tick_params(labelsize=15)
ax_e.set_title('(e) Z along 27°–48°N', fontsize=18, fontweight='bold', loc='left')

# ==================== (f) 200 hPa v-wind map ====================
i200   = np.argmin(np.abs(plev - 200))
v200   = v_r[i200]
lon2d_r, lat2d_r = np.meshgrid(lon_r, lat_r)

lvls_f = np.linspace(-1.0, 1.0, 15)
norm_f = BoundaryNorm(lvls_f, ncolors=256)

cf_f = ax_f.contourf(lon2d_r, lat2d_r, v200, levels=lvls_f,
                      cmap=cmaps.BlueWhiteOrangeRed, norm=norm_f, extend='both',
                      transform=ccrs.PlateCarree())
ax_f.set_extent([50, 180, 20, 50], crs=ccrs.PlateCarree())
add_shp(ax_f, tp_shp, ec='k', lw=1.8, ls='--')
add_shp(ax_f, cont_shp, ec='dimgray', lw=0.6)

ax_f.set_xticks(np.arange(60, 181, 20), crs=ccrs.PlateCarree())
ax_f.set_yticks(np.arange(20, 51, 10), crs=ccrs.PlateCarree())
ax_f.xaxis.set_major_formatter(LongitudeFormatter())
ax_f.yaxis.set_major_formatter(LatitudeFormatter())
ax_f.tick_params(labelsize=20)

cb_f = plt.colorbar(cf_f, ax=ax_f, orientation='horizontal',
                    shrink=0.65, pad=0.06, aspect=35)
cb_f.set_label('', fontsize=13)
cb_f.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
cb_f.ax.tick_params(labelsize=15)
ax_f.set_title('(f) 200 hPa meridional wind anomaly',
               fontsize=18, fontweight='bold', loc='left')

# ======================== Save ========================
plt.savefig(output_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigure saved → {output_fig}")
