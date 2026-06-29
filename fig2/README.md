# Figure 2 — 2025 JJA Atmospheric & Ocean-Surface Anomaly Fields

This folder contains the code and final figure for **Figure 2** of the paper:

> **Teleconnection between Tibetan Plateau Warming and Marine Heatwave over the northwestern Pacific**
>
> Jianning Tao¹, Wei Hua¹\*, Huijun Wang²,³, Lihua Zhu¹, Xiaofei Wu¹, Shangru Li¹
>
> ¹ School of Atmospheric Sciences / Climate Change and Resource Utilization in Complex Terrain Regions Key Laboratory of Sichuan Province / Sichuan Provincial Engineering Research Center for Meteorological Disaster Prediction and Early Warning, Chengdu University of Information Technology, Chengdu, China
>
> ² State Key Laboratory of Climate System Prediction and Risk Management / Key Laboratory of Meteorological Disaster, Ministry of Education / Collaborative Innovation Center on Forecast and Evaluation of Meteorological Disasters, Nanjing University of Information Science and Technology, Nanjing, China
>
> ³ School of Atmospheric Sciences, Nanjing University of Information Science and Technology, Nanjing, China
>
> \* Corresponding author: Wei Hua

Figure 2 is a six-panel (3×2) view of the 2025 summer (JJA) anomalies in the atmospheric circulation
and ocean-surface forcing associated with the Northwestern Pacific marine heatwave. Each anomaly is
2025 JJA minus the 1982–2024 climatology; stippling marks where |anomaly| exceeds one interannual
standard deviation.

![Figure 2](figures/combined_3x2_anomalies_2025_summer_aligned_new.png)

---

## Panels

| Panel | Content | Variable / source |
|-------|---------|--------------------|
| **(a)** | Geopotential-height anomaly, longitude–pressure section averaged over 27–48°N | ERA5 `z` (detrended) |
| **(b)** | Vertical-velocity anomaly, longitude–pressure section averaged over 27–48°N | ERA5 `w` (detrended) |
| **(c)** | Total cloud cover anomaly (map) | ERA5 `tcc` (detrended) |
| **(d)** | Surface net shortwave-radiation anomaly (map) | ERA5 `avg_snswrf` (detrended) |
| **(e)** | Ocean mixed-layer-depth anomaly (map) | GODAS/NCEP `dbss_obml` |
| **(f)** | Sea-level-pressure (contours) and 10 m wind (vectors) anomalies, shaded by wind-speed anomaly | ERA5 `msl`, `u10`, `v10` |

The dashed vertical lines in (a)/(b) and the black box in (c)–(f) mark the analysis region
(~27–48°N, 127–164°E).

---

## Repository layout

```
fig2/
├── README.md
├── code/
│   └── plot_combined_3x2_anomalies_aligned.py   # self-contained script that produces Figure 2
└── figures/
    └── combined_3x2_anomalies_2025_summer_aligned_new.png   # final Figure 2 (300 dpi)
```

> **Note:** Figure 2 is computed directly from the raw input fields and produces **no intermediate
> CSV files** (unlike Figure 1). The script is fully self-contained — it imports no project modules.

## Requirements

- Python ≥ 3.10
- `numpy`, `xarray`, `matplotlib`, `cartopy`, `cmaps`

## Reproducing the figure

The script uses **absolute paths** pointing at the original compute environment. To regenerate
elsewhere, update the path constants near the top of
`code/plot_combined_3x2_anomalies_aligned.py` (`PATH_Z`, `PATH_W`, `PATH_TCC`, `PATH_QNET`,
`PATH_MLD`, `PATH_SLP`, `PATH_10UV`, `SHP_PATH`) to your local copies, then run:

```bash
python plot_combined_3x2_anomalies_aligned.py
```

This writes `combined_3x2_anomalies_2025_summer_aligned_new.png` (300 dpi) next to the script.

## Input data

| Field | Panel | Original path / product |
|-------|-------|--------------------------|
| Geopotential height `z` (1000–50 hPa, JJA, detrended) | (a) | `ERA5/Z_1000_50hpa_1982_2025_JJA_detrended.nc` |
| Vertical velocity `w` (1000–200 hPa, detrended) | (b) | `W_1982_2025_1000_200hpa_detrended_signal.nc` |
| Total cloud cover `tcc` (detrended) | (c) | `TCC_1982_2025_summer_detrended.nc` |
| Surface net shortwave `avg_snswrf` (detrended) | (d) | `ERA5/Qnet_new/Qnet_1982_2025_summer_detrended.nc` |
| Ocean mixed-layer depth `dbss_obml` | (e) | `NCEP/obml/dbss_obml_1982_2025.nc` |
| Sea-level pressure `msl` | (f) | `ERA5/slp_1982-2025.nc` |
| 10 m wind `u10`, `v10` | (f) | `ERA5/10uv_1982_2025.nc` |
| World continent shapefile | map land masks | `WORLD_SHP/continent.shp` |

## Citation

If you use this code or figure, please cite the paper above. Code released to accompany the paper.
