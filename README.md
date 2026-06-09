# Role of Western Tibetan Plateau Thermal Forcing in the 2025 Summer Extreme Marine Heatwave over the Northwestern Pacific

Code, derived data, and final figures accompanying the paper:

> **Role of Western Tibetan Plateau Thermal Forcing in the 2025 Summer Extreme Marine Heatwave over the Northwestern Pacific**
>
> Jianning Tao¹, Wei Hua¹\*, Lihua Zhu¹, Xiaofei Wu¹, Shangru Li¹
>
> ¹ School of Atmospheric Sciences / Climate Change and Resource Utilization in Complex Terrain Regions Key Laboratory of Sichuan Province / Sichuan Provincial Engineering Research Center for Meteorological Disaster Prediction and Early Warning, Chengdu University of Information Technology, Chengdu, China
>
> \* Corresponding author: Wei Hua

---

## Figures

Each figure lives in its own self-documented folder (`code/`, `figures/`, and `data/` where
applicable). Click a figure's README for panel descriptions, requirements, input-data sources, and
reproduction steps.

| Figure | Description | Folder |
|--------|-------------|--------|
| **Figure 1** | 2025 NW Pacific summer marine-heatwave overview: SSTA map, MHW time series, variability, raw-vs-detrended trend, and mixed-layer / surface-flux heat budget (6 panels) | [`fig1/`](fig1/README.md) |
| **Figure 2** | 2025 JJA atmospheric & ocean-surface anomaly fields: geopotential height, vertical velocity, cloud cover, net shortwave, mixed-layer depth, SLP + 10 m wind (6 panels) | [`fig2/`](fig2/README.md) |
| **Figure 3** | Upper-tropospheric circulation, Tibetan Plateau heating, and the SST link: V200 + jet axis, T2m map, T & Z cross-sections at 36°N, eddy Z200 + Plumb flux, T2m index, SST-onto-T2m regression (7 panels) | [`fig3/`](fig3/README.md) |
| **Figure 4** | Idealized linear baroclinic model (LBM) response to western-Tibetan-Plateau diabatic heating: prescribed heating map + vertical profile, geopotential-height & temperature cross-sections at 36°N, downstream height cross-section, and the 200-hPa meridional-wind wave train (6 panels) | [`fig4/`](fig4/README.md) |

### Figure 1
[![Figure 1](fig1/figures/sst_6panel_combined_new.png)](fig1/README.md)

### Figure 2
[![Figure 2](fig2/figures/combined_3x2_anomalies_2025_summer_aligned_new.png)](fig2/README.md)

### Figure 3
[![Figure 3](fig3/figures/combined_multirow_figure_new.png)](fig3/README.md)

### Figure 4
[![Figure 4](fig4/figures/fig4_tibet_lbm_36N.png)](fig4/README.md)

---

## Repository layout

```
.
├── README.md                 # this landing page
├── fig1/                      # Figure 1 — MHW overview & heat budget
│   ├── README.md
│   ├── code/                  # figure script, base module, calculation-only CSV entry point
│   ├── figures/               # final Figure 1 (600 dpi)
│   └── data/                  # derived CSV products (traceable to the code)
├── fig2/                      # Figure 2 — atmospheric & ocean-surface anomalies
│   ├── README.md
│   ├── code/                  # self-contained figure script
│   └── figures/               # final Figure 2 (300 dpi)
├── fig3/                      # Figure 3 — circulation, TP heating, SST link
│   ├── README.md
│   ├── code/                  # self-contained figure script
│   └── figures/               # final Figure 3 (300 dpi)
└── fig4/                      # Figure 4 — idealized LBM response to TP heating
    ├── README.md
    ├── code/                  # self-contained figure script
    └── figures/               # final Figure 4 (high-resolution render)
```

## Requirements

Python ≥ 3.10 with `numpy`, `pandas`, `scipy`, `xarray`, `matplotlib`, `cartopy`, `cmaps`, and
[`marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves) (Figure 1 only). See each figure's
README for the exact per-figure dependency list.

## Data availability

The repository ships the **code** and the **small derived products** (Figure 1 CSVs) plus the
**final figures**. The raw input datasets (OISST, GODAS/NCEP ocean fields, ERA5 atmospheric and
surface-flux fields) are large external products and are **not** redistributed here; their original
products and paths are listed in each figure's README under *Input data*.

## Citation

If you use this code or these figures, please cite the paper above.
