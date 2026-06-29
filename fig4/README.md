# Figure 4 — Idealized LBM Response to Western Tibetan Plateau Thermal Forcing

This folder contains the code and final figure for **Figure 4** of the paper:

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

Figure 4 is an idealized **linear baroclinic model (LBM)** experiment that isolates the atmospheric
response to western-Tibetan-Plateau diabatic heating. A steady diabatic heating anomaly centered
over the western Tibetan Plateau (panels **a**, **b**) is prescribed in the model (T21 horizontal /
L20 vertical resolution); the model is integrated to a steady state and the response is taken as the
time mean of the **last 10 integration days**. Panels **c**–**f** show that steady response in
geopotential height, temperature, and 200-hPa meridional wind, illustrating the downstream wave
train that propagates from the plateau toward the Northwestern Pacific.

![Figure 4](figures/fig4_tibet_lbm_36N.png)

---

## Panels

| Panel | Content | Variable / source |
|-------|---------|--------------------|
| **(a)** | Prescribed diabatic heating at σ = 0.65 (level of maximum heating), horizontal map over Asia; dashed = TP boundary | forcing `t` (K day⁻¹) |
| **(b)** | Vertical profile of the prescribed heating at the forcing-center column (heating rate vs. σ) | forcing `t` (K day⁻¹) |
| **(c)** | Geopotential-height response, longitude–pressure cross-section along 36°N, 50–90°E (grey = topography) | response `z` (m) |
| **(d)** | Temperature response, longitude–pressure cross-section along 36°N, 50–90°E (grey = topography) | response `t` (K) |
| **(e)** | Geopotential-height response, longitude–pressure cross-section averaged 27°–48°N, 100°–160°E (downstream) | response `z` (m) |
| **(f)** | 200-hPa meridional-wind response, map 50°–180°E, 20°–50°N (downstream wave train); dashed = TP boundary | response `v` (m s⁻¹) |

The heating center and its σ-level are detected automatically from the forcing field (the maximum of
the prescribed heating); panel (a) is drawn at that level and panel (b) is the column through that
center.

---

## Repository layout

```
fig4/
├── README.md
├── code/
│   └── plot_fig4_tibet_36N.py          # self-contained script that produces Figure 4
└── figures/
    └── fig4_tibet_lbm_36N.png          # final Figure 4 (high-resolution render)
```

> **Note:** Figure 4 is produced directly from the LBM experiment output (one forcing file and one
> response file) and produces **no intermediate CSV files**. The script is fully self-contained — it
> imports no project modules.

## Requirements

- Python ≥ 3.10
- `numpy`, `xarray`, `matplotlib`, `cartopy`, `cmaps`

## Reproducing the figure

The script uses **absolute paths** pointing at the original compute environment. To regenerate
elsewhere, update the path constants near the top of `code/plot_fig4_tibet_36N.py`
(`force_file`, `res_file`, `geo_file`, `tp_shp`, `cont_shp`, `output_fig`) to your local copies,
then run:

```bash
python plot_fig4_tibet_36N.py
```

This writes `fig4_tibet_lbm_36N.png`.

> **Resolution note:** the PNG committed here is the final **high-resolution (600 dpi)** render. The
> script as listed saves at `dpi=300`; set `dpi=600` in the final `plt.savefig(...)` call to
> reproduce the committed resolution. A vector PDF can be produced by adding a second `plt.savefig`
> with a `.pdf` target.

## Input data

| Field | Panel(s) | Original path / product |
|-------|----------|--------------------------|
| Prescribed diabatic heating `t` (LBM forcing, T21L20) | (a), (b) | `LBM/ln_solver/data/tintgr/tibet_heat.frc.t21l20.nc` |
| LBM steady-state response `z`, `t`, `u`, `v` (T21L20; mean of last 10 days) | (c)–(f) | `LBM/ln_solver/data/tintgr/tibet_heat.t21l20.nc` |
| Model orography / geopotential `z` (topography fill) | (c), (d) | `SHP/dixing/geo.nc` |
| World continent shapefile | map land masks (a), (f) | `WORLD_SHP/continent.shp` |
| Tibetan Plateau boundary (2021) | (a), (f) | `TP_SHP/TPBoundary_new(2021)/...` |

The two LBM `.nc` files are the experiment's prescribed forcing and the model response; they are
produced by the linear baroclinic model run (not redistributed here). The shapefiles and orography
are the same auxiliary products used elsewhere in this repository.

## Citation

If you use this code or figure, please cite the paper above. Code released to accompany the paper.
