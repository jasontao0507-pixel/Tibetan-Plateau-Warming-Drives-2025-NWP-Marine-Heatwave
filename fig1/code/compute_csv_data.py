#!/usr/bin/env python3
"""Regenerate the three derived CSV products used by Figure 1 — calculation only.

This is a thin, plotting-free entry point so the CSV calculations can be traced
and reproduced independently of figure rendering. It does NOT redefine any math:
it calls the exact same functions that the figure script uses, so the numbers are
guaranteed identical to those behind the published figure.

Provenance (CSV  <-  function  <-  file):

  data/sst_panel_d_raw_vs_detrended_1982_2025.csv
      <- compute_panel_c_data()        <- plot_sst_2x2_combined.py
      Box-mean JJA SST, raw vs de-trended, 1982-2025, anomalies vs 1982-2010.

  data/tp_heat_budget_2025_summer_terms.csv
      <- compute_panel_d_budget()      <- plot_sst_2x2_combined.py
      2025 JJA mixed-layer heat budget: dSSTa, Qnet term, advection term, residual.

  data/tp_qnet_component_contribution_2025.csv
      <- compute_panel_f_qnet_components() <- plot_sst_6panel_combined.py
      2025 JJA decomposition of Qnet into shortwave / longwave / sensible / latent.

The output paths and all input-data paths are inherited from
``plot_sst_6panel_combined`` (imported below). Adjust the path constants in that
script (and in ``plot_sst_2x2_combined.py``) to your local environment before running.

Usage:
    python compute_csv_data.py
"""

import os

# Importing the figure script wires up the base module and remaps the CSV output
# paths (base.OUT_PANEL_C_CSV / base.OUT_PANEL_D_CSV / OUT_PANEL_F_CSV) without
# triggering any plotting (main() is guarded by __name__ == "__main__").
import plot_sst_6panel_combined as fig1


def main():
    os.makedirs(fig1.OUT_DIR, exist_ok=True)

    print("=== Computing Figure 1 derived CSVs (calculation only) ===")

    # (d) raw vs de-trended box-mean JJA SST series -> sst_panel_d_raw_vs_detrended_1982_2025.csv
    fig1.base.compute_panel_c_data()

    # (e) mixed-layer heat budget terms          -> tp_heat_budget_2025_summer_terms.csv
    fig1.base.compute_panel_d_budget()

    # (f) Qnet component contributions           -> tp_qnet_component_contribution_2025.csv
    fig1.compute_panel_f_qnet_components()

    print("Done. CSVs written to:", fig1.OUT_DIR)
    print(" -", fig1.OUT_PANEL_D_CSV)
    print(" -", fig1.base.OUT_PANEL_D_CSV)
    print(" -", fig1.OUT_PANEL_F_CSV)


if __name__ == "__main__":
    main()
