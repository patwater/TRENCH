"""
data_prep.py

Assembles the Phase 2 analytical panel by:
  1. Loading the enriched jurisdiction layer from Phase 1.
  2. Computing richer fragmentation metrics from the spatial structure
     (queen contiguity neighbour count, boundary-length share of perimeter).
  3. Synthesising PCI scores and cost-per-lane-mile based on published
     SGV/SoCal benchmarks and the known covariate structure.  When live
     API data becomes available, drop in the real columns and nothing else
     changes.

Output: data/output/phase2/panel_data.parquet  (city-level, n=24)
"""

import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen

DATA_DIR   = pathlib.Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR = DATA_DIR / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible seed for synthetic data
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Fragmentation metrics
# ---------------------------------------------------------------------------

def _compute_fragmentation_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Derive two fragmentation measures that vary continuously across cities,
    unlike the grid-derived jurisdictional_density (which had only 3 levels).

    n_neighbors        – number of queen-contiguous jurisdictions
    boundary_share     – shared-border length / total perimeter  (0-1)
    fragmentation_idx  – composite z-score of the above two (higher = more
                         fragmented / more coordination burden)
    """
    gdf = gdf.copy().to_crs("EPSG:3310")

    # Queen weights
    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    gdf["n_neighbors"] = [len(w.neighbors[i]) for i in range(len(gdf))]

    # Boundary share: shared edges / perimeter
    perimeters   = gdf.geometry.length
    shared_lens  = []
    geoms = list(gdf.geometry)
    for i, g in enumerate(geoms):
        nbr_idx = w.neighbors[i]
        shared  = sum(g.intersection(geoms[j]).length for j in nbr_idx)
        shared_lens.append(shared)

    gdf["boundary_share"] = [
        s / p if p > 0 else 0.0
        for s, p in zip(shared_lens, perimeters)
    ]

    # Composite fragmentation z-score
    def _zscore(s):
        return (s - s.mean()) / s.std()

    gdf["fragmentation_idx"] = (
        _zscore(gdf["n_neighbors"]) + _zscore(gdf["boundary_share"])
    ) / 2

    return gdf.to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# Synthetic PCI + cost data (calibrated to published SoCal benchmarks)
# ---------------------------------------------------------------------------
# Data-generating process is explicit so reviewers can see assumed effect
# sizes and swap in real data with zero model changes.
#
# Calibration anchors (2022-2023 published sources):
#   • SGV average weighted PCI:  ~56  (SaveCaliforniaStreets 2023)
#   • Cost per lane-mile range:  $45K–$140K  (CA SCO Streets & Roads, FY22)
#   • Income elasticity of PCI:  +0.15 (Atkinson & Sinha 2019, municipal panel)
#   • Fragmentation effect on PCI: –3 to –6 pts per SD  (Bel & Sebö 2018)
#   • Contract city cost premium: +8–12%  (Warner & Hefetz 2012)

def _synthesize_pci_and_cost(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()

    income_z   = df["median_hh_income_z"].fillna(0).values
    poverty_z  = ((df["poverty_rate"] - df["poverty_rate"].mean())
                  / df["poverty_rate"].std()).fillna(0).values
    frag_z     = df["fragmentation_idx"].values
    is_contract = (df["city_class"] == "Contract").astype(float).values
    area_z     = ((df["area_sq_mi"] - df["area_sq_mi"].mean())
                  / df["area_sq_mi"].std()).fillna(0).values

    n = len(df)

    # --- PCI (production-side outcome; higher = better roads) ---
    # PCI = 56 + 4.8*income_z – 4.2*poverty_z – 3.9*frag_z + noise
    pci_signal = (56
                  + 4.8  * income_z
                  - 4.2  * poverty_z
                  - 3.9  * frag_z
                  + 1.5  * area_z)           # larger cities maintain better
    pci_noise  = rng.normal(0, 5.5, n)
    df["pci_score"] = np.clip(pci_signal + pci_noise, 20, 95).round(1)

    # --- ADT proxy (Average Daily Traffic) --------------------------------
    # Higher-density areas have higher ADT; adds wear independent of income.
    adt_base = 18_000 + 3_500 * frag_z + rng.normal(0, 2_000, n)
    df["avg_daily_traffic"] = np.clip(adt_base, 5_000, 45_000).astype(int)

    # --- Cost per lane-mile  -----------------------------------------------
    # cost = 72K + 11K*frag_z – 14K*income_z + 9K*is_contract + noise
    cost_signal = (72_000
                   + 11_000 * frag_z
                   - 14_000 * income_z
                   + 9_000  * is_contract
                   + 2_500  * poverty_z)
    cost_noise  = rng.normal(0, 7_500, n)
    df["cost_per_lane_mile"] = np.clip(cost_signal + cost_noise, 30_000, 160_000).round(0)

    # --- Total lane miles (needed for SFA scaling) ------------------------
    # Roughly proportional to area; contract cities tend to be smaller
    lane_base = 25 + 4.5 * df["area_sq_mi"].values + rng.normal(0, 8, n)
    df["total_lane_miles"] = np.clip(lane_base, 8, 200).round(1)

    # --- Log transforms for regression ------------------------------------
    df["log_cost_per_lm"] = np.log(df["cost_per_lane_mile"])
    df["log_lane_miles"]  = np.log(df["total_lane_miles"])

    # --- data_source flag -------------------------------------------------
    df["pci_source"]  = "synthetic_calibrated"
    df["cost_source"] = "synthetic_calibrated"

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_panel() -> tuple[gpd.GeoDataFrame, "libpysal.weights.W"]:
    """
    Returns (panel_gdf, spatial_weights_W).
    panel_gdf has one row per SGV city with all Phase 1 + Phase 2 fields.
    """
    jur_path = DATA_DIR / "san_gabriel_valley_jurisdictions.geojson"
    gdf = gpd.read_file(jur_path)

    # Richer fragmentation metrics
    gdf = _compute_fragmentation_metrics(gdf)

    # Synthetic PCI + cost
    rng = np.random.default_rng(RNG_SEED)
    gdf = _synthesize_pci_and_cost(gdf, rng)

    # Dummy for regression
    gdf["is_contract"] = (gdf["city_class"] == "Contract").astype(int)

    # Queen weights on projected CRS
    gdf_proj = gdf.to_crs("EPSG:3310")
    w = Queen.from_dataframe(gdf_proj, silence_warnings=True)
    w.transform = "r"          # row-standardise for Moran / spatial lag

    # Persist panel
    out_path = OUTPUT_DIR / "panel_data.parquet"
    gdf.drop(columns="geometry").to_parquet(out_path, index=False)
    print(f"[data_prep] Panel written → {out_path}  ({len(gdf)} cities)")

    return gdf, w


if __name__ == "__main__":
    gdf, w = build_panel()
    print(gdf[["name", "n_neighbors", "boundary_share", "fragmentation_idx",
               "pci_score", "cost_per_lane_mile"]].to_string(index=False))
