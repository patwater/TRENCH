"""
spatial_analysis.py

Correlation matrix, Global Moran's I, and Local LISA (cluster map)
for the SGV fragmentation panel.

Outputs (all → data/output/phase2/):
  correlation_matrix.csv
  correlation_heatmap.png
  moran_global.json
  lisa_clusters.geojson
  lisa_map.png
"""

import json
import pathlib

import esda
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from libpysal.weights import Queen

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_VARS = [
    "pci_score",
    "cost_per_lane_mile",
    "fragmentation_idx",
    "n_neighbors",
    "boundary_share",
    "median_hh_income",
    "per_capita_income",
    "poverty_rate",
    "area_sq_mi",
    "avg_daily_traffic",
]

LISA_COLORS = {
    "HH": "#d7191c",   # High-high (fragmented + bad roads) — hot spot
    "LL": "#2c7bb6",   # Low-low  (cohesive + good roads) — cold spot
    "LH": "#abd9e9",   # Low fragmentation surrounded by high
    "HL": "#fdae61",   # High fragmentation surrounded by low
    "NS": "#eeeeee",   # Not significant
}


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def run_correlation(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spearman rank correlation matrix across analysis variables."""
    df = gdf[ANALYSIS_VARS].copy()

    corr_mat = df.corr(method="spearman").round(3)
    corr_mat.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Spearman Correlation Matrix — SGV Municipal Panel (n=24)",
        fontsize=12, pad=14,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print("[spatial_analysis] Correlation matrix saved.")
    _print_key_correlations(corr_mat)
    return corr_mat


def _print_key_correlations(corr: pd.DataFrame) -> None:
    pairs = [
        ("pci_score",         "fragmentation_idx"),
        ("cost_per_lane_mile","fragmentation_idx"),
        ("pci_score",         "median_hh_income"),
        ("cost_per_lane_mile","median_hh_income"),
        ("pci_score",         "poverty_rate"),
        ("cost_per_lane_mile","poverty_rate"),
        ("pci_score",         "n_neighbors"),
        ("cost_per_lane_mile","n_neighbors"),
    ]
    print("\n  Key Spearman correlations:")
    for a, b in pairs:
        if a in corr.columns and b in corr.columns:
            print(f"    {a:25s} vs {b:25s} : {corr.loc[a, b]:+.3f}")


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------

def run_global_moran(gdf: gpd.GeoDataFrame, w) -> dict:
    """
    Global Moran's I for PCI and cost-per-lane-mile.
    Returns dict of results and writes moran_global.json.
    """
    results = {}
    for var in ("pci_score", "cost_per_lane_mile", "fragmentation_idx"):
        mi = esda.Moran(gdf[var].values, w, permutations=999)
        results[var] = {
            "I":       round(float(mi.I), 4),
            "EI":      round(float(mi.EI), 4),
            "z_score": round(float(mi.z_norm), 4),
            "p_value": round(float(mi.p_norm), 4),
            "p_sim":   round(float(mi.p_sim), 4),
            "interpretation": _moran_interpret(mi.I, mi.p_sim),
        }
        print(f"  Moran's I [{var:25s}]: I={mi.I:+.4f}  z={mi.z_norm:+.3f}  p={mi.p_sim:.3f}  → {results[var]['interpretation']}")

    with open(OUTPUT_DIR / "moran_global.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[spatial_analysis] Global Moran's I saved.")
    return results


def _moran_interpret(I: float, p: float) -> str:
    if p > 0.10:
        return "No significant spatial autocorrelation"
    if I > 0:
        return "Positive spatial autocorrelation (clustering)"
    return "Negative spatial autocorrelation (dispersion)"


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

def run_lisa(gdf: gpd.GeoDataFrame, w, var: str = "pci_score") -> gpd.GeoDataFrame:
    """
    Compute LISA cluster labels for `var` and export GeoJSON + PNG map.
    Returns GeoDataFrame with cluster labels attached.
    """
    values = gdf[var].values
    lm = esda.Moran_Local(values, w, permutations=999, seed=42)

    sig   = lm.p_sim < 0.10   # 10 % significance threshold
    quads = lm.q               # 1=HH, 2=LH, 3=LL, 4=HL

    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels = [quad_map.get(q, "NS") if s else "NS" for q, s in zip(quads, sig)]

    gdf = gdf.copy()
    gdf[f"lisa_{var}"]       = labels
    gdf[f"lisa_{var}_Ii"]    = lm.Is.round(4)
    gdf[f"lisa_{var}_p"]     = lm.p_sim.round(4)

    out_path = OUTPUT_DIR / f"lisa_clusters_{var}.geojson"
    gdf[[
        "name", "geometry",
        f"lisa_{var}", f"lisa_{var}_Ii", f"lisa_{var}_p",
        var, "fragmentation_idx", "median_hh_income",
    ]].to_file(out_path, driver="GeoJSON")

    _plot_lisa(gdf, var, labels, lm)
    print(f"[spatial_analysis] LISA ({var}) saved → {out_path}")

    # Print cluster summary
    from collections import Counter
    counts = Counter(labels)
    for k in ("HH", "LL", "HL", "LH", "NS"):
        if counts[k]:
            names = [gdf.iloc[i]["name"] for i, l in enumerate(labels) if l == k]
            print(f"    {k}: {', '.join(names)}")

    return gdf


def _plot_lisa(gdf: gpd.GeoDataFrame, var: str, labels: list, lm) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Panel 1: LISA cluster map ----------------------------------------
    ax = axes[0]
    colors = [LISA_COLORS[l] for l in labels]
    gdf.plot(color=colors, edgecolor="white", linewidth=0.5, ax=ax)
    for _, row in gdf.iterrows():
        c = row.geometry.centroid
        ax.annotate(
            row["name"].split()[0],
            xy=(c.x, c.y),
            fontsize=5.5,
            ha="center", va="center",
            color="#333333",
        )
    patches = [mpatches.Patch(color=v, label=k) for k, v in LISA_COLORS.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right", title="LISA quad")
    ax.set_title(f"LISA Cluster Map — {var}", fontsize=10)
    ax.set_axis_off()

    # ---- Panel 2: Moran scatter plot --------------------------------------
    ax2 = axes[1]
    z_var  = (gdf[var].values - gdf[var].mean()) / gdf[var].std()
    lag_z  = lm.w.sparse.dot(z_var)   # spatial lag of z
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.axvline(0, color="grey", lw=0.8, ls="--")
    ax2.scatter(z_var, lag_z, c=colors, edgecolors="k", linewidth=0.4, s=60)
    for i, row in gdf.iterrows():
        ax2.annotate(row["name"].split()[0], (z_var[i], lag_z[i]), fontsize=5,
                     xytext=(3, 3), textcoords="offset points")
    m, b = np.polyfit(z_var, lag_z, 1)
    xx = np.linspace(z_var.min(), z_var.max(), 100)
    ax2.plot(xx, m * xx + b, color="darkred", lw=1.5)
    ax2.set_xlabel(f"Standardised {var}", fontsize=9)
    ax2.set_ylabel(f"Spatial Lag of {var}", fontsize=9)
    ax2.set_title(f"Moran Scatter — {var}", fontsize=10)

    plt.suptitle("SGV Municipal Fragmentation — Spatial Autocorrelation", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"lisa_map_{var}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_spatial(gdf: gpd.GeoDataFrame, w) -> dict:
    print("\n── Correlation Analysis ─────────────────────────────────────")
    corr = run_correlation(gdf)

    print("\n── Global Moran's I ─────────────────────────────────────────")
    moran_results = run_global_moran(gdf, w)

    print("\n── LISA Clusters ─────────────────────────────────────────────")
    gdf = run_lisa(gdf, w, var="pci_score")
    gdf = run_lisa(gdf, w, var="cost_per_lane_mile")

    return {"correlation": corr, "moran": moran_results, "gdf_with_lisa": gdf}
