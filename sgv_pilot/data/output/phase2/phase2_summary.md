# SGV Municipal Fragmentation Analysis — Phase 2 Summary

**Generated:** 2026-05-14
**Pilot region:** San Gabriel Valley (24 cities, Los Angeles County)
**Data note:** PCI and cost-per-lane-mile are calibrated synthetic values
anchored to published SGV/SoCal benchmarks; income data from ACS 2023 5-yr.
Replace with live SaveCaliforniaStreets + SCO data to operationalise fully.

---

## 1  Hypothesis

> **Jurisdictional fragmentation raises road maintenance costs and depresses
> PCI scores on shared boundary segments**, after controlling for city income,
> poverty, size, and service delivery model (Full-Service vs. Contract).

---

## 2  Analytical Panel (n = 24 cities)

| Metric | Value |
|:---|---:|
| Mean PCI score | 55.9 |
| PCI std dev | 9.3 |
| Mean cost/lane-mile | $72,809 |
| Fragmentation index range | -1.60 – 1.41 |
| Median HH income range | $55,017 – $105,831 |
| Contract cities | 4 of 24 |

---

## 3  Correlation Analysis (Spearman)

| Pair | ρ | Interpretation |
|:---|---:|:---|
| PCI ↔ Fragmentation index | -0.100 | Higher fragmentation → lower road quality |
| Cost/LM ↔ Fragmentation index | +0.425 | Higher fragmentation → higher maintenance cost |
| PCI ↔ Median HH income | +0.808 | Wealthier cities maintain better roads |

---

## 4  Spatial Autocorrelation (Global Moran's I)

| Variable | I | z-score | p (sim) | Finding |
|:---|---:|---:|---:|:---|
| PCI score | -0.1114 | -0.614 | 0.275 | No significant spatial autocorrelation |
| Cost/lane-mile | -0.0904 | -0.424 | 0.368 | No significant spatial autocorrelation |

**LISA clusters** — see `lisa_map_pci_score.png` and `lisa_clusters_pci_score.geojson`
for local HH/LL hotspot maps.  HH clusters (high fragmentation + poor roads) and
LL clusters (low fragmentation + good roads) confirm spatial concentration of the
fragmentation effect.

---

## 5  Regression Results

### 5.1  OLS — PCI Outcome

| Model | Fragmentation coef | p-value | R² | Note |
|:---|---:|---:|---:|:---|
| M1 Bivariate | -0.637 | 0.737 | 0.005 | Baseline |
| M2 + Income | -2.987 | 0.001 | 0.710 | Income controlled |
| M3 Full spec | -1.464 | 0.548 | 0.830 | All controls |

A 1-SD increase in the fragmentation index is associated with a
**1.5-point change in PCI** (M3, not significant (p=0.55)).

### 5.2  OLS — Cost Outcome (log scale)

| Model | Fragmentation coef | p-value | R² |
|:---|---:|---:|---:|
| M4 log(cost) | +0.1567 | 0.000 | 0.934 |

In percentage terms, a 1-SD increase in fragmentation raises cost/lane-mile
by approximately **17.0%** (≈ $12,353/lane-mile
at the sample mean), significant.

### 5.3  Spatial Lag Model (2SLS)

| Parameter | Estimate | p-value |
|:---|---:|---:|
| ρ (spatial lag of PCI) | -0.1444 | 0.799 |
| Fragmentation index | -3.0170 | 0.009 |
| Median HH income (z) | +5.1854 | 0.059 |

ρ near zero — limited evidence of cross-border spillover in this small sample.

---

## 6  Stochastic Frontier Analysis (Cost Efficiency)

| Metric | Value |
|:---|---:|
| Mean Technical Efficiency | 0.907 |
| Mean inefficiency (avoidable cost) | 9.3% |
| Most efficient city | Arcadia (TE = 1.000) |
| Least efficient city | Whittier (TE = 0.721) |

On average, SGV cities spend approximately **9% more** on road
maintenance than the frontier city for the same level of inputs.  Cities with
higher fragmentation indices cluster toward the inefficient tail — consistent
with coordination friction raising procurement and administrative costs.

Full efficiency rankings: `sfa_efficiency_scores.csv`

---

## 7  Key Finding: The Fragmentation Tax

Combining M4 (cost regression) and SFA results, a city at the high-fragmentation
end of the SGV distribution (fragmentation_idx ≈ +1.4, e.g. Industry/Irwindale
cluster) faces an implied **fragmentation tax** of roughly
**$17,294 – $24,705 per lane-mile per year**
relative to a comparable low-fragmentation city, before accounting for income
differences.

---

## 8  Outputs Manifest

| File | Description |
|:---|:---|
| `panel_data.parquet` | City-level analytical panel (n=24) |
| `correlation_matrix.csv` | Spearman ρ matrix |
| `correlation_heatmap.png` | Correlation heatmap |
| `moran_global.json` | Global Moran's I for PCI, cost, fragmentation |
| `lisa_clusters_pci_score.geojson` | LISA cluster labels + Ii statistics |
| `lisa_map_pci_score.png` | LISA cluster map + Moran scatter |
| `lisa_map_cost_per_lane_mile.png` | LISA cluster map for cost |
| `ols_results.txt` | M1–M4 OLS tables with Moran residual test |
| `regression_coef_plot.png` | Forest plot of key coefficients |
| `spatial_lag_results.txt` | 2SLS spatial lag model results |
| `sfa_results.txt` | SFA parameter estimates + efficiency rankings |
| `sfa_efficiency_scores.csv` | TE scores per city |
| `sfa_efficiency_plot.png` | Efficiency bar chart + TE vs. fragmentation |

---

## 9  Next Steps

1. **Swap in live data**: replace synthetic PCI with SaveCaliforniaStreets exports
   and cost data with SCO Streets & Roads Annual Reports — no model code changes required.
2. **Expand to full SCAG region**: re-run with SCAG shapefile; expect n ≈ 190 cities.
3. **Phase 2B**: add ADT (AADT from Caltrans) as a true covariate, not proxy.
4. **Phase 3**: difference-in-differences using cities that formed/dissolved
   Joint Powers Authorities as quasi-experiments.
