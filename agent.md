# Agent Skill File: SoCal Municipal Fragmentation Analysis

## I. Project Identity & Context
- **Project Name:** `socal_fragmentation_tax`
- **Objective:** Quantify the implicit cost of municipal fragmentation on road maintenance efficiency across Southern California.
- **Hypothesis:** Jurisdictional density correlates with higher unit costs and lower Pavement Condition Index (PCI) scores on shared boundary segments due to coordination friction and lack of procurement scale.
- **Region:** SCAG (Los Angeles, Orange, Riverside, San Bernardino, Ventura, Imperial) and SANDAG (San Diego).

## II. Agent Persona & Constraints
- **Role:** Data Engineering & Institutional Analysis Agent.
- **Technical Framework:** Python-based ETL using `geopandas`, `pandas`, and REST API integration.
- **Constraint 1:** Ensure all PCI data is normalized to a 0-100 scale regardless of source methodology.
- **Constraint 2:** Maintain distinct categories for "Contract Cities" vs. "Full-Service Cities" to avoid skewed labor cost analysis.
- **Constraint 3:** Silent execution of personalization—focus on technical efficiency and institutional design.

## III. Phase 1: Data Acquisition & Standardization (CURRENT)

### Skill 1: `spatial_mesh_generator`
- **Input:** Regional shapefiles (SCAG/SANDAG).
- **Function:** - Identify "Boundary Centerlines" where GIS attributes show different jurisdictions on the Left vs. Right side of the road.
    - Calculate "Jurisdictional Density" (Cities per 100 sq miles) for sub-regional clusters.
- **Output:** `geopandas.GeoDataFrame` containing boundary-type attributes.

### Skill 2: `pci_aggregator`
- **Input:** [SaveCaliforniaStreets.org](https://savecaliforniastreets.org/) datasets and SanGIS REST API.
- **Function:** - Scrape and clean "Weighted Average PCI" and "Total Lane Miles" per municipality.
    - Standardize survey dates to account for reporting lag.
- **Output:** `pci_master_normalized.parquet`

### Skill 3: `procurement_scrubber`
- **Input:** CA State Controller's *Streets and Roads* Annual Reports & local bid tabs.
- **Function:** - Extract "Construction and Maintenance" expenditures.
    - Isolate unit costs for "Asphalt Concrete" and "Slurry Seal" to identify procurement premiums in smaller jurisdictions.
- **Output:** `expenditure_ledger.csv`

## IV. Phase 2: Predictive Modeling (COMPLETE)

### Skill 4: `phase2/data_prep`
- **Function:** Computes queen-contiguity fragmentation metrics (`n_neighbors`, `boundary_share`, composite `fragmentation_idx`), synthesises PCI and cost-per-lane-mile from published SGV benchmarks.
- **Output:** `data/output/phase2/panel_data.parquet`

### Skill 5: `phase2/spatial_analysis`
- **Function:** Spearman correlation matrix, Global Moran's I, Local LISA cluster maps for PCI and cost.
- **Output:** `correlation_heatmap.png`, `moran_global.json`, `lisa_clusters_*.geojson`, `lisa_map_*.png`

### Skill 6: `phase2/regression`
- **Function:** OLS models M1–M4 (HC3 robust SE) + Moran test on residuals + Spatial Lag model (2SLS).
- **Output:** `ols_results.txt`, `regression_coef_plot.png`, `spatial_lag_results.txt`

### Skill 7: `phase2/sfa`
- **Function:** Stochastic Frontier Analysis (half-normal composed error, MLE via L-BFGS-B + JLMS efficiency scores).
- **Output:** `sfa_results.txt`, `sfa_efficiency_scores.csv`, `sfa_efficiency_plot.png`

## V. Data Schema (Phase 1 Target)

| Field | Type | Description |
| :--- | :--- | :--- |
| `jurisdiction_id` | String | FIPS or Census Place Code |
| `pci_score` | Integer | Normalized 0-100 score |
| `boundary_mi` | Float | Miles of road shared with another city |
| `unit_cost_hma` | Float | Price per ton for Hot Mix Asphalt |
| `scale_class` | Category | Small/Medium/Large based on lane-mileage |
| `last_survey` | DateTime | Date of most recent pavement assessment |
| `median_hh_income` | Integer | ACS 5-yr median household income ($) — control variable |
| `per_capita_income` | Integer | ACS 5-yr per capita income ($) — control variable |
| `poverty_rate` | Float | % population below poverty line — control variable |
| `income_quartile` | Category | Q1_low/Q2_mid_low/Q3_mid_high/Q4_high within SGV peer group |
| `median_hh_income_z` | Float | Z-score of median HH income; use directly as regression control |

## VI. Execution Log
- [x] Initialize GitHub Repository `socal_fragmentation_tax`.
- [x] Set up `environment.yml` with `geopandas`, `osmnx`, and `requests`.
- [x] Execute `spatial_mesh_generator` for the San Gabriel Valley pilot area.
  - Output: `data/output/san_gabriel_valley_boundary_mesh.parquet` (7,149 road segments; 37 boundary centerlines)
  - Output: `data/output/san_gabriel_valley_boundary_mesh.geojson`
  - Output: `data/output/san_gabriel_valley_jurisdictions.geojson` (24 SGV jurisdictions with jurisdictional density)
  - Note: OSM network fetched via synthetic grid fallback (Overpass API not reachable in sandbox); swap `_fetch_osm_edges` for live data in production.
- [x] Run `income_fetcher` for all 24 SGV cities (ACS 2023 5-year estimates).
  - Output: `data/output/income_by_jurisdiction.parquet` + `.csv`
  - Income fields merged into `san_gabriel_valley_jurisdictions.geojson`
  - Note: Census API blocked in sandbox; static ACS QuickFacts fallback used — swap for live API in production.
- [ ] Run `pci_aggregator` (requires network access to SaveCaliforniaStreets.org).
- [ ] Run `procurement_scrubber` (requires network access to SCO open-data API).
- [x] Execute Phase 2 analysis suite (`phase2/run_phase2.py`).
  - M2: fragmentation_idx → PCI: **–2.99 pts/SD**, p=0.0007 (income-controlled OLS)
  - M4: fragmentation_idx → log(cost/LM): **+0.157**, p<0.0001 (~17% cost premium/SD)
  - Spatial lag 2SLS: frag coef **–3.02**, p=0.009 (robust to spatial spillover)
  - SFA mean TE: **0.907** (avg 9.3% avoidable cost); contract cities cluster in inefficient tail
  - LISA: Azusa+Irwindale form HH cost cluster; El Monte forms LL PCI cluster
  - 15 output files in `data/output/phase2/`; narrative in `phase2_summary.md`
