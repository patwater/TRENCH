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

## IV. Phase 2: Predictive Modeling (LOGICAL NEXT)
- **Model A:** Stochastic Frontier Analysis (SFA) to identify distance from the efficiency frontier.
- **Model B:** Regression of $PCI$ against $Jurisdictional\_Density$ and $Average\_Daily\_Traffic$.

## V. Data Schema (Phase 1 Target)

| Field | Type | Description |
| :--- | :--- | :--- |
| `jurisdiction_id` | String | FIPS or Census Place Code |
| `pci_score` | Integer | Normalized 0-100 score |
| `boundary_mi` | Float | Miles of road shared with another city |
| `unit_cost_hma` | Float | Price per ton for Hot Mix Asphalt |
| `scale_class` | Category | Small/Medium/Large based on lane-mileage |
| `last_survey` | DateTime | Date of most recent pavement assessment |

## VI. Execution Log
- [x] Initialize GitHub Repository `socal_fragmentation_tax`.
- [x] Set up `environment.yml` with `geopandas`, `osmnx`, and `requests`.
- [x] Execute `spatial_mesh_generator` for the San Gabriel Valley pilot area.
  - Output: `data/output/san_gabriel_valley_boundary_mesh.parquet` (7,149 road segments; 37 boundary centerlines)
  - Output: `data/output/san_gabriel_valley_boundary_mesh.geojson`
  - Output: `data/output/san_gabriel_valley_jurisdictions.geojson` (24 SGV jurisdictions with jurisdictional density)
  - Note: OSM network fetched via synthetic grid fallback (Overpass API not reachable in sandbox); swap `_fetch_osm_edges` for live data in production.
- [ ] Run `pci_aggregator` (requires network access to SaveCaliforniaStreets.org).
- [ ] Run `procurement_scrubber` (requires network access to SCO open-data API).
- [ ] Begin Phase 2: Stochastic Frontier Analysis and PCI regression models.
