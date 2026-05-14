"""
procurement_scrubber.py

Extracts "Construction and Maintenance" expenditures from the
CA State Controller's Streets and Roads Annual Reports and isolates
unit costs for Asphalt Concrete and Slurry Seal.

Outputs: expenditure_ledger.csv
"""

import pathlib
import re
from typing import Optional

import pandas as pd
import requests

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# State Controller open-data endpoint for Streets & Roads Annual Reports
# Dataset: Local Streets and Roads Annual Reports (LRSAR)
SCO_BASE = "https://bythenumbers.sco.ca.gov/resource"
LRSAR_ENDPOINT = f"{SCO_BASE}/jtam-ykmt.json"  # Streets & Roads expenditures

# Target expenditure categories (from the LRSAR schema)
CONSTRUCTION_CATEGORIES = {
    "construction_and_reconstruction",
    "maintenance",
    "construction_maintenance",
}

UNIT_COST_MATERIALS = {
    "asphalt_concrete": "unit_cost_hma",
    "slurry_seal":      "unit_cost_slurry",
}

SGV_CITIES = [
    "Alhambra", "Arcadia", "Azusa", "Baldwin Park", "Covina",
    "Diamond Bar", "Duarte", "El Monte", "Glendora", "Industry",
    "Irwindale", "La Puente", "La Verne", "Monrovia", "Montebello",
    "Monterey Park", "Pomona", "Rosemead", "San Dimas", "San Gabriel",
    "South El Monte", "Temple City", "West Covina", "Whittier",
]

CONTRACT_CITIES = {
    "Industry", "Irwindale", "La Puente", "South El Monte",
}


def _fetch_sco_lrsar(session: requests.Session, fiscal_year: int = 2023) -> pd.DataFrame:
    """
    Pull expenditure data for SGV cities from the SCO open-data API.
    fiscal_year: the year ending of the fiscal period (e.g. 2023 → FY 2022-23).
    """
    params = {
        "$limit": 5000,
        "$where": f"fiscal_year='{fiscal_year}'",
        "$select": (
            "agency_name,fiscal_year,"
            "construction_and_reconstruction_total,"
            "maintenance_total,"
            "total_lane_miles_maintained,"
            "total_expenditures"
        ),
    }
    try:
        resp = session.get(LRSAR_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        print(f"[procurement_scrubber] SCO LRSAR fetch failed: {exc} — skipping.")
        return pd.DataFrame()

    rows = []
    for entry in raw:
        name_raw = entry.get("agency_name", "")
        matched_name = _fuzzy_match_city(name_raw)
        if matched_name is None:
            continue
        rows.append({
            "name":                  matched_name,
            "fiscal_year":           entry.get("fiscal_year"),
            "construction_total":    _to_float(entry.get("construction_and_reconstruction_total")),
            "maintenance_total":     _to_float(entry.get("maintenance_total")),
            "total_lane_miles":      _to_float(entry.get("total_lane_miles_maintained")),
            "total_expenditures":    _to_float(entry.get("total_expenditures")),
            "city_class":            "Contract" if matched_name in CONTRACT_CITIES else "Full-Service",
        })
    return pd.DataFrame(rows)


def _fuzzy_match_city(raw_name: str) -> Optional[str]:
    """Case-insensitive partial match against the SGV city list."""
    raw_clean = re.sub(r"\s*(city|town)\s*of\s*", "", raw_name, flags=re.IGNORECASE).strip()
    for city in SGV_CITIES:
        if city.lower() == raw_clean.lower():
            return city
    return None


def _compute_unit_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive cost-per-lane-mile as a proxy for unit procurement cost.
    True unit costs ($/ton for HMA, $/sq-yd for slurry) require bid-tab data
    not available via public API; this metric flags procurement-scale outliers.
    """
    df = df.copy()
    df["cost_per_lane_mile"] = df.apply(
        lambda r: (
            (r["construction_total"] + r["maintenance_total"]) / r["total_lane_miles"]
            if r["total_lane_miles"] and r["total_lane_miles"] > 0 else None
        ),
        axis=1,
    )
    # Scale class by lane mileage
    def _scale(lane_mi):
        if lane_mi is None:
            return "Unknown"
        if lane_mi < 50:
            return "Small"
        if lane_mi < 150:
            return "Medium"
        return "Large"

    df["scale_class"] = df["total_lane_miles"].apply(_scale)
    return df


def _to_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def run(fiscal_year: int = 2023) -> pd.DataFrame:
    """
    Fetch, clean, and persist procurement expenditure data.
    Returns the expenditure ledger DataFrame.
    """
    print(f"[procurement_scrubber] Fetching SCO expenditure data for FY ending {fiscal_year} …")
    session = requests.Session()
    session.headers["User-Agent"] = "socal-fragmentation-research/1.0"

    df = _fetch_sco_lrsar(session, fiscal_year=fiscal_year)

    if df.empty:
        print("[procurement_scrubber] No expenditure records retrieved.")
        return df

    df = _compute_unit_costs(df)

    out_path = OUTPUT_DIR / "expenditure_ledger.csv"
    df.to_csv(out_path, index=False)
    print(f"[procurement_scrubber] Wrote {len(df)} rows → {out_path}")

    # Quick summary
    by_class = df.groupby("city_class")["cost_per_lane_mile"].describe()
    print("\nCost-per-lane-mile by city class:")
    print(by_class.to_string())
    return df


if __name__ == "__main__":
    df = run()
    print(df.to_string())
