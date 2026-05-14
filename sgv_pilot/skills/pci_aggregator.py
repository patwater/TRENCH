"""
pci_aggregator.py

Scrapes and cleans Weighted Average PCI and Total Lane Miles per municipality
from SaveCaliforniaStreets.org and the SanGIS REST API.

Outputs: pci_master_normalized.parquet
"""

import pathlib
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_CA_STREETS_BASE = "https://savecaliforniastreets.org"
SANGIS_REST_BASE = "https://services.arcgis.com/yp3B4dSmgjYi3T6l/arcgis/rest/services"

# Jurisdictions to pull; extend as coverage expands
SGV_CITIES = [
    "Alhambra", "Arcadia", "Azusa", "Baldwin Park", "Covina",
    "Diamond Bar", "Duarte", "El Monte", "Glendora", "Industry",
    "Irwindale", "La Puente", "La Verne", "Monrovia", "Montebello",
    "Monterey Park", "Pomona", "Rosemead", "San Dimas", "San Gabriel",
    "South El Monte", "Temple City", "West Covina", "Whittier",
]


def _normalize_pci(raw_score: float, source_max: float = 100.0) -> float:
    """Clamp and normalize any PCI value to a 0–100 integer-compatible float."""
    return max(0.0, min(100.0, raw_score * (100.0 / source_max)))


def _fetch_save_ca_streets(session: requests.Session) -> pd.DataFrame:
    """
    Pull city-level PCI summary data from SaveCaliforniaStreets.
    The site exposes a public JSON endpoint used by its interactive map.
    Falls back to an empty DataFrame if unreachable.
    """
    url = f"{SAVE_CA_STREETS_BASE}/wp-json/scs/v1/cities"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        print(f"[pci_aggregator] SaveCaliforniaStreets fetch failed: {exc} — skipping.")
        return pd.DataFrame()

    rows = []
    for entry in raw:
        name = entry.get("city_name", "")
        if name not in SGV_CITIES:
            continue
        pci_raw = entry.get("pci", None)
        if pci_raw is None:
            continue
        rows.append({
            "name": name,
            "pci_score": round(_normalize_pci(float(pci_raw))),
            "total_lane_miles": entry.get("lane_miles", None),
            "survey_year": entry.get("survey_year", None),
            "source": "SaveCaliforniaStreets",
        })
    return pd.DataFrame(rows)


def _fetch_sangis(session: requests.Session) -> pd.DataFrame:
    """
    Pull pavement condition data from the SanGIS REST API (San Diego region).
    Returns empty DataFrame for SGV-only runs; included for completeness.
    """
    layer_url = f"{SANGIS_REST_BASE}/Pavement_Condition_Index/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "JURISDICTN,PCI,SURVEY_DATE,LANE_MILES",
        "f": "json",
        "resultRecordCount": 2000,
    }
    try:
        resp = session.get(layer_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[pci_aggregator] SanGIS fetch failed: {exc} — skipping.")
        return pd.DataFrame()

    rows = []
    for feat in data.get("features", []):
        attrs = feat.get("attributes", {})
        pci_raw = attrs.get("PCI")
        if pci_raw is None:
            continue
        survey_ts = attrs.get("SURVEY_DATE")
        survey_date = (
            datetime.utcfromtimestamp(survey_ts / 1000).date()
            if survey_ts else None
        )
        rows.append({
            "name": attrs.get("JURISDICTN", ""),
            "pci_score": round(_normalize_pci(float(pci_raw))),
            "total_lane_miles": attrs.get("LANE_MILES"),
            "last_survey": survey_date,
            "source": "SanGIS",
        })
    return pd.DataFrame(rows)


def _standardize_survey_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce survey date fields to datetime and flag jurisdictions whose
    most recent survey is more than 5 years old (potential reporting lag).
    """
    today = date.today()

    def _parse(val) -> Optional[date]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, (int, float)):
            return date(int(val), 1, 1)
        try:
            return pd.to_datetime(str(val)).date()
        except Exception:
            return None

    df = df.copy()
    df["last_survey"] = df.get("last_survey", df.get("survey_year", None)).apply(_parse)
    df["reporting_lag_yrs"] = df["last_survey"].apply(
        lambda d: (today - d).days / 365.25 if d else None
    )
    df["stale_data_flag"] = df["reporting_lag_yrs"].apply(
        lambda y: y is not None and y > 5
    )
    return df


def run() -> pd.DataFrame:
    """
    Fetch, merge, normalize, and persist PCI data.
    Returns the master normalized DataFrame.
    """
    print("[pci_aggregator] Fetching PCI data …")
    session = requests.Session()
    session.headers["User-Agent"] = "socal-fragmentation-research/1.0"

    frames = [_fetch_save_ca_streets(session), _fetch_sangis(session)]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    if combined.empty:
        print("[pci_aggregator] No data retrieved from any source.")
        return combined

    combined = _standardize_survey_dates(combined)

    # Deduplicate: keep most-recent survey per jurisdiction
    combined = combined.sort_values("last_survey", na_position="last")
    combined = combined.drop_duplicates(subset=["name"], keep="last")

    out_path = OUTPUT_DIR / "pci_master_normalized.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[pci_aggregator] Wrote {len(combined)} rows → {out_path}")
    return combined


if __name__ == "__main__":
    df = run()
    print(df.to_string())
