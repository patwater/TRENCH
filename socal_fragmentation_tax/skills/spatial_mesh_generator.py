"""
spatial_mesh_generator.py

Identifies boundary centerlines and computes jurisdictional density
for sub-regional clusters in SCAG/SANDAG territory.

Inputs:  OSM road network for a pilot bounding box (no local shapefiles required).
Outputs: GeoDataFrame with boundary-type attributes written to data/output/.
"""

import json
import math
import pathlib
from typing import Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# San Gabriel Valley bounding box  (W, S, E, N)
SGV_BBOX = (-118.15, 34.00, -117.70, 34.20)

# Cities within the San Gabriel Valley pilot area (Census FIPS place codes)
# Format: (name, fips_place_code, approx_area_sq_mi)
SGV_JURISDICTIONS = [
    ("Alhambra",       "0600884", 7.6),
    ("Arcadia",        "0602252", 11.1),
    ("Azusa",          "0603526", 9.5),
    ("Baldwin Park",   "0604094", 6.8),
    ("Covina",         "0616742", 7.0),
    ("Diamond Bar",    "0619192", 14.9),
    ("Duarte",         "0619766", 6.6),
    ("El Monte",       "0622230", 9.6),
    ("Glendora",       "0630378", 19.3),
    ("Industry",       "0636028", 12.0),
    ("Irwindale",      "0636392", 6.1),
    ("La Puente",      "0640004", 3.5),
    ("La Verne",       "0640032", 12.5),
    ("Monrovia",       "0648788", 13.7),
    ("Montebello",     "0648860", 8.3),
    ("Monterey Park",  "0648972", 7.7),
    ("Pomona",         "0658072", 22.9),
    ("Rosemead",       "0663218", 5.2),
    ("San Dimas",      "0664882", 15.4),
    ("San Gabriel",    "0665028", 4.1),
    ("South El Monte", "0672016", 3.0),
    ("Temple City",    "0678218", 3.9),
    ("West Covina",    "0683668", 16.0),
    ("Whittier",       "0684200", 14.8),
]

# Contract cities (Lakewood model — purchase law enforcement from county)
CONTRACT_CITIES = {
    "Industry", "Irwindale", "La Puente", "South El Monte",
}


def _build_jurisdiction_polygons(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Construct approximate Voronoi-based jurisdiction polygons from city
    centroid seeds clipped to the pilot bounding box.  In production this
    would be replaced by actual Census TIGER/Line place shapefiles.
    """
    west, south, east, north = bbox
    region_poly = Polygon([
        (west, south), (east, south), (east, north), (west, north), (west, south)
    ])

    # Seed centroids spread proportionally across the bounding box
    rng_x = east - west
    rng_y = north - south
    n = len(SGV_JURISDICTIONS)
    cols = math.ceil(math.sqrt(n * rng_x / rng_y))
    rows = math.ceil(n / cols)

    seeds: list[dict] = []
    for idx, (name, fips, area_sq_mi) in enumerate(SGV_JURISDICTIONS):
        col = idx % cols
        row = idx // cols
        cx = west + rng_x * (col + 0.5) / cols
        cy = south + rng_y * (row + 0.5) / rows
        seeds.append({
            "jurisdiction_id": fips,
            "name": name,
            "area_sq_mi": area_sq_mi,
            "city_class": "Contract" if name in CONTRACT_CITIES else "Full-Service",
            "geometry": Point(cx, cy),
        })

    seed_gdf = gpd.GeoDataFrame(seeds, crs="EPSG:4326")

    # Voronoi tessellation clipped to region
    from shapely.ops import voronoi_diagram
    mp = seed_gdf.geometry.unary_union
    regions = voronoi_diagram(mp, envelope=region_poly)
    voronoi_polys = list(regions.geoms)

    # Match each Voronoi cell back to its seed by containment
    matched: list[Optional[Polygon]] = [None] * len(seeds)
    for poly in voronoi_polys:
        clipped = poly.intersection(region_poly)
        for i, s in enumerate(seeds):
            if clipped.contains(s["geometry"]):
                matched[i] = clipped
                break

    for i, m in enumerate(matched):
        if m is None:
            matched[i] = region_poly  # fallback

    seed_gdf = seed_gdf.copy()
    seed_gdf["geometry"] = matched
    return seed_gdf.set_geometry("geometry")


def _identify_boundary_centerlines(
    jur_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Tag each road segment as 'Boundary' if it falls within a buffer of a
    shared border between two different jurisdictions.
    """
    jur_proj = jur_gdf.to_crs("EPSG:3310")   # CA Albers, metres
    roads_proj = roads_gdf.to_crs("EPSG:3310")

    # Build shared-border lines between adjacent jurisdictions
    border_lines = []
    names = jur_proj["name"].tolist()
    geoms = jur_proj["geometry"].tolist()
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            shared = geoms[i].intersection(geoms[j])
            if not shared.is_empty and shared.geom_type in ("LineString", "MultiLineString"):
                border_lines.append({
                    "left_jurisdiction": names[i],
                    "right_jurisdiction": names[j],
                    "geometry": shared,
                })

    if not border_lines:
        roads_proj["is_boundary"] = False
        roads_proj["left_jurisdiction"] = None
        roads_proj["right_jurisdiction"] = None
        return roads_proj.to_crs("EPSG:4326")

    border_gdf = gpd.GeoDataFrame(border_lines, crs="EPSG:3310")
    border_buffer = border_gdf.copy()
    border_buffer["geometry"] = border_gdf.geometry.buffer(15)  # 15-metre corridor

    # Spatial join: roads within a border buffer
    joined = gpd.sjoin(roads_proj, border_buffer, how="left", predicate="within")
    roads_proj["is_boundary"] = ~joined["index_right"].isna()
    roads_proj["left_jurisdiction"] = joined.get("left_jurisdiction", None)
    roads_proj["right_jurisdiction"] = joined.get("right_jurisdiction", None)

    return roads_proj.to_crs("EPSG:4326")


def _compute_jurisdictional_density(
    jur_gdf: gpd.GeoDataFrame,
    cluster_radius_mi: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    For each jurisdiction centroid, count how many other jurisdiction centroids
    fall within cluster_radius_mi and express as cities-per-100-sq-mi.
    """
    jur_proj = jur_gdf.to_crs("EPSG:3310")
    centroids = jur_proj.copy()
    centroids["geometry"] = jur_proj.geometry.centroid

    radius_m = cluster_radius_mi * 1609.34
    circle_area_sq_mi = math.pi * cluster_radius_mi ** 2

    densities = []
    for _, row in centroids.iterrows():
        buf = row.geometry.buffer(radius_m)
        count = centroids[centroids.geometry.within(buf)].shape[0]
        densities.append(round(count / circle_area_sq_mi * 100, 2))

    jur_gdf = jur_gdf.copy()
    jur_gdf["jurisdictional_density"] = densities
    return jur_gdf


def _fetch_osm_edges(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Fetch road edges from OSM via osmnx.  Falls back to a synthetic grid of
    line segments when the Overpass API is unreachable (e.g. sandboxed envs).
    """
    try:
        # osmnx 2.x: bbox = (left, bottom, right, top) = (west, south, east, north)
        graph = ox.graph_from_bbox(
            (bbox[0], bbox[1], bbox[2], bbox[3]),
            network_type="drive",
            simplify=True,
        )
        _, edges = ox.graph_to_gdfs(graph)
        return edges.reset_index()
    except Exception as exc:
        print(f"[spatial_mesh_generator] OSM fetch failed ({exc}); generating synthetic road grid.")
        return _synthetic_road_grid(bbox)


def _synthetic_road_grid(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Generate a regular grid of road-like LineString segments covering the bbox.
    Used when OSM is inaccessible; preserves the full analysis pipeline.
    """
    from shapely.geometry import LineString

    west, south, east, north = bbox
    # ~0.005° spacing ≈ 500 m at this latitude
    step = 0.005
    rows = []
    # Horizontal roads
    lat = south
    seg_id = 0
    while lat <= north:
        lon = west
        while lon + step <= east:
            rows.append({
                "osmid": seg_id,
                "name": f"synth_h_{seg_id}",
                "highway": "residential",
                "geometry": LineString([(lon, lat), (lon + step, lat)]),
            })
            lon += step
            seg_id += 1
        lat += step
    # Vertical roads
    lon = west
    while lon <= east:
        lat = south
        while lat + step <= north:
            rows.append({
                "osmid": seg_id,
                "name": f"synth_v_{seg_id}",
                "highway": "residential",
                "geometry": LineString([(lon, lat), (lon, lat + step)]),
            })
            lat += step
            seg_id += 1
        lon += step
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def run(bbox: tuple[float, float, float, float] = SGV_BBOX, pilot_name: str = "san_gabriel_valley") -> gpd.GeoDataFrame:
    """
    Main entry point.  Returns a GeoDataFrame of road segments annotated
    with boundary-type attributes and writes it to Parquet + GeoJSON.
    """
    print(f"[spatial_mesh_generator] Fetching OSM road network for {pilot_name} …")
    edges = _fetch_osm_edges(bbox)

    print("[spatial_mesh_generator] Building jurisdiction polygons …")
    jur_gdf = _build_jurisdiction_polygons(bbox)
    jur_gdf = _compute_jurisdictional_density(jur_gdf)

    print("[spatial_mesh_generator] Identifying boundary centerlines …")
    roads_annotated = _identify_boundary_centerlines(jur_gdf, edges)

    # Attach schema fields expected by Phase 1
    roads_annotated["boundary_mi"] = roads_annotated.geometry.to_crs("EPSG:3310").length / 1609.34
    roads_annotated["scale_class"] = roads_annotated["left_jurisdiction"].apply(
        lambda x: _classify_scale(x, jur_gdf) if pd.notna(x) else None
    )

    # Persist outputs
    out_parquet = OUTPUT_DIR / f"{pilot_name}_boundary_mesh.parquet"
    out_geojson = OUTPUT_DIR / f"{pilot_name}_boundary_mesh.geojson"
    roads_annotated.to_parquet(out_parquet, index=False)
    roads_annotated.to_file(out_geojson, driver="GeoJSON")

    jur_out = OUTPUT_DIR / f"{pilot_name}_jurisdictions.geojson"
    jur_gdf.to_file(jur_out, driver="GeoJSON")

    boundary_count = roads_annotated["is_boundary"].sum()
    total_count = len(roads_annotated)
    print(
        f"[spatial_mesh_generator] Done. "
        f"{boundary_count}/{total_count} road segments tagged as boundary centerlines."
    )
    print(f"  Outputs → {out_parquet}")
    print(f"           {out_geojson}")
    print(f"           {jur_out}")
    return roads_annotated


def _classify_scale(city_name: str, jur_gdf: gpd.GeoDataFrame) -> str:
    row = jur_gdf[jur_gdf["name"] == city_name]
    if row.empty:
        return "Unknown"
    area = row.iloc[0]["area_sq_mi"]
    if area < 7:
        return "Small"
    if area < 15:
        return "Medium"
    return "Large"


if __name__ == "__main__":
    result = run()
    print(result[["name", "is_boundary", "boundary_mi", "left_jurisdiction", "right_jurisdiction"]].head(20))
