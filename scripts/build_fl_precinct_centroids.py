#!/usr/bin/env python3
"""
Build Florida precinct centroid GeoJSON for map hover/search fallback.

Outputs:
  data/fl_precinct_centroids.geojson
  data/fl_voting_precincts.geojson

Source:
  data/fl_2024.zip (VEST precinct shapefile by default)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import geopandas as gpd
from shapely.geometry import mapping


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def normalize_precinct_code(value: object) -> str:
    raw = str(value or "").strip()
    integer_decimal = re.fullmatch(r"(\d+)\.0+", raw)
    return integer_decimal.group(1) if integer_decimal else raw


def normalize_precinct_norm(county_name: str, prec_id: str) -> str:
    raw = f"{county_name} - {prec_id}"
    raw = re.sub(r"[^A-Z0-9 .\-]", "", raw.upper())
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def rounded_geometry_mapping(geometry: object, precision: int = 6) -> dict:
    """Return compact GeoJSON geometry with stable, web-sized coordinates."""

    payload = mapping(geometry)

    def round_coordinates(value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        if value and isinstance(value[0], (int, float)):
            return [round(float(number), precision) for number in value]
        return [round_coordinates(item) for item in value]

    if "coordinates" in payload:
        payload["coordinates"] = round_coordinates(payload["coordinates"])
    return payload


def build_county_code_map(precinct_path: Path, county_geojson: Path) -> Dict[str, str]:
    precincts = gpd.read_file(precinct_path)[["COUNTY", "geometry"]].copy()
    counties = gpd.read_file(county_geojson)[["NAME20", "geometry"]].copy()

    precincts = precincts[precincts["COUNTY"].notna() & precincts.geometry.notna()]
    counties = counties[counties["NAME20"].notna() & counties.geometry.notna()]

    if precincts.crs is None:
        precincts = precincts.set_crs(4326)
    if counties.crs is None:
        counties = counties.set_crs(4326)
    if precincts.crs != counties.crs:
        counties = counties.to_crs(precincts.crs)

    joined = gpd.sjoin(precincts, counties, how="left", predicate="intersects")
    joined = joined[joined["NAME20"].notna()]
    mode_map = (
        joined.groupby("COUNTY")["NAME20"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {normalize_text(k): normalize_text(v) for k, v in mode_map.items()}


def load_friendly_names(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    counties = payload.get("counties") or {}
    return {
        normalize_text(county): {
            normalize_text(code): str(name or "").strip()
            for code, name in code_map.items()
            if normalize_text(code) and str(name or "").strip()
        }
        for county, code_map in counties.items()
        if isinstance(code_map, dict)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FL precinct centroids GeoJSON.")
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument(
        "--precinct-shapefile",
        default="fl_2024.zip",
        help="Precinct shapefile path (under data-dir unless absolute).",
    )
    parser.add_argument(
        "--county-geojson",
        default="tl_2020_12_county20.geojson",
        help="County boundaries used to map COUNTY code to county name.",
    )
    parser.add_argument(
        "--output",
        default="fl_precinct_centroids.geojson",
        help="Output GeoJSON filename (under data-dir unless absolute).",
    )
    parser.add_argument(
        "--polygons-output",
        default="fl_voting_precincts.geojson",
        help="Simplified polygon GeoJSON filename (under data-dir unless absolute).",
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0005,
        help="Polygon simplification tolerance in WGS84 degrees (default: 0.0005).",
    )
    parser.add_argument(
        "--friendly-names",
        default="precinct_friendly_names.json",
        help="NCPrecinctMap-style county/code friendly-name index.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    precinct_path = Path(args.precinct_shapefile).expanduser()
    if not precinct_path.is_absolute():
        precinct_path = (data_dir / precinct_path).resolve()
    county_geojson = Path(args.county_geojson).expanduser()
    if not county_geojson.is_absolute():
        county_geojson = (data_dir / county_geojson).resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (data_dir / output_path).resolve()
    polygons_output_path = Path(args.polygons_output).expanduser()
    if not polygons_output_path.is_absolute():
        polygons_output_path = (data_dir / polygons_output_path).resolve()
    friendly_names_path = Path(args.friendly_names).expanduser()
    if not friendly_names_path.is_absolute():
        friendly_names_path = (data_dir / friendly_names_path).resolve()

    if not precinct_path.exists():
        print(f"Missing precinct shapefile: {precinct_path}", file=sys.stderr)
        return 2
    if not county_geojson.exists():
        print(f"Missing county geojson: {county_geojson}", file=sys.stderr)
        return 2

    county_code_map = build_county_code_map(precinct_path, county_geojson)
    friendly_names = load_friendly_names(friendly_names_path)

    gdf = gpd.read_file(precinct_path)
    for col in ("COUNTY", "PRECINCT"):
        if col not in gdf.columns:
            print(f"Missing required column {col} in {precinct_path.name}", file=sys.stderr)
            return 2

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.to_crs(4326)

    gdf["county_code"] = gdf["COUNTY"].map(normalize_text)
    gdf["county_nam"] = gdf["county_code"].map(county_code_map)
    gdf["prec_id"] = gdf["PRECINCT"].map(normalize_precinct_code)
    gdf["pct_std"] = gdf["PCT_STD"].map(str).str.strip() if "PCT_STD" in gdf.columns else ""
    gdf = gdf[gdf["county_nam"].notna() & (gdf["county_nam"] != "") & (gdf["prec_id"] != "")]

    gdf["precinct_norm"] = gdf.apply(
        lambda r: normalize_precinct_norm(str(r["county_nam"]), str(r["prec_id"])),
        axis=1,
    )
    gdf["precinct_name"] = gdf["county_nam"] + " - " + gdf["prec_id"]
    gdf["precinct_full_name"] = gdf.apply(
        lambda r: friendly_names.get(normalize_text(r["county_nam"]), {}).get(
            normalize_text(r["prec_id"]),
            "",
        ),
        axis=1,
    )

    points = gdf.copy()
    points["geometry"] = points.geometry.representative_point()

    features = []
    for _, row in points.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "county_nam": str(row["county_nam"]),
                    "county_code": str(row["county_code"]),
                    "prec_id": str(row["prec_id"]),
                    "PCT_STD": str(row["pct_std"]),
                    "precinct_name": str(row["precinct_name"]),
                    "precinct_norm": str(row["precinct_norm"]),
                    "precinct_full_name": str(row["precinct_full_name"]),
                },
                "geometry": rounded_geometry_mapping(geom),
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    polygon_features = []
    simplify_tolerance = max(0.0, float(args.simplify_tolerance))
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        if geom is None or geom.is_empty:
            continue
        polygon_features.append(
            {
                "type": "Feature",
                "properties": {
                    "county_nam": str(row["county_nam"]),
                    "county_code": str(row["county_code"]),
                    "prec_id": str(row["prec_id"]),
                    "PCT_STD": str(row["pct_std"]),
                    "precinct_name": str(row["precinct_name"]),
                    "precinct_norm": str(row["precinct_norm"]),
                    "precinct_full_name": str(row["precinct_full_name"]),
                },
                "geometry": rounded_geometry_mapping(geom),
            }
        )

    polygons_payload = {"type": "FeatureCollection", "features": polygon_features}
    polygons_output_path.parent.mkdir(parents=True, exist_ok=True)
    polygons_output_path.write_text(
        json.dumps(polygons_payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "features_written": len(features),
                "polygons_output": str(polygons_output_path),
                "polygon_features_written": len(polygon_features),
                "simplify_tolerance": simplify_tolerance,
                "source_precinct_rows": int(len(gdf)),
                "county_codes_mapped": int(len(county_code_map)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
