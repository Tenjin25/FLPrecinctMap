#!/usr/bin/env python3
"""
Build Florida precinct centroid GeoJSON for map hover/search fallback.

Output:
  data/fl_precinct_centroids.geojson

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


def normalize_precinct_norm(county_name: str, prec_id: str) -> str:
    raw = f"{county_name} - {prec_id}"
    raw = re.sub(r"[^A-Z0-9 .\-]", "", raw.upper())
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


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

    if not precinct_path.exists():
        print(f"Missing precinct shapefile: {precinct_path}", file=sys.stderr)
        return 2
    if not county_geojson.exists():
        print(f"Missing county geojson: {county_geojson}", file=sys.stderr)
        return 2

    county_code_map = build_county_code_map(precinct_path, county_geojson)

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
    gdf["prec_id"] = gdf["PRECINCT"].map(lambda v: str(v).strip())
    gdf["pct_std"] = gdf["PCT_STD"].map(str).str.strip() if "PCT_STD" in gdf.columns else ""
    gdf = gdf[gdf["county_nam"].notna() & (gdf["county_nam"] != "") & (gdf["prec_id"] != "")]

    gdf["precinct_norm"] = gdf.apply(
        lambda r: normalize_precinct_norm(str(r["county_nam"]), str(r["prec_id"])),
        axis=1,
    )
    gdf["precinct_name"] = gdf["county_nam"] + " - " + gdf["prec_id"]

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
                },
                "geometry": mapping(geom),
            }
        )

    payload = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "features_written": len(features),
                "source_precinct_rows": int(len(gdf)),
                "county_codes_mapped": int(len(county_code_map)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

