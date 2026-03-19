#!/usr/bin/env python3
"""
Validate proposed congressional-line contest slices against geometry + county totals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Set

import geopandas as gpd


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_district_id(value: object) -> str:
    s = str(value).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_proposed_district_ids(path: Path) -> Set[str]:
    gdf = gpd.read_file(path)
    if "DISTRICT" not in gdf.columns:
        raise ValueError(f"DISTRICT column not found in {path}")
    out = {
        normalize_district_id(v)
        for v in gdf["DISTRICT"].tolist()
        if normalize_district_id(v)
    }
    return out


def contest_totals_from_county_slice(path: Path) -> Dict[str, int]:
    payload = read_json(path)
    rows = payload.get("rows", [])
    dem = sum(int(r.get("dem_votes", 0) or 0) for r in rows)
    rep = sum(int(r.get("rep_votes", 0) or 0) for r in rows)
    other = sum(int(r.get("other_votes", 0) or 0) for r in rows)
    return {"dem": dem, "rep": rep, "other": other, "total": dem + rep + other}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate proposed congressional contest slices.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--proposed-geojson", default="fl_proposed_congressional_districts.geojson")
    parser.add_argument("--district-contests-dir", default="district_contests_proposed_congressional")
    parser.add_argument("--county-contests-dir", default="contests")
    parser.add_argument("--coverage-warn-threshold", type=float, default=98.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    proposed_geojson = (data_dir / args.proposed_geojson).resolve()
    proposed_dir = (data_dir / args.district_contests_dir).resolve()
    county_contests_dir = (data_dir / args.county_contests_dir).resolve()
    manifest_path = proposed_dir / "manifest.json"

    district_ids = load_proposed_district_ids(proposed_geojson)
    manifest = read_json(manifest_path)
    entries = [e for e in (manifest.get("files") or []) if (e or {}).get("scope") == "congressional"]

    errors: List[str] = []
    warnings: List[str] = []
    coverage_values: List[float] = []
    per_file: List[dict] = []

    for e in entries:
        file_name = str(e.get("file") or "")
        contest_type = str(e.get("contest_type") or "")
        year = int(e.get("year") or 0)
        path = proposed_dir / file_name
        if not path.exists():
            errors.append(f"Missing slice file: {file_name}")
            continue

        payload = read_json(path)
        results = ((payload.get("general") or {}).get("results") or {})
        keys = {normalize_district_id(k) for k in results.keys() if normalize_district_id(k)}
        missing_in_geo = sorted(keys - district_ids)
        missing_in_results = sorted(district_ids - keys)
        if missing_in_geo:
            errors.append(f"{file_name}: district ids not in proposed geojson: {missing_in_geo[:5]}")
        if missing_in_results:
            warnings.append(f"{file_name}: missing districts in results: {missing_in_results[:5]}")

        dem = sum(int((v or {}).get("dem_votes", 0) or 0) for v in results.values())
        rep = sum(int((v or {}).get("rep_votes", 0) or 0) for v in results.values())
        other = sum(int((v or {}).get("other_votes", 0) or 0) for v in results.values())
        total = dem + rep + other

        md = int(e.get("dem_total", 0) or 0)
        mr = int(e.get("rep_total", 0) or 0)
        mo = int(e.get("other_total", 0) or 0)
        if (dem, rep, other) != (md, mr, mo):
            warnings.append(
                f"{file_name}: manifest totals differ from slice totals "
                f"(manifest {md}/{mr}/{mo}, slice {dem}/{rep}/{other})"
            )

        coverage = float(((payload.get("meta") or {}).get("match_coverage_pct", 0.0) or 0.0))
        coverage_values.append(coverage)
        if coverage < args.coverage_warn_threshold:
            warnings.append(f"{file_name}: coverage {coverage:.3f}% below {args.coverage_warn_threshold:.1f}%")

        county_path = county_contests_dir / f"{contest_type}_{year}.json"
        county_delta = None
        if county_path.exists():
            ct = contest_totals_from_county_slice(county_path)
            county_delta = {
                "dem": dem - ct["dem"],
                "rep": rep - ct["rep"],
                "other": other - ct["other"],
                "total": total - ct["total"],
            }

        per_file.append(
            {
                "file": file_name,
                "contest_type": contest_type,
                "year": year,
                "district_rows": len(results),
                "coverage_pct": round(coverage, 6),
                "totals": {"dem": dem, "rep": rep, "other": other, "total": total},
                "county_total_delta": county_delta,
            }
        )

    coverage_summary = {}
    if coverage_values:
        coverage_summary = {
            "min": round(min(coverage_values), 6),
            "median": round(median(coverage_values), 6),
            "max": round(max(coverage_values), 6),
        }

    summary = {
        "proposed_geojson": str(proposed_geojson),
        "district_count_geojson": len(district_ids),
        "manifest_entries_checked": len(entries),
        "coverage_summary_pct": coverage_summary,
        "errors": errors,
        "warnings": warnings,
        "files": per_file,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
