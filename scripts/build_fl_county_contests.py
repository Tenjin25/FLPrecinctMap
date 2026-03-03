#!/usr/bin/env python3
"""
Build county-level contest slices for the FL map app from VEST precinct shapefiles.

Outputs:
  data/contests/{contest_type}_{year}.json
  data/contests/manifest.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import geopandas as gpd
import pandas as pd


VOTE_COL_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{3}[A-Z][A-Z0-9]{3}$")

OFFICE_TO_CONTEST = {
    "PRE": "president",
    "USS": "us_senate",
    "GOV": "governor",
    "LTG": "lieutenant_governor",
    "ATG": "attorney_general",
    "SOS": "secretary_of_state",
    "TRE": "treasurer",
    "CFO": "treasurer",
    "AUD": "auditor",
    "LAB": "labor_commissioner",
    "INS": "insurance_commissioner",
    "AGR": "agriculture_commissioner",
    "SPI": "superintendent",
}


def margin_color(winner: str, margin_abs_pct: float) -> str:
    if winner == "TIE":
        return "#f7f7f7"
    is_r = winner == "REP"
    if margin_abs_pct >= 40:
        return "#67000d" if is_r else "#08306b"
    if margin_abs_pct >= 30:
        return "#a50f15" if is_r else "#08519c"
    if margin_abs_pct >= 20:
        return "#cb181d" if is_r else "#3182bd"
    if margin_abs_pct >= 10:
        return "#ef3b2c" if is_r else "#6baed6"
    if margin_abs_pct >= 5.5:
        return "#fb6a4a" if is_r else "#9ecae1"
    if margin_abs_pct >= 1:
        return "#fcae91" if is_r else "#c6dbef"
    if margin_abs_pct >= 0.5:
        return "#fee8c8" if is_r else "#e1f5fe"
    return "#f7f7f7"


def parse_contest_columns(columns: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    contest_cols: Dict[str, Dict[str, List[str]]] = {}
    for col in columns:
        if col in {"COUNTY", "PRECINCT", "PCT_STD"}:
            continue
        if not VOTE_COL_PATTERN.match(col):
            continue
        office = col[3:6]
        contest = OFFICE_TO_CONTEST.get(office)
        if not contest:
            continue
        party = col[6].upper()
        bucket = "other"
        if party == "D":
            bucket = "dem"
        elif party == "R":
            bucket = "rep"
        contest_cols.setdefault(contest, {"dem": [], "rep": [], "other": []})[bucket].append(col)
    return contest_cols


def sum_numeric(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=df.index)
    return (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        .astype(float)
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def build_county_code_map(shapefile_zip: Path, county_geojson: Path) -> Dict[str, str]:
    precincts = gpd.read_file(shapefile_zip)[["COUNTY", "geometry"]].copy()
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
    return {str(k).strip().upper(): str(v).strip().upper() for k, v in mode_map.items()}


def build() -> None:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    contests_dir = data_dir / "contests"
    contests_dir.mkdir(parents=True, exist_ok=True)

    years = [2014, 2016, 2018, 2020, 2022, 2024]
    county_map = build_county_code_map(
        data_dir / "fl_2024.zip",
        data_dir / "tl_2020_12_county20.geojson",
    )

    manifest_entries = []
    built = 0

    for year in years:
        shp = data_dir / f"fl_{year}.zip"
        if not shp.exists():
            continue
        df = gpd.read_file(shp, ignore_geometry=True)
        if "COUNTY" not in df.columns:
            continue
        df["county_name"] = df["COUNTY"].astype(str).str.strip().str.upper().map(county_map)
        df = df[df["county_name"].notna()]

        contests = parse_contest_columns(df.columns)
        for contest_type, cols in sorted(contests.items()):
            dem = sum_numeric(df, cols["dem"])
            rep = sum_numeric(df, cols["rep"])
            other = sum_numeric(df, cols["other"])
            total = dem + rep + other

            rows = pd.DataFrame(
                {
                    "county": df["county_name"],
                    "dem_votes": dem,
                    "rep_votes": rep,
                    "other_votes": other,
                    "total_votes": total,
                }
            )
            rows = rows.groupby("county", as_index=False)[["dem_votes", "rep_votes", "other_votes", "total_votes"]].sum()
            rows = rows.sort_values("county")
            if rows.empty:
                continue

            payload_rows = []
            for _, r in rows.iterrows():
                tv = float(r["total_votes"])
                dv = float(r["dem_votes"])
                rv = float(r["rep_votes"])
                ov = float(r["other_votes"])
                margin = rv - dv
                margin_pct = (margin / tv * 100.0) if tv > 0 else 0.0
                winner = "REP" if rv > dv else ("DEM" if dv > rv else "TIE")
                payload_rows.append(
                    {
                        "county": r["county"],
                        "dem_votes": round(dv, 3),
                        "rep_votes": round(rv, 3),
                        "other_votes": round(ov, 3),
                        "total_votes": round(tv, 3),
                        "dem_candidate": "Democrat",
                        "rep_candidate": "Republican",
                        "margin": round(margin, 3),
                        "margin_pct": round(margin_pct, 6),
                        "winner": winner,
                        "color": margin_color(winner, abs(margin_pct)),
                    }
                )

            filename = f"{contest_type}_{year}.json"
            write_json(contests_dir / filename, {"rows": payload_rows})
            built += 1

            dem_total = float(rows["dem_votes"].sum())
            rep_total = float(rows["rep_votes"].sum())
            other_total = float(rows["other_votes"].sum())
            manifest_entries.append(
                {
                    "year": year,
                    "contest_type": contest_type,
                    "file": filename,
                    "rows": len(payload_rows),
                    "dem_total": round(dem_total, 3),
                    "rep_total": round(rep_total, 3),
                    "other_total": round(other_total, 3),
                    "major_party_contested": bool(dem_total > 0 and rep_total > 0),
                }
            )

    manifest_entries.sort(key=lambda e: (e["contest_type"], int(e["year"])))
    write_json(contests_dir / "manifest.json", {"files": manifest_entries})

    print(json.dumps({"contests_built": built, "manifest_entries": len(manifest_entries)}, indent=2))


if __name__ == "__main__":
    build()
