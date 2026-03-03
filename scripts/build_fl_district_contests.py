#!/usr/bin/env python3
"""
Build Florida district contest slices from VEST precinct shapefiles.

Outputs:
  data/district_contests/{scope}_{contest_type}_{year}.json
  data/district_contests/manifest.json

Allocation modes:
  1) block            - blockfile + block->district crosswalks (recommended)
  2) precinct_weights - precomputed precinct->district weights
  3) spatial          - geometry intersection fallback
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import pandas as pd


SCOPES = ("congressional", "state_house", "state_senate")

DISTRICT_GEOJSON_DEFAULTS = {
    "congressional": "fl_congressional_districts.geojson",
    "state_house": "fl_state_house_districts.geojson",
    "state_senate": "fl_state_senate_districts.geojson",
}

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

DISTRICT_COL_ALIASES = {
    "congressional": [
        "DISTRICT",
        "district",
        "district_id",
        "CD",
        "CD116FP",
        "CD117FP",
        "CD118FP",
        "congressional_district",
    ],
    "state_house": [
        "DISTRICT",
        "district",
        "district_id",
        "SLDLST",
        "house_district",
        "state_house_district",
        "hd",
    ],
    "state_senate": [
        "DISTRICT",
        "district",
        "district_id",
        "SLDUST",
        "senate_district",
        "state_senate_district",
        "sd",
    ],
}

PCT_COL_ALIASES = ["PCT_STD", "pct_std", "pct_key", "precinct_key", "PRECINCT_KEY"]
COUNTY_COL_ALIASES = ["COUNTY", "county", "county_code", "county_fips"]
PRECINCT_COL_ALIASES = ["PRECINCT", "precinct", "prec", "vtd", "vtdid"]
YEAR_COL_ALIASES = ["year", "YEAR", "election_year", "ElectionYear"]
BLOCK_COL_ALIASES = [
    "block_geoid",
    "BLOCK_GEOID",
    "block",
    "BLOCK",
    "blk2020ge",
    "blk2010ge",
    "GEOID20",
    "GEOID10",
]
WEIGHT_COL_ALIASES = ["weight", "WEIGHT", "wt", "WT", "share", "SHARE"]
BLOCK_WEIGHT_COL_ALIASES = ["block_weight", "BLOCK_WEIGHT", "vap_weight", "pop_weight"]

VOTE_COL_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{3}[A-Z][A-Z0-9]{3}$")


@dataclass
class YearScopeWeights:
    scope: str
    year: int
    weights: pd.DataFrame  # columns: pct_key, district, weight
    source: str


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip().upper()
    return re.sub(r"\s+", "", s)


def normalize_geoid(value: object) -> str:
    s = normalize_text(value)
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    return digits or s


def detect_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in df.columns}
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit:
            return hit
    return None


def read_csv_maybe_zipped(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [
                n
                for n in zf.namelist()
                if n.lower().endswith(".csv") or n.lower().endswith(".txt")
            ]
            if not names:
                raise ValueError(f"No CSV/TXT table found inside zip: {path}")
            with zf.open(names[0]) as fh:
                sep = "|" if names[0].lower().endswith(".txt") else ","
                return pd.read_csv(fh, sep=sep, low_memory=False)
    sep = "|" if path.suffix.lower() == ".txt" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def build_pct_key(
    df: pd.DataFrame,
    pct_col: Optional[str] = None,
    county_col: Optional[str] = None,
    precinct_col: Optional[str] = None,
) -> pd.Series:
    pct_col = pct_col or detect_column(df, PCT_COL_ALIASES)
    if pct_col:
        return df[pct_col].map(normalize_text)
    county_col = county_col or detect_column(df, COUNTY_COL_ALIASES)
    precinct_col = precinct_col or detect_column(df, PRECINCT_COL_ALIASES)
    if not county_col or not precinct_col:
        raise ValueError("Could not find precinct key columns (PCT_STD or COUNTY+PRECINCT).")
    return (
        df[county_col].map(normalize_text)
        + df[precinct_col].map(lambda v: f"-{normalize_text(v)}")
    )


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


def load_precinct_contest_rows(shapefile_path: Path, year: int) -> Dict[str, pd.DataFrame]:
    gdf = gpd.read_file(shapefile_path, ignore_geometry=True)
    pct_key = build_pct_key(gdf)
    contest_cols = parse_contest_columns(gdf.columns)
    out: Dict[str, pd.DataFrame] = {}
    for contest, colset in contest_cols.items():
        dem = sum_numeric(gdf, colset["dem"])
        rep = sum_numeric(gdf, colset["rep"])
        other = sum_numeric(gdf, colset["other"])
        total = dem + rep + other
        out[contest] = pd.DataFrame(
            {
                "year": year,
                "pct_key": pct_key,
                "contest_type": contest,
                "dem_votes": dem,
                "rep_votes": rep,
                "other_votes": other,
                "total_votes": total,
            }
        )
    return out


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["pct_key"] = w["pct_key"].map(normalize_text)
    w["district"] = w["district"].map(lambda v: str(v).strip())
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    w = w[(w["pct_key"] != "") & (w["district"] != "") & (w["weight"] > 0)]
    w = w.groupby(["pct_key", "district"], as_index=False)["weight"].sum()
    totals = w.groupby("pct_key")["weight"].transform("sum")
    w["weight"] = w["weight"] / totals
    return w


def load_precinct_weight_csv(
    csv_path: Path,
    year: int,
    scope: str,
) -> YearScopeWeights:
    df = read_csv_maybe_zipped(csv_path)
    year_col = detect_column(df, YEAR_COL_ALIASES)
    if year_col:
        y = pd.to_numeric(df[year_col], errors="coerce")
        df = df[y == year]
    if df.empty:
        raise ValueError(f"No rows for year {year} in {csv_path}")
    district_col = detect_column(df, DISTRICT_COL_ALIASES[scope])
    if not district_col:
        raise ValueError(f"Could not find district column in {csv_path} for scope={scope}")
    weight_col = detect_column(df, WEIGHT_COL_ALIASES) or "weight"
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    pct_key = build_pct_key(df)
    w = pd.DataFrame(
        {
            "pct_key": pct_key,
            "district": df[district_col],
            "weight": df[weight_col],
        }
    )
    return YearScopeWeights(scope=scope, year=year, weights=normalize_weights(w), source=f"precinct_weights:{csv_path.name}")


def read_nhgis_2010_to_2020_crosswalk(path: Path) -> pd.DataFrame:
    df = read_csv_maybe_zipped(path)
    col_2010 = detect_column(df, ["blk2010ge", "blk2010gj", "block2010", "geoid10"])
    col_2020 = detect_column(df, ["blk2020ge", "blk2020gj", "block2020", "geoid20"])
    col_w = detect_column(df, WEIGHT_COL_ALIASES)
    if not col_2010 or not col_2020:
        raise ValueError(f"NHGIS crosswalk missing 2010/2020 block columns: {path}")
    if not col_w:
        df["__w__"] = 1.0
        col_w = "__w__"
    out = pd.DataFrame(
        {
            "block2010": df[col_2010].map(normalize_geoid),
            "block2020": df[col_2020].map(normalize_geoid),
            "xw_weight": pd.to_numeric(df[col_w], errors="coerce").fillna(0.0),
        }
    )
    out = out[(out["block2010"] != "") & (out["block2020"] != "") & (out["xw_weight"] > 0)]
    return out


def load_block_based_weights(
    scope: str,
    year: int,
    blockfile_path: Path,
    district_block_xwalk_path: Path,
    block_id_year: int,
    nhgis_2010_to_2020: Optional[pd.DataFrame],
) -> YearScopeWeights:
    blocks = read_csv_maybe_zipped(blockfile_path)
    year_col = detect_column(blocks, YEAR_COL_ALIASES)
    if year_col:
        y = pd.to_numeric(blocks[year_col], errors="coerce")
        blocks = blocks[y == year]
    if blocks.empty:
        raise ValueError(f"No block rows for year {year} in {blockfile_path}")

    block_col = detect_column(blocks, BLOCK_COL_ALIASES)
    if not block_col:
        raise ValueError(f"Missing block id column in {blockfile_path}")
    pct_key = build_pct_key(blocks)

    block_weight_col = detect_column(blocks, BLOCK_WEIGHT_COL_ALIASES) or detect_column(blocks, WEIGHT_COL_ALIASES)
    if block_weight_col:
        b_weight = pd.to_numeric(blocks[block_weight_col], errors="coerce").fillna(0.0)
    else:
        b_weight = pd.Series(1.0, index=blocks.index)

    tmp = pd.DataFrame(
        {
            "pct_key": pct_key,
            "block_raw": blocks[block_col].map(normalize_geoid),
            "block_weight": b_weight,
        }
    )
    tmp = tmp[(tmp["pct_key"] != "") & (tmp["block_raw"] != "") & (tmp["block_weight"] > 0)]
    if tmp.empty:
        raise ValueError(f"No usable block rows in {blockfile_path}")

    if block_id_year == 2010:
        if nhgis_2010_to_2020 is None:
            raise ValueError("block_id_year=2010 requires --nhgis-2010-2020 path.")
        tmp = tmp.rename(columns={"block_raw": "block2010"})
        tmp = tmp.merge(nhgis_2010_to_2020, on="block2010", how="left")
        tmp = tmp[tmp["block2020"].notna() & (tmp["xw_weight"] > 0)]
        tmp["block_weight"] = tmp["block_weight"] * tmp["xw_weight"]
        tmp["block2020"] = tmp["block2020"].map(normalize_geoid)
    else:
        tmp["block2020"] = tmp["block_raw"].map(normalize_geoid)

    dbx = read_csv_maybe_zipped(district_block_xwalk_path)
    db_block_col = detect_column(dbx, BLOCK_COL_ALIASES)
    db_dist_col = detect_column(dbx, DISTRICT_COL_ALIASES[scope])
    if not db_block_col or not db_dist_col:
        raise ValueError(
            f"Missing block/district columns in district crosswalk: {district_block_xwalk_path}"
        )
    db_w_col = detect_column(dbx, WEIGHT_COL_ALIASES)
    if not db_w_col:
        dbx["__w__"] = 1.0
        db_w_col = "__w__"

    dbx = pd.DataFrame(
        {
            "block2020": dbx[db_block_col].map(normalize_geoid),
            "district": dbx[db_dist_col],
            "district_weight": pd.to_numeric(dbx[db_w_col], errors="coerce").fillna(0.0),
        }
    )
    dbx = dbx[(dbx["block2020"] != "") & (dbx["district_weight"] > 0)]

    merged = tmp.merge(dbx, on="block2020", how="left")
    merged = merged[merged["district"].notna() & (merged["district_weight"] > 0)]
    if merged.empty:
        raise ValueError(
            f"No block overlaps between {blockfile_path.name} and {district_block_xwalk_path.name}"
        )
    merged["weight"] = merged["block_weight"] * merged["district_weight"]
    w = merged.groupby(["pct_key", "district"], as_index=False)["weight"].sum()
    return YearScopeWeights(
        scope=scope,
        year=year,
        weights=normalize_weights(w),
        source=f"block:{blockfile_path.name}+{district_block_xwalk_path.name}",
    )


def load_spatial_weights(
    scope: str,
    year: int,
    precinct_shapefile_path: Path,
    district_geojson_path: Path,
) -> YearScopeWeights:
    precincts = gpd.read_file(precinct_shapefile_path)
    districts = gpd.read_file(district_geojson_path)
    dcol = detect_column(districts, DISTRICT_COL_ALIASES[scope])
    if not dcol:
        raise ValueError(f"No district id column found in {district_geojson_path}")
    precincts = precincts.copy()
    precincts["pct_key"] = build_pct_key(precincts)
    precincts = precincts[(precincts["pct_key"] != "") & precincts.geometry.notna()][["pct_key", "geometry"]]
    districts = districts[districts.geometry.notna()][[dcol, "geometry"]].rename(columns={dcol: "district"})

    if precincts.crs is None:
        precincts = precincts.set_crs(4326)
    if districts.crs is None:
        districts = districts.set_crs(4326)
    if precincts.crs != districts.crs:
        districts = districts.to_crs(precincts.crs)

    metric_crs = "EPSG:3857"
    p = precincts.to_crs(metric_crs)
    d = districts.to_crs(metric_crs)

    p["prec_area"] = p.geometry.area
    inter = gpd.overlay(p[["pct_key", "geometry"]], d[["district", "geometry"]], how="intersection", keep_geom_type=False)
    if inter.empty:
        raise ValueError(f"Spatial intersection produced no overlaps for {scope} {year}")
    inter["inter_area"] = inter.geometry.area
    inter = inter.merge(p[["pct_key", "prec_area"]], on="pct_key", how="left")
    inter = inter[inter["prec_area"] > 0]
    inter["weight"] = inter["inter_area"] / inter["prec_area"]
    w = inter.groupby(["pct_key", "district"], as_index=False)["weight"].sum()
    return YearScopeWeights(scope=scope, year=year, weights=normalize_weights(w), source=f"spatial:{district_geojson_path.name}")


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


def aggregate_contest_to_district(
    precinct_rows: pd.DataFrame,
    weights: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    votes = precinct_rows.copy()
    votes["pct_key"] = votes["pct_key"].map(normalize_text)

    matched_keys = set(weights["pct_key"].unique())
    matched_mask = votes["pct_key"].isin(matched_keys)
    total_votes = float(votes["total_votes"].sum())
    matched_votes = float(votes.loc[matched_mask, "total_votes"].sum())
    coverage_pct = (matched_votes / total_votes * 100.0) if total_votes > 0 else 0.0

    merged = votes.merge(weights, on="pct_key", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["district", "dem_votes", "rep_votes", "other_votes", "total_votes"]), 0.0

    for col in ("dem_votes", "rep_votes", "other_votes", "total_votes"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0) * merged["weight"]

    agg = (
        merged.groupby("district", as_index=False)[["dem_votes", "rep_votes", "other_votes", "total_votes"]]
        .sum()
        .sort_values("district")
    )
    return agg, coverage_pct


def district_result_row(dem: float, rep: float, other: float, total: float) -> Dict[str, object]:
    total = float(total)
    dem = float(dem)
    rep = float(rep)
    other = float(other)
    margin = rep - dem
    margin_pct = (margin / total * 100.0) if total > 0 else 0.0
    winner = "REP" if rep > dem else ("DEM" if dem > rep else "TIE")
    return {
        "dem_votes": round(dem, 3),
        "rep_votes": round(rep, 3),
        "other_votes": round(other, 3),
        "total_votes": round(total, 3),
        "margin": round(margin, 3),
        "margin_pct": round(margin_pct, 6),
        "winner": winner,
        "dem_candidate": "Democrat",
        "rep_candidate": "Republican",
        "color": margin_color(winner, abs(margin_pct)),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def parse_years(years_arg: str) -> List[int]:
    years = []
    for tok in years_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        years.append(int(tok))
    return sorted(set(years))


def path_or_none(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    return Path(path_text).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FL district contest slices from VEST shapefiles.")
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument("--years", default="2014,2016,2018,2020,2022,2024")
    parser.add_argument("--scopes", default="congressional,state_house,state_senate")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <data-dir>/district_contests)")
    parser.add_argument(
        "--allocation-method",
        choices=["block", "precinct_weights", "spatial"],
        default="block",
    )
    parser.add_argument(
        "--shapefile-template",
        default="fl_{year}.zip",
        help="Template under data-dir for yearly VEST shapefile.",
    )

    parser.add_argument("--congressional-weights", default=None)
    parser.add_argument("--state-house-weights", default=None)
    parser.add_argument("--state-senate-weights", default=None)

    parser.add_argument(
        "--blockfile-template",
        default="blockfiles/fl_blocks_{year}.csv",
        help="Template for year blockfile (under data-dir unless absolute).",
    )
    parser.add_argument("--congressional-block-crosswalk", default=None)
    parser.add_argument("--state-house-block-crosswalk", default=None)
    parser.add_argument("--state-senate-block-crosswalk", default=None)
    parser.add_argument("--block-id-year", type=int, choices=[2010, 2020], default=2020)
    parser.add_argument("--nhgis-2010-2020", default="nhgis_blk2010_blk2020_12.zip")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (data_dir / "district_contests").resolve()
    )
    years = parse_years(args.years)
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    bad_scopes = [s for s in scopes if s not in SCOPES]
    if bad_scopes:
        print(f"Invalid scopes: {bad_scopes}", file=sys.stderr)
        return 2

    precinct_weight_paths = {
        "congressional": path_or_none(args.congressional_weights),
        "state_house": path_or_none(args.state_house_weights),
        "state_senate": path_or_none(args.state_senate_weights),
    }
    block_xwalk_paths = {
        "congressional": path_or_none(args.congressional_block_crosswalk),
        "state_house": path_or_none(args.state_house_block_crosswalk),
        "state_senate": path_or_none(args.state_senate_block_crosswalk),
    }

    nhgis_df = None
    if args.allocation_method == "block" and args.block_id_year == 2010:
        nhgis_path = path_or_none(args.nhgis_2010_2020)
        if nhgis_path is None:
            nhgis_path = (data_dir / "nhgis_blk2010_blk2020_12.zip").resolve()
        nhgis_df = read_nhgis_2010_to_2020_crosswalk(nhgis_path)

    manifest_entries: List[dict] = []
    written = 0

    for year in years:
        shp_path = Path(args.shapefile_template.format(year=year))
        if not shp_path.is_absolute():
            shp_path = (data_dir / shp_path).resolve()
        if not shp_path.exists():
            print(f"[WARN] Missing shapefile for {year}: {shp_path}")
            continue

        precinct_contests = load_precinct_contest_rows(shp_path, year)
        if not precinct_contests:
            print(f"[WARN] No contest columns found in {shp_path.name}")
            continue

        for scope in scopes:
            try:
                if args.allocation_method == "precinct_weights":
                    p = precinct_weight_paths[scope]
                    if not p:
                        raise ValueError(f"Missing --{scope.replace('_', '-')}-weights argument.")
                    weights_obj = load_precinct_weight_csv(p, year=year, scope=scope)
                elif args.allocation_method == "block":
                    xw = block_xwalk_paths[scope]
                    if not xw:
                        raise ValueError(
                            f"Missing --{scope.replace('_', '-')}-block-crosswalk argument."
                        )
                    bf = Path(args.blockfile_template.format(year=year))
                    if not bf.is_absolute():
                        bf = (data_dir / bf).resolve()
                    weights_obj = load_block_based_weights(
                        scope=scope,
                        year=year,
                        blockfile_path=bf,
                        district_block_xwalk_path=xw,
                        block_id_year=args.block_id_year,
                        nhgis_2010_to_2020=nhgis_df,
                    )
                else:
                    dpath = (data_dir / DISTRICT_GEOJSON_DEFAULTS[scope]).resolve()
                    weights_obj = load_spatial_weights(
                        scope=scope,
                        year=year,
                        precinct_shapefile_path=shp_path,
                        district_geojson_path=dpath,
                    )
            except Exception as exc:
                print(f"[WARN] {scope} {year}: failed to load weights ({exc})")
                continue

            for contest_type, contest_rows in sorted(precinct_contests.items()):
                agg, coverage_pct = aggregate_contest_to_district(contest_rows, weights_obj.weights)
                if agg.empty:
                    continue

                result_map: Dict[str, dict] = {}
                for _, row in agg.iterrows():
                    dist_key = str(row["district"]).strip()
                    result_map[dist_key] = district_result_row(
                        row["dem_votes"],
                        row["rep_votes"],
                        row["other_votes"],
                        row["total_votes"],
                    )

                dem_total = float(agg["dem_votes"].sum())
                rep_total = float(agg["rep_votes"].sum())
                other_total = float(agg["other_votes"].sum())

                payload = {
                    "meta": {
                        "scope": scope,
                        "contest_type": contest_type,
                        "year": year,
                        "source": "VEST precinct shapefile allocation",
                        "allocation_method": args.allocation_method,
                        "allocation_source": weights_obj.source,
                        "match_coverage_pct": round(coverage_pct, 6),
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    "general": {
                        "results": result_map,
                    },
                }

                filename = f"{scope}_{contest_type}_{year}.json"
                out_path = out_dir / filename
                if not args.dry_run:
                    write_json(out_path, payload)
                    written += 1

                manifest_entries.append(
                    {
                        "scope": scope,
                        "contest_type": contest_type,
                        "year": year,
                        "file": filename,
                        "rows": len(result_map),
                        "dem_total": round(dem_total, 3),
                        "rep_total": round(rep_total, 3),
                        "other_total": round(other_total, 3),
                        "major_party_contested": bool(dem_total > 0 and rep_total > 0),
                        "match_coverage_pct": round(coverage_pct, 6),
                    }
                )

    manifest_entries.sort(key=lambda r: (r["scope"], r["contest_type"], int(r["year"])))
    manifest_payload = {"files": manifest_entries}
    manifest_path = out_dir / "manifest.json"
    if not args.dry_run:
        write_json(manifest_path, manifest_payload)

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "allocation_method": args.allocation_method,
                "slices_built": len(manifest_entries),
                "files_written": written + (0 if args.dry_run else 1),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
