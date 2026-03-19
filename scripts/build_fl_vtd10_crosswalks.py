#!/usr/bin/env python3
"""
Build Florida VTD2010 crosswalks (useful for 2012/2014 reconciliation).

Inputs (defaults under data/):
  - nhgis_blk2010_blk2020_12.zip
  - tl_2020_12_tabblock10.zip
  - tl_2020_12_tabblock20.zip
  - tl_2012_12_vtd10.zip
  - tl_2020_12_vtd20.zip
  - fl_congressional_districts.geojson
  - fl_proposed_congressional_districts.geojson
  - fl_state_house_districts.geojson
  - fl_state_senate_districts.geojson

Outputs (under data/crosswalks by default):
  - vtd10_to_vtd20_weights.csv
  - vtd10_to_congressional_current_weights.csv
  - vtd10_to_congressional_proposed_weights.csv
  - vtd10_to_state_house_weights.csv
  - vtd10_to_state_senate_weights.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import pandas as pd


def normalize_geoid(value: object, width: int = 15) -> str:
    s = "" if value is None else str(value).strip()
    digits = re.sub(r"\D", "", s)
    return digits.zfill(width) if digits else ""


def read_nhgis_crosswalk(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name:
            raise ValueError(f"No CSV found in {path}")
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, low_memory=False)
    if "blk2010ge" not in df.columns or "blk2020ge" not in df.columns or "weight" not in df.columns:
        raise ValueError("NHGIS file missing required columns: blk2010ge, blk2020ge, weight")
    out = pd.DataFrame(
        {
            "blk2010ge": df["blk2010ge"].map(normalize_geoid),
            "blk2020ge": df["blk2020ge"].map(normalize_geoid),
            "weight": pd.to_numeric(df["weight"], errors="coerce").fillna(0.0),
        }
    )
    out = out[(out["blk2010ge"] != "") & (out["blk2020ge"] != "") & (out["weight"] > 0)]
    return out


def read_geo(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def map_blocks_to_target(
    blocks: gpd.GeoDataFrame,
    block_geoid_col: str,
    target: gpd.GeoDataFrame,
    target_id_col: str,
    target_out_col: str,
) -> pd.DataFrame:
    target_join_col = "__target_id__"
    b = blocks[[block_geoid_col, "geometry"]].copy()
    b = b[b[block_geoid_col].notna() & b.geometry.notna()]
    b[block_geoid_col] = b[block_geoid_col].map(normalize_geoid)
    b = b[b[block_geoid_col] != ""]

    t = target[[target_id_col, "geometry"]].copy()
    t = t[t[target_id_col].notna() & t.geometry.notna()]
    t = t.rename(columns={target_id_col: target_join_col})

    # Use representative points to avoid heavy polygon overlay for large block layers.
    b["geometry"] = b.geometry.representative_point()
    joined = gpd.sjoin(b, t, how="left", predicate="within")
    if joined[target_join_col].isna().any():
        missing = joined[joined[target_join_col].isna()][[block_geoid_col, "geometry"]]
        if not missing.empty:
            fallback = gpd.sjoin(missing, t, how="left", predicate="intersects")
            fallback = (
                fallback[[block_geoid_col, target_join_col]]
                .dropna()
                .drop_duplicates(subset=[block_geoid_col])
            )
            joined = joined.merge(
                fallback,
                on=block_geoid_col,
                how="left",
                suffixes=("", "_fallback"),
            )
            joined[target_join_col] = joined[target_join_col].fillna(joined[f"{target_join_col}_fallback"])
            joined = joined.drop(columns=[f"{target_join_col}_fallback"])

    out = joined[[block_geoid_col, target_join_col]].dropna().drop_duplicates()
    out = out.rename(columns={block_geoid_col: "block_geoid", target_join_col: target_out_col})
    out[target_out_col] = out[target_out_col].map(lambda v: str(v).strip())
    out = out[out[target_out_col] != ""]
    return out


def build_weight_table(
    nhgis_df: pd.DataFrame,
    from_map: pd.DataFrame,
    to_map: pd.DataFrame,
    from_col: str,
    to_col: str,
) -> pd.DataFrame:
    x = nhgis_df.merge(
        from_map.rename(columns={"block_geoid": "blk2010ge"}),
        on="blk2010ge",
        how="inner",
    )
    x = x.merge(
        to_map.rename(columns={"block_geoid": "blk2020ge"}),
        on="blk2020ge",
        how="inner",
    )
    if x.empty:
        return pd.DataFrame(columns=[from_col, to_col, "weight"])

    x["weight"] = pd.to_numeric(x["weight"], errors="coerce").fillna(0.0)
    x = x[x["weight"] > 0]
    agg = x.groupby([from_col, to_col], as_index=False)["weight"].sum()
    totals = agg.groupby(from_col)["weight"].transform("sum")
    agg = agg[totals > 0].copy()
    agg["weight"] = agg["weight"] / totals[totals > 0]
    agg = agg.sort_values([from_col, to_col]).reset_index(drop=True)
    return agg


def write_csv(path: Path, rows: pd.DataFrame, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for _, row in rows.iterrows():
            out = {}
            for c in columns:
                v = row[c]
                if c == "weight":
                    out[c] = f"{float(v):.10f}"
                else:
                    out[c] = v
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FL VTD2010 crosswalk tables.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="crosswalks", help="Output dir under data-dir unless absolute.")
    parser.add_argument("--nhgis", default="nhgis_blk2010_blk2020_12.zip")
    parser.add_argument("--tabblock10", default="tl_2020_12_tabblock10.zip")
    parser.add_argument("--tabblock20", default="tl_2020_12_tabblock20.zip")
    parser.add_argument("--vtd10", default="tl_2012_12_vtd10.zip")
    parser.add_argument("--vtd20", default="tl_2020_12_vtd20.zip")
    parser.add_argument("--congressional-current", default="fl_congressional_districts.geojson")
    parser.add_argument("--congressional-proposed", default="fl_proposed_congressional_districts.geojson")
    parser.add_argument("--state-house", default="fl_state_house_districts.geojson")
    parser.add_argument("--state-senate", default="fl_state_senate_districts.geojson")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (data_dir / out_dir).resolve()

    def p(rel: str) -> Path:
        path = Path(rel).expanduser()
        if not path.is_absolute():
            path = (data_dir / path).resolve()
        return path

    nhgis_path = p(args.nhgis)
    tabblock10_path = p(args.tabblock10)
    tabblock20_path = p(args.tabblock20)
    vtd10_path = p(args.vtd10)
    vtd20_path = p(args.vtd20)
    cd_current_path = p(args.congressional_current)
    cd_proposed_path = p(args.congressional_proposed)
    sh_path = p(args.state_house)
    ss_path = p(args.state_senate)

    nhgis_df = read_nhgis_crosswalk(nhgis_path)
    block10 = read_geo(tabblock10_path)
    block20 = read_geo(tabblock20_path)
    vtd10 = read_geo(vtd10_path)
    vtd20 = read_geo(vtd20_path)
    cd_current = read_geo(cd_current_path)
    cd_proposed = read_geo(cd_proposed_path)
    state_house = read_geo(sh_path)
    state_senate = read_geo(ss_path)

    b10_to_vtd10 = map_blocks_to_target(block10, "GEOID10", vtd10, "GEOID10", "from_vtd10")
    b20_to_vtd20 = map_blocks_to_target(block20, "GEOID20", vtd20, "GEOID20", "to_vtd20")
    b20_to_cd_current = map_blocks_to_target(block20, "GEOID20", cd_current, "DISTRICT", "district")
    b20_to_cd_proposed = map_blocks_to_target(block20, "GEOID20", cd_proposed, "DISTRICT", "district")
    b20_to_state_house = map_blocks_to_target(block20, "GEOID20", state_house, "DISTRICT", "district")
    b20_to_state_senate = map_blocks_to_target(block20, "GEOID20", state_senate, "DISTRICT", "district")

    vtd10_to_vtd20 = build_weight_table(nhgis_df, b10_to_vtd10, b20_to_vtd20, "from_vtd10", "to_vtd20")
    vtd10_to_cd_current = build_weight_table(
        nhgis_df, b10_to_vtd10, b20_to_cd_current, "from_vtd10", "district"
    )
    vtd10_to_cd_proposed = build_weight_table(
        nhgis_df, b10_to_vtd10, b20_to_cd_proposed, "from_vtd10", "district"
    )
    vtd10_to_state_house = build_weight_table(
        nhgis_df, b10_to_vtd10, b20_to_state_house, "from_vtd10", "district"
    )
    vtd10_to_state_senate = build_weight_table(
        nhgis_df, b10_to_vtd10, b20_to_state_senate, "from_vtd10", "district"
    )

    write_csv(out_dir / "vtd10_to_vtd20_weights.csv", vtd10_to_vtd20, ["from_vtd10", "to_vtd20", "weight"])
    write_csv(
        out_dir / "vtd10_to_congressional_current_weights.csv",
        vtd10_to_cd_current,
        ["from_vtd10", "district", "weight"],
    )
    write_csv(
        out_dir / "vtd10_to_congressional_proposed_weights.csv",
        vtd10_to_cd_proposed,
        ["from_vtd10", "district", "weight"],
    )
    write_csv(
        out_dir / "vtd10_to_state_house_weights.csv",
        vtd10_to_state_house,
        ["from_vtd10", "district", "weight"],
    )
    write_csv(
        out_dir / "vtd10_to_state_senate_weights.csv",
        vtd10_to_state_senate,
        ["from_vtd10", "district", "weight"],
    )

    summary = {
        "output_dir": str(out_dir),
        "from_vtd10_count": int(vtd10_to_vtd20["from_vtd10"].nunique()) if not vtd10_to_vtd20.empty else 0,
        "vtd10_to_vtd20_rows": int(len(vtd10_to_vtd20)),
        "vtd10_to_cd_current_rows": int(len(vtd10_to_cd_current)),
        "vtd10_to_cd_proposed_rows": int(len(vtd10_to_cd_proposed)),
        "vtd10_to_state_house_rows": int(len(vtd10_to_state_house)),
        "vtd10_to_state_senate_rows": int(len(vtd10_to_state_senate)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
