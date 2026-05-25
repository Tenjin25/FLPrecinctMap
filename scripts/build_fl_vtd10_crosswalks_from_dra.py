#!/usr/bin/env python3
"""
Build Florida VTD2010 crosswalks using DRA's 2020 VTD GeoJSON from GitHub.

This pipeline makes the existing crosswalk build reproducible inside this
repository by:
  1) downloading the missing Census/TIGER geometry inputs when needed,
  2) downloading DRA's corrected Florida 2020 VTD GeoJSON from GitHub,
  3) emitting the same weight tables the app already consumes.

Outputs (under data/crosswalks by default):
  - vtd10_to_vtd20_weights.csv
  - vtd10_to_congressional_current_weights.csv
  - vtd10_to_congressional_proposed_weights.csv
  - vtd10_to_state_house_weights.csv
  - vtd10_to_state_senate_weights.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

from build_fl_vtd10_crosswalks import (
    build_weight_table,
    map_blocks_to_target,
    read_geo,
    read_nhgis_crosswalk,
    write_csv,
)


TIGER_2020PL_FL_ROOT = "https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/12_FLORIDA/12"
TIGER_2020_TABBLOCK10_URL = f"{TIGER_2020PL_FL_ROOT}/tl_2020_12_tabblock10.zip"
TIGER_2020_TABBLOCK20_URL = f"{TIGER_2020PL_FL_ROOT}/tl_2020_12_tabblock20.zip"
TIGER_2012_VTD10_URL = "https://www2.census.gov/geo/tiger/TIGER2012/VTD/tl_2012_12_vtd10.zip"
DRA_FL_2020_GEOJSON_URL = "https://raw.githubusercontent.com/dra2020/vtd_data/master/2020_VTD/FL/Geojson_FL.v07.zip"
DRA_FL_2020_GEOJSON_MEMBER = "FL_2020_VD_tabblock.vtd.datasets.geojson"


def resolve_under(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    return dest


def extract_zip_member(zip_path: Path, member_name: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member_name) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    return dest


def pick_dra_vtd20_id_column(columns: list[str]) -> str:
    for candidate in ("id", "GEOID20", "geoid20", "GEOID"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a DRA 2020 VTD identifier column. "
        f"Available columns: {', '.join(columns)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FL VTD2010 crosswalk tables using DRA's Florida 2020 VTD GeoJSON."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="crosswalks", help="Output dir under data-dir unless absolute.")
    parser.add_argument("--sources-dir", default="sources/dra_vtd_data", help="Download/extract cache under data-dir unless absolute.")
    parser.add_argument("--nhgis", default="nhgis_blk2010_blk2020_12.zip")
    parser.add_argument("--tabblock10-url", default=TIGER_2020_TABBLOCK10_URL)
    parser.add_argument("--tabblock20-url", default=TIGER_2020_TABBLOCK20_URL)
    parser.add_argument("--vtd10-url", default=TIGER_2012_VTD10_URL)
    parser.add_argument("--dra-geojson-url", default=DRA_FL_2020_GEOJSON_URL)
    parser.add_argument("--dra-geojson-member", default=DRA_FL_2020_GEOJSON_MEMBER)
    parser.add_argument("--congressional-current", default="fl_congressional_districts.geojson")
    parser.add_argument("--congressional-proposed", default="fl_proposed_congressional_districts.geojson")
    parser.add_argument("--state-house", default="fl_state_house_districts.geojson")
    parser.add_argument("--state-senate", default="fl_state_senate_districts.geojson")
    parser.add_argument("--force-download", action="store_true", help="Re-download remote inputs and re-extract DRA GeoJSON.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = resolve_under(data_dir, args.output_dir)
    sources_dir = resolve_under(data_dir, args.sources_dir)

    nhgis_path = resolve_under(data_dir, args.nhgis)
    tabblock10_zip = sources_dir / "tl_2020_12_tabblock10.zip"
    tabblock20_zip = sources_dir / "tl_2020_12_tabblock20.zip"
    vtd10_zip = sources_dir / "tl_2012_12_vtd10.zip"
    dra_geojson_zip = sources_dir / "Geojson_FL.v07.zip"
    dra_geojson_path = sources_dir / args.dra_geojson_member

    cd_current_path = resolve_under(data_dir, args.congressional_current)
    cd_proposed_path = resolve_under(data_dir, args.congressional_proposed)
    sh_path = resolve_under(data_dir, args.state_house)
    ss_path = resolve_under(data_dir, args.state_senate)

    if not nhgis_path.exists():
        raise FileNotFoundError(
            f"Missing NHGIS 2010->2020 block crosswalk: {nhgis_path}. "
            "This repository already expects that archive under data/."
        )

    download_file(args.tabblock10_url, tabblock10_zip, force=args.force_download)
    download_file(args.tabblock20_url, tabblock20_zip, force=args.force_download)
    download_file(args.vtd10_url, vtd10_zip, force=args.force_download)
    download_file(args.dra_geojson_url, dra_geojson_zip, force=args.force_download)
    extract_zip_member(
        dra_geojson_zip,
        args.dra_geojson_member,
        dra_geojson_path,
        force=args.force_download,
    )

    nhgis_df = read_nhgis_crosswalk(nhgis_path)
    block10 = read_geo(tabblock10_zip)
    block20 = read_geo(tabblock20_zip)
    vtd10 = read_geo(vtd10_zip)
    vtd20_dra = read_geo(dra_geojson_path)
    cd_current = read_geo(cd_current_path)
    cd_proposed = read_geo(cd_proposed_path)
    state_house = read_geo(sh_path)
    state_senate = read_geo(ss_path)

    dra_vtd20_id_col = pick_dra_vtd20_id_column(list(vtd20_dra.columns))

    b10_to_vtd10 = map_blocks_to_target(block10, "GEOID10", vtd10, "GEOID10", "from_vtd10")
    b20_to_vtd20 = map_blocks_to_target(block20, "GEOID20", vtd20_dra, dra_vtd20_id_col, "to_vtd20")
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
        "sources_dir": str(sources_dir),
        "dra_geojson_url": args.dra_geojson_url,
        "dra_geojson_member": args.dra_geojson_member,
        "dra_vtd20_id_column": dra_vtd20_id_col,
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
