#!/usr/bin/env python3
"""
Build Florida 2026 congressional-line assets and backfill 2012 district contests.

Creates / updates:
  - data/crosswalks/vtd10_to_congressional_2026_weights.csv
  - data/district_contests_2026_congressional/*  (VEST years + 2012)
  - 2012 president/us_senate slices for current CD/HD/SD and proposed CD folders
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Reuse allocation helpers from the existing district builder.
from build_fl_district_contests import (
    VEST_COUNTY_ABBREV_TO_FIPS,
    aggregate_contest_to_district,
    allocate_integer_votes,
    district_result_row,
    load_precinct_contest_rows,
    load_precinct_weight_csv,
    normalize_text,
    write_json,
)


PRECINCT_TXT_COLUMNS = [
    "county_code",
    "county_name",
    "election_id",
    "election_date",
    "election_name",
    "precinct_code",
    "precinct_name",
    "ballots_cast",
    "aux_1",
    "aux_2",
    "aux_3",
    "office_desc",
    "district_desc",
    "race_code",
    "candidate_name",
    "party_code",
    "candidate_id",
    "candidate_code",
    "votes",
]

OFFICE_TO_CONTEST = {
    "president of the united states": "president",
    "united states senator": "us_senate",
}

CANDIDATE_NAME_CLEANUPS = {
    "obama / biden": "Barack Obama",
    "romney / ryan": "Mitt Romney",
    "bill nelson": "Bill Nelson",
    "connie mack": "Connie Mack",
}


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_party_bucket(party_code: object) -> str:
    p = str(party_code or "").strip().upper()
    if p in {"DEM", "D"}:
        return "dem"
    if p in {"REP", "R"}:
        return "rep"
    return "other"


def should_skip_candidate(name: object) -> bool:
    s = clean_text(name).lower()
    if not s:
        return True
    if "undervote" in s or "under vote" in s:
        return True
    if "overvote" in s or "over vote" in s:
        return True
    if "writein" in s.replace(" ", "") or "write-in" in s:
        return True
    return False


def normalize_precinct_to_vtd_suffix(value: object) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw.split(".", 1)[0]
    raw = raw.replace(".", "")
    if not re.fullmatch(r"\d+", raw):
        return ""
    return str(int(raw)).zfill(4)


def select_precinct_files(folder: Path) -> List[Path]:
    candidates = [p for p in sorted(folder.glob("*PctResults*.txt")) if p.is_file()]
    if not candidates:
        candidates = [p for p in sorted(folder.glob("*.txt")) if p.is_file()]
    by_county: Dict[str, List[Path]] = {}
    for path in candidates:
        county = path.name.split("_", 1)[0].strip().upper() or path.stem.upper()
        by_county.setdefault(county, []).append(path)
    selected: List[Path] = []
    for county in sorted(by_county):
        files = sorted(by_county[county])
        recount = [f for f in files if "recount" in f.name.lower()]
        selected.append(sorted(recount if recount else files)[-1])
    return selected


def build_vtd10_to_congressional_2026_weights(
    data_dir: Path,
    block_assign_zip: Path,
    eog_blockfile: Path,
    vtd10_to_vtd20_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    with zipfile.ZipFile(block_assign_zip) as zf:
        vtd_blocks = pd.read_csv(zf.open("BlockAssign_ST12_FL_VTD.txt"), sep="|", dtype=str)

    vtd_blocks["BLOCKID"] = vtd_blocks["BLOCKID"].astype(str).str.strip()
    vtd_blocks["COUNTYFP"] = vtd_blocks["COUNTYFP"].astype(str).str.strip().str.zfill(3)
    vtd_blocks["DISTRICT"] = vtd_blocks["DISTRICT"].astype(str).str.strip()
    vtd_blocks["to_vtd20"] = "12" + vtd_blocks["COUNTYFP"] + vtd_blocks["DISTRICT"]
    vtd_blocks = vtd_blocks[(vtd_blocks["BLOCKID"] != "") & (vtd_blocks["to_vtd20"].str.len() >= 11)]

    cd26 = pd.read_csv(eog_blockfile, header=None, names=["BLOCKID", "district"], dtype=str)
    cd26["BLOCKID"] = cd26["BLOCKID"].astype(str).str.strip()
    cd26["district"] = cd26["district"].map(lambda v: str(int(str(v).strip())) if str(v).strip().isdigit() else "")
    cd26 = cd26[(cd26["BLOCKID"] != "") & (cd26["district"] != "")]

    joined = vtd_blocks[["BLOCKID", "to_vtd20"]].merge(cd26, on="BLOCKID", how="inner")
    if joined.empty:
        raise RuntimeError("No overlap between BlockAssign VTD blocks and EOGPCRP2026.txt")

    vtd20_to_cd = (
        joined.groupby(["to_vtd20", "district"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )
    totals = vtd20_to_cd.groupby("to_vtd20")["weight"].transform("sum")
    vtd20_to_cd["weight"] = vtd20_to_cd["weight"] / totals

    vtd10_to_vtd20 = pd.read_csv(vtd10_to_vtd20_path, dtype={"from_vtd10": str, "to_vtd20": str})
    vtd10_to_vtd20["weight"] = pd.to_numeric(vtd10_to_vtd20["weight"], errors="coerce").fillna(0.0)
    vtd10_to_vtd20 = vtd10_to_vtd20[vtd10_to_vtd20["weight"] > 0]

    merged = vtd10_to_vtd20.merge(vtd20_to_cd, on="to_vtd20", how="inner", suffixes=("_vtd", "_cd"))
    if merged.empty:
        raise RuntimeError("Failed to bridge VTD2010 -> VTD2020 -> 2026 CD")

    merged["weight"] = merged["weight_vtd"] * merged["weight_cd"]
    out = (
        merged.groupby(["from_vtd10", "district"], as_index=False)["weight"]
        .sum()
        .sort_values(["from_vtd10", "district"])
        .reset_index(drop=True)
    )
    totals = out.groupby("from_vtd10")["weight"].transform("sum")
    out = out[totals > 0].copy()
    out["weight"] = out["weight"] / totals[totals > 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.10f")
    return out


def load_2012_precinct_statewide_contests(folder: Path) -> Dict[str, Dict[str, object]]:
    files = select_precinct_files(folder)
    if not files:
        raise FileNotFoundError(f"No precinct result files found in {folder}")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=PRECINCT_TXT_COLUMNS,
            dtype=str,
            encoding="latin1",
            low_memory=False,
        )
        if df.empty:
            continue
        df["office_key"] = df["office_desc"].map(lambda v: clean_text(v).lower())
        df = df[df["office_key"].isin(OFFICE_TO_CONTEST)].copy()
        if df.empty:
            continue
        df["contest_type"] = df["office_key"].map(OFFICE_TO_CONTEST)
        df["candidate_name"] = df["candidate_name"].map(clean_text)
        df = df[~df["candidate_name"].map(should_skip_candidate)]
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
        df = df[df["votes"] > 0]
        if df.empty:
            continue

        county = path.name.split("_", 1)[0].strip().upper()
        fips = VEST_COUNTY_ABBREV_TO_FIPS.get(county, "")
        suffix = df["precinct_code"].map(normalize_precinct_to_vtd_suffix)
        df["pct_key"] = ("12" + fips + suffix).where((fips != "") & (suffix != ""), "")
        df = df[df["pct_key"] != ""].copy()
        if df.empty:
            continue
        df["party_bucket"] = df["party_code"].map(normalize_party_bucket)
        frames.append(df[["contest_type", "pct_key", "party_bucket", "candidate_name", "votes"]])

    if not frames:
        raise RuntimeError(f"No statewide contest rows parsed from {folder}")

    all_rows = pd.concat(frames, ignore_index=True)
    out: Dict[str, Dict[str, object]] = {}
    for contest_type, part in all_rows.groupby("contest_type"):
        pivot = (
            part.groupby(["pct_key", "party_bucket"], as_index=False)["votes"]
            .sum()
            .pivot(index="pct_key", columns="party_bucket", values="votes")
            .fillna(0)
        )
        for col in ("dem", "rep", "other"):
            if col not in pivot.columns:
                pivot[col] = 0
        rows = pivot.reset_index().rename(
            columns={"dem": "dem_votes", "rep": "rep_votes", "other": "other_votes"}
        )
        rows["total_votes"] = rows["dem_votes"] + rows["rep_votes"] + rows["other_votes"]
        rows["year"] = 2012
        rows["contest_type"] = contest_type

        dem_names = (
            part[part["party_bucket"] == "dem"]
            .groupby("candidate_name", as_index=False)["votes"]
            .sum()
            .sort_values(["votes", "candidate_name"], ascending=[False, True])
        )
        rep_names = (
            part[part["party_bucket"] == "rep"]
            .groupby("candidate_name", as_index=False)["votes"]
            .sum()
            .sort_values(["votes", "candidate_name"], ascending=[False, True])
        )
        dem_raw = clean_text(dem_names.iloc[0]["candidate_name"]) if not dem_names.empty else "Democrat"
        rep_raw = clean_text(rep_names.iloc[0]["candidate_name"]) if not rep_names.empty else "Republican"
        dem_candidate = CANDIDATE_NAME_CLEANUPS.get(dem_raw.lower(), dem_raw)
        rep_candidate = CANDIDATE_NAME_CLEANUPS.get(rep_raw.lower(), rep_raw)
        out[str(contest_type)] = {
            "rows": rows,
            "dem_candidate": dem_candidate,
            "rep_candidate": rep_candidate,
        }
    return out


def merge_manifest(out_dir: Path, new_entries: List[dict], replace_scopes: Optional[Iterable[str]] = None) -> None:
    manifest_path = out_dir / "manifest.json"
    existing: List[dict] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = list(payload.get("files") or [])

    replace = set(replace_scopes or [])
    if replace:
        existing = [e for e in existing if str(e.get("scope") or "") not in replace]

    keyed = {
        (str(e.get("scope")), str(e.get("contest_type")), int(e.get("year"))): e
        for e in existing
        if e.get("scope") and e.get("contest_type") and e.get("year") is not None
    }
    for entry in new_entries:
        key = (str(entry["scope"]), str(entry["contest_type"]), int(entry["year"]))
        keyed[key] = entry

    merged = sorted(keyed.values(), key=lambda r: (r["scope"], r["contest_type"], int(r["year"])))
    write_json(manifest_path, {"files": merged})


def write_allocated_contest(
    out_dir: Path,
    scope: str,
    contest_type: str,
    year: int,
    contest_payload: Dict[str, object],
    weights_obj,
    source: str,
    allocation_method: str,
) -> dict:
    contest_rows = contest_payload["rows"].copy()
    if weights_obj.key_format == "vtd10_geoid" and "pct_key_vtd10" in contest_rows.columns:
        contest_rows["pct_key"] = contest_rows["pct_key_vtd10"]
    contest_rows["pct_key"] = contest_rows["pct_key"].map(normalize_text)

    dem_candidate = str(contest_payload.get("dem_candidate") or "Democrat")
    rep_candidate = str(contest_payload.get("rep_candidate") or "Republican")
    agg, coverage_pct = aggregate_contest_to_district(contest_rows, weights_obj.weights)
    if agg.empty:
        raise RuntimeError(f"No allocated rows for {scope} {contest_type} {year}")

    agg = agg.sort_values("district").reset_index(drop=True)
    dem_alloc = allocate_integer_votes(agg["dem_votes"])
    rep_alloc = allocate_integer_votes(agg["rep_votes"])
    other_alloc = allocate_integer_votes(agg["other_votes"])

    result_map: Dict[str, dict] = {}
    for i, row in agg.iterrows():
        dist_key = str(row["district"]).strip()
        # Keep district ids without leading zeros to match existing slices.
        if dist_key.isdigit():
            dist_key = str(int(dist_key))
        result_map[dist_key] = district_result_row(
            dem_alloc[i],
            rep_alloc[i],
            other_alloc[i],
            dem_candidate,
            rep_candidate,
        )

    dem_total = int(sum(v["dem_votes"] for v in result_map.values()))
    rep_total = int(sum(v["rep_votes"] for v in result_map.values()))
    other_total = int(sum(v["other_votes"] for v in result_map.values()))

    payload = {
        "meta": {
            "scope": scope,
            "contest_type": contest_type,
            "year": year,
            "source": source,
            "allocation_method": allocation_method,
            "allocation_source": weights_obj.source,
            "match_coverage_pct": round(coverage_pct, 6),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "general": {"results": result_map},
    }
    filename = f"{scope}_{contest_type}_{year}.json"
    write_json(out_dir / filename, payload)
    return {
        "scope": scope,
        "contest_type": contest_type,
        "year": year,
        "file": filename,
        "rows": len(result_map),
        "dem_total": dem_total,
        "rep_total": rep_total,
        "other_total": other_total,
        "major_party_contested": bool(dem_total > 0 and rep_total > 0),
        "match_coverage_pct": round(coverage_pct, 6),
    }


def build_vest_years_for_2026(
    data_dir: Path,
    out_dir: Path,
    weights_csv: Path,
    years: List[int],
) -> List[dict]:
    weights_obj = load_precinct_weight_csv(weights_csv, year=2026, scope="congressional")
    entries: List[dict] = []
    for year in years:
        shp = data_dir / f"fl_{year}.zip"
        if not shp.exists():
            print(f"[WARN] Missing VEST shapefile for {year}: {shp.name}")
            continue
        contests = load_precinct_contest_rows(shp, year)
        if not contests:
            print(f"[WARN] No contest columns in {shp.name}")
            continue
        for contest_type, contest_payload in sorted(contests.items()):
            entry = write_allocated_contest(
                out_dir=out_dir,
                scope="congressional",
                contest_type=contest_type,
                year=year,
                contest_payload=contest_payload,
                weights_obj=weights_obj,
                source="VEST precinct shapefile allocation onto 2026 congressional lines",
                allocation_method="precinct_weights",
            )
            entries.append(entry)
            print(
                f"[OK] 2026 CD {contest_type} {year}: "
                f"D={entry['dem_total']:,} R={entry['rep_total']:,} "
                f"coverage={entry['match_coverage_pct']:.2f}%"
            )
    return entries


def build_2012_for_targets(
    contests_2012: Dict[str, Dict[str, object]],
    targets: List[Tuple[str, Path, Path]],
) -> None:
    for scope, weights_csv, out_dir in targets:
        if not weights_csv.exists():
            print(f"[WARN] Missing weights for {scope}: {weights_csv}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_obj = load_precinct_weight_csv(weights_csv, year=2012, scope=scope)
        entries: List[dict] = []
        for contest_type, contest_payload in sorted(contests_2012.items()):
            entry = write_allocated_contest(
                out_dir=out_dir,
                scope=scope,
                contest_type=contest_type,
                year=2012,
                contest_payload=contest_payload,
                weights_obj=weights_obj,
                source="FL DOS precinct text allocation via VTD2010 crosswalk",
                allocation_method="precinct_weights",
            )
            entries.append(entry)
            print(
                f"[OK] {out_dir.name}/{scope}_{contest_type}_2012: "
                f"D={entry['dem_total']:,} R={entry['rep_total']:,} "
                f"coverage={entry['match_coverage_pct']:.2f}%"
            )
        merge_manifest(out_dir, entries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FL 2026 congressional contest layer + 2012 backfill.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--vest-years", default="2014,2016,2018,2020,2022,2024")
    parser.add_argument("--skip-vest", action="store_true")
    parser.add_argument("--skip-2012", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    crosswalks = data_dir / "crosswalks"
    out_2026 = data_dir / "district_contests_2026_congressional"
    out_2026.mkdir(parents=True, exist_ok=True)

    weights_2026_path = crosswalks / "vtd10_to_congressional_2026_weights.csv"
    print("Building 2026 congressional VTD10 weights...")
    weights_2026 = build_vtd10_to_congressional_2026_weights(
        data_dir=data_dir,
        block_assign_zip=data_dir / "BlockAssign_ST12_FL.zip",
        eog_blockfile=data_dir / "EOGPCRP2026.txt",
        vtd10_to_vtd20_path=crosswalks / "vtd10_to_vtd20_weights.csv",
        out_path=weights_2026_path,
    )
    print(
        f"[OK] Wrote {weights_2026_path.name}: "
        f"{len(weights_2026):,} rows, {weights_2026['from_vtd10'].nunique():,} VTDs, "
        f"{weights_2026['district'].nunique()} districts"
    )

    vest_entries: List[dict] = []
    if not args.skip_vest:
        years = [int(tok.strip()) for tok in args.vest_years.split(",") if tok.strip()]
        print(f"Allocating VEST years onto 2026 lines: {years}")
        vest_entries = build_vest_years_for_2026(
            data_dir=data_dir,
            out_dir=out_2026,
            weights_csv=weights_2026_path,
            years=years,
        )
        merge_manifest(out_2026, vest_entries, replace_scopes=["congressional"])

    if not args.skip_2012:
        precinct_2012 = data_dir / "precinctlevelelectionresults2012gen"
        print(f"Loading 2012 precinct statewide contests from {precinct_2012.name}...")
        contests_2012 = load_2012_precinct_statewide_contests(precinct_2012)
        print(f"[OK] Parsed 2012 contests: {', '.join(sorted(contests_2012))}")

        targets = [
            ("congressional", crosswalks / "vtd10_to_congressional_current_weights.csv", data_dir / "district_contests"),
            ("state_house", crosswalks / "vtd10_to_state_house_weights.csv", data_dir / "district_contests"),
            ("state_senate", crosswalks / "vtd10_to_state_senate_weights.csv", data_dir / "district_contests"),
            (
                "congressional",
                crosswalks / "vtd10_to_congressional_proposed_weights.csv",
                data_dir / "district_contests_proposed_congressional",
            ),
            ("congressional", weights_2026_path, out_2026),
        ]
        build_2012_for_targets(contests_2012, targets)

    print(f"Done. 2026 contest folder: {out_2026}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
