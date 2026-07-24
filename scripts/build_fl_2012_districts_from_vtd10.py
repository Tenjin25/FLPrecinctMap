#!/usr/bin/env python3
"""
Allocate 2012 FL DOS precinct results onto district layers via Census VTD2010 geometry.

This is needed because many 2012 precinct IDs do not match simple numeric VTD GEOID
suffixes (Broward letter codes, Miami-Dade FIPS 086, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from build_fl_district_contests import (
    VEST_COUNTY_ABBREV_TO_FIPS,
    allocate_integer_votes,
    district_result_row,
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

# Census uses 086 for Miami-Dade; many election files still use DAD/025.
COUNTY_ABBREV_TO_CENSUS_FIPS = dict(VEST_COUNTY_ABBREV_TO_FIPS)
COUNTY_ABBREV_TO_CENSUS_FIPS["DAD"] = "086"

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


def precinct_code_variants(code: object) -> List[str]:
    raw = clean_text(code).upper()
    if not raw:
        return []
    out = {raw, raw.replace(" ", ""), raw.replace("-", ""), raw.replace(".", "")}
    if raw.isdigit():
        n = str(int(raw))
        out.update({n, n.zfill(3), n.zfill(4), n.zfill(6)})
    # Strip trailing letter variants for parent matching later.
    m = re.fullmatch(r"([A-Z]*\d+)([A-Z])", raw.replace(" ", ""))
    if m:
        out.add(m.group(1))
    return sorted(out)


def vtd_name_token(name: object) -> str:
    s = clean_text(name).upper()
    if not s:
        return ""
    s = s.replace("-VOTING DISTRICT", "").replace("VOTING DISTRICT", "").strip()
    # Keep leading precinct-like token: A001, L013, 356.1, 1016, etc.
    m = re.match(r"^([A-Z]*\d+(?:\.\d+)?[A-Z]?)", s.replace(" ", ""))
    if m:
        return m.group(1).replace(".", "")
    return re.sub(r"[^A-Z0-9]", "", s.split()[0]) if s else ""


def numbers_in_text(value: object) -> List[str]:
    out: List[str] = []
    for n in re.findall(r"\d+", str(value or "")):
        if not n:
            continue
        out.append(n)
        out.append(str(int(n)))
    return out


def dad_canon_keys_from_precinct_name(name: object) -> List[str]:
    """Miami-Dade crossref names look like 'PRECINCT 001.0' / 'PRECINCT 021.1'."""
    m = re.search(r"(?i)PRECINCT\s*(\d+)(?:\.(\d+))?", str(name or ""))
    if not m:
        return []
    whole = str(int(m.group(1)))
    keys = [whole, m.group(1), m.group(1).zfill(3), m.group(1).zfill(4)]
    if m.group(2):
        keys.extend([whole + m.group(2), f"{whole}.{m.group(2)}", m.group(1) + m.group(2)])
    return keys


def leading_number_keys_from_label(label: object) -> List[str]:
    """Flagler-style labels: '01 Bunnell City Hall' -> 1 / 01."""
    m = re.match(r"^\s*(\d+)", str(label or ""))
    if not m:
        return []
    raw = m.group(1)
    return [raw, str(int(raw)), raw.zfill(2), raw.zfill(3), raw.zfill(4)]


def load_precinct_crossref(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """county_code -> precinct_code -> extra match keys."""
    if not path.exists():
        print(f"[WARN] Missing precinct crossref: {path}")
        return {}
    df = pd.read_excel(path, dtype={"CountyCode": str, "PrecinctID": str, "AltId2_PrecinctFVRS": str})
    out: Dict[str, Dict[str, List[str]]] = {}
    for _, row in df.iterrows():
        county = clean_text(row.get("CountyCode")).upper()
        pid = clean_text(row.get("PrecinctID"))
        if not county or not pid:
            continue
        keys: List[str] = []
        alt = clean_text(row.get("AltId2_PrecinctFVRS")).upper()
        if alt:
            keys.append(alt)
            if alt.startswith(county):
                keys.append(alt[len(county) :])
        pname = row.get("PrecinctName")
        keys.extend(dad_canon_keys_from_precinct_name(pname))
        keys.extend(leading_number_keys_from_label(pname))
        keys.extend(leading_number_keys_from_label(row.get("PollingPlaceName")))
        # Keep unique while preserving order.
        seen = set()
        uniq = []
        for key in keys:
            k = clean_text(key).upper()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        out.setdefault(county, {})[pid] = uniq
        # Also index zero-stripped / int forms of PrecinctID.
        if pid.isdigit():
            out[county].setdefault(str(int(pid)), uniq)
    print(f"[OK] Loaded precinct crossref for {len(out)} counties from {path.name}")
    return out


def build_vtd_index(vtd: gpd.GeoDataFrame) -> Dict[str, Dict[str, List[str]]]:
    """countyfp -> {match_key -> [geoid10, ...]}"""
    index: Dict[str, Dict[str, List[str]]] = {}
    for _, row in vtd.iterrows():
        county = str(row.get("COUNTYFP10") or "").zfill(3)
        geoid = str(row.get("GEOID10") or "").strip()
        if not county or not geoid:
            continue
        keys = set()
        vtdst = str(row.get("VTDST10") or "").strip().upper()
        name = str(row.get("NAME10") or "")
        token = vtd_name_token(name)
        if vtdst:
            keys.add(vtdst)
            keys.add(vtdst.lstrip("0") or "0")
            if vtdst.isdigit():
                keys.add(str(int(vtdst)))
                keys.add(str(int(vtdst)).zfill(4))
        if token:
            keys.add(token)
            keys.add(token.lstrip("0") or token)
            # Parent key without trailing letter: A039A -> A039
            m = re.fullmatch(r"([A-Z]*\d+)([A-Z])", token)
            if m:
                keys.add(m.group(1))
        if geoid.isdigit() and len(geoid) >= 4:
            suffix = geoid[-4:]
            keys.add(suffix)
            keys.add(suffix.lstrip("0") or "0")
            keys.add(str(int(suffix)))
        # Number mentions inside VTD names (helps Citrus / some split labels).
        for n in numbers_in_text(name):
            keys.add(f"NUM:{n}")
        bucket = index.setdefault(county, {})
        for key in keys:
            if not key:
                continue
            bucket.setdefault(key, []).append(geoid)
    # de-dupe geoid lists
    for county, mapping in index.items():
        for key, geoids in list(mapping.items()):
            mapping[key] = sorted(set(geoids))
    return index


def resolve_precinct_to_vtds(
    county_code: str,
    county_fips: str,
    precinct_code: str,
    vtd_index: Dict[str, Dict[str, List[str]]],
    crossref: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    mapping = vtd_index.get(county_fips) or {}
    keys = list(precinct_code_variants(precinct_code))
    for extra in crossref.get(county_code, {}).get(precinct_code, []):
        keys.extend(precinct_code_variants(extra))
        keys.append(extra)
    raw = clean_text(precinct_code).upper().replace(" ", "")
    if raw.isdigit():
        keys.append(f"NUM:{str(int(raw))}")
    elif raw:
        keys.append(f"NUM:{raw}")

    for key in keys:
        hit = mapping.get(key)
        if hit:
            return hit
    # Parent split fallback: precinct A039 -> VTDs A039A/A039B keyed under parent.
    if raw:
        kids = []
        for key, geoids in mapping.items():
            if re.fullmatch(re.escape(raw) + r"[A-Z]", key):
                kids.extend(geoids)
        if kids:
            return sorted(set(kids))
    return []


def load_2012_precinct_votes(folder: Path) -> pd.DataFrame:
    files = select_precinct_files(folder)
    frames: List[pd.DataFrame] = []
    for path in files:
        county = path.name.split("_", 1)[0].strip().upper()
        fips = COUNTY_ABBREV_TO_CENSUS_FIPS.get(county, "")
        if not fips:
            print(f"[WARN] Unknown county abbrev {county} in {path.name}")
            continue
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
        df["party_bucket"] = df["party_code"].map(normalize_party_bucket)
        df["county_code"] = county
        df["county_fips"] = fips
        df["precinct_code"] = df["precinct_code"].map(clean_text)
        frames.append(
            df[
                [
                    "contest_type",
                    "county_code",
                    "county_fips",
                    "precinct_code",
                    "party_bucket",
                    "candidate_name",
                    "votes",
                ]
            ]
        )
    if not frames:
        raise RuntimeError(f"No 2012 statewide contest rows found in {folder}")
    return pd.concat(frames, ignore_index=True)


def map_votes_to_vtd(
    votes: pd.DataFrame,
    vtd_index: Dict[str, Dict[str, List[str]]],
    crossref: Dict[str, Dict[str, List[str]]],
) -> Tuple[pd.DataFrame, float]:
    rows = []
    total_votes = float(votes["votes"].sum())
    matched_votes = 0.0
    unmatched_by_county: Dict[str, float] = {}

    grouped = votes.groupby(
        ["contest_type", "county_code", "county_fips", "precinct_code", "party_bucket", "candidate_name"],
        as_index=False,
    )["votes"].sum()

    for _, row in grouped.iterrows():
        geoids = resolve_precinct_to_vtds(
            county_code=row["county_code"],
            county_fips=row["county_fips"],
            precinct_code=row["precinct_code"],
            vtd_index=vtd_index,
            crossref=crossref,
        )
        votes_i = float(row["votes"])
        if not geoids:
            unmatched_by_county[row["county_code"]] = unmatched_by_county.get(row["county_code"], 0.0) + votes_i
            continue
        matched_votes += votes_i
        share = votes_i / len(geoids)
        for geoid in geoids:
            rows.append(
                {
                    "contest_type": row["contest_type"],
                    "from_vtd10": geoid,
                    "party_bucket": row["party_bucket"],
                    "candidate_name": row["candidate_name"],
                    "votes": share,
                }
            )

    coverage = (matched_votes / total_votes * 100.0) if total_votes else 0.0
    if unmatched_by_county:
        worst = sorted(unmatched_by_county.items(), key=lambda kv: kv[1], reverse=True)[:12]
        print("[WARN] Unmatched 2012 precinct vote share by county:")
        for county, v in worst:
            print(f"  {county}: {v:,.0f} votes")
    print(f"[OK] Precinct->VTD coverage: {coverage:.2f}% ({matched_votes:,.0f}/{total_votes:,.0f})")
    return pd.DataFrame(rows), coverage


def assign_vtds_to_districts(
    vtd_gdf: gpd.GeoDataFrame,
    district_geojson: Path,
    district_col_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    districts = gpd.read_file(district_geojson)
    if districts.crs is None:
        districts = districts.set_crs(4326)
    districts = districts.to_crs(4326)

    dcol = None
    for cand in district_col_candidates or ["DISTRICT", "district", "CD", "SLDLST", "SLDUST"]:
        if cand in districts.columns:
            dcol = cand
            break
    if not dcol:
        raise ValueError(f"No district id column in {district_geojson}")

    v = vtd_gdf[["GEOID10", "geometry"]].copy()
    if v.crs is None:
        v = v.set_crs(4326)
    v = v.to_crs(4326)
    v["geometry"] = v.geometry.representative_point()

    d = districts[[dcol, "geometry"]].rename(columns={dcol: "district"}).copy()
    joined = gpd.sjoin(v, d, how="left", predicate="within")
    if joined["district"].isna().any():
        missing = joined[joined["district"].isna()][["GEOID10", "geometry"]]
        fallback = gpd.sjoin(missing, d, how="left", predicate="intersects")
        fallback = fallback[["GEOID10", "district"]].dropna().drop_duplicates("GEOID10")
        joined = joined.merge(fallback, on="GEOID10", how="left", suffixes=("", "_fb"))
        joined["district"] = joined["district"].fillna(joined["district_fb"])
        joined = joined.drop(columns=[c for c in joined.columns if c.endswith("_fb")])

    out = joined[["GEOID10", "district"]].dropna().drop_duplicates("GEOID10")
    out["district"] = out["district"].map(lambda x: str(int(x)) if str(x).strip().isdigit() else str(x).strip())
    out = out.rename(columns={"GEOID10": "from_vtd10"})
    return out


def contest_candidates(vtd_votes: pd.DataFrame, contest_type: str) -> Tuple[str, str]:
    part = vtd_votes[vtd_votes["contest_type"] == contest_type]
    dem = (
        part[part["party_bucket"] == "dem"]
        .groupby("candidate_name", as_index=False)["votes"]
        .sum()
        .sort_values(["votes", "candidate_name"], ascending=[False, True])
    )
    rep = (
        part[part["party_bucket"] == "rep"]
        .groupby("candidate_name", as_index=False)["votes"]
        .sum()
        .sort_values(["votes", "candidate_name"], ascending=[False, True])
    )
    dem_raw = clean_text(dem.iloc[0]["candidate_name"]) if not dem.empty else "Democrat"
    rep_raw = clean_text(rep.iloc[0]["candidate_name"]) if not rep.empty else "Republican"
    return (
        CANDIDATE_NAME_CLEANUPS.get(dem_raw.lower(), dem_raw),
        CANDIDATE_NAME_CLEANUPS.get(rep_raw.lower(), rep_raw),
    )


def allocate_contest_to_districts(
    vtd_votes: pd.DataFrame,
    vtd_to_district: pd.DataFrame,
    contest_type: str,
    coverage_pct: float,
    source: str,
    allocation_source: str,
) -> Tuple[dict, dict]:
    part = vtd_votes[vtd_votes["contest_type"] == contest_type].copy()
    dem_candidate, rep_candidate = contest_candidates(vtd_votes, contest_type)

    pivot = (
        part.groupby(["from_vtd10", "party_bucket"], as_index=False)["votes"]
        .sum()
        .pivot(index="from_vtd10", columns="party_bucket", values="votes")
        .fillna(0.0)
    )
    for col in ("dem", "rep", "other"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    rows = pivot.reset_index().rename(columns={"dem": "dem_votes", "rep": "rep_votes", "other": "other_votes"})
    merged = rows.merge(vtd_to_district, on="from_vtd10", how="inner")
    if merged.empty:
        raise RuntimeError(f"No VTD->district matches for {contest_type}")

    agg = (
        merged.groupby("district", as_index=False)[["dem_votes", "rep_votes", "other_votes"]]
        .sum()
        .sort_values("district")
        .reset_index(drop=True)
    )
    dem_alloc = allocate_integer_votes(agg["dem_votes"])
    rep_alloc = allocate_integer_votes(agg["rep_votes"])
    other_alloc = allocate_integer_votes(agg["other_votes"])

    result_map = {}
    for i, row in agg.iterrows():
        dist = str(row["district"]).strip()
        if dist.isdigit():
            dist = str(int(dist))
        result_map[dist] = district_result_row(
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
            "scope": None,  # filled by caller
            "contest_type": contest_type,
            "year": 2012,
            "source": source,
            "allocation_method": "vtd10_spatial",
            "allocation_source": allocation_source,
            "match_coverage_pct": round(coverage_pct, 6),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "general": {"results": result_map},
    }
    manifest = {
        "contest_type": contest_type,
        "year": 2012,
        "rows": len(result_map),
        "dem_total": dem_total,
        "rep_total": rep_total,
        "other_total": other_total,
        "major_party_contested": bool(dem_total > 0 and rep_total > 0),
        "match_coverage_pct": round(coverage_pct, 6),
    }
    return payload, manifest


def merge_manifest(out_dir: Path, new_entries: List[dict]) -> None:
    manifest_path = out_dir / "manifest.json"
    existing: List[dict] = []
    if manifest_path.exists():
        existing = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("files") or [])
    keyed = {
        (str(e.get("scope")), str(e.get("contest_type")), int(e.get("year"))): e
        for e in existing
        if e.get("scope") and e.get("contest_type") and e.get("year") is not None
    }
    for entry in new_entries:
        keyed[(str(entry["scope"]), str(entry["contest_type"]), int(entry["year"]))] = entry
    merged = sorted(keyed.values(), key=lambda r: (r["scope"], r["contest_type"], int(r["year"])))
    write_json(manifest_path, {"files": merged})


def main() -> int:
    parser = argparse.ArgumentParser(description="Allocate 2012 precinct results via VTD2010 geometry.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--vtd10", default="tl_2012_12_vtd10.zip")
    parser.add_argument("--precinct-dir", default="precinctlevelelectionresults2012gen")
    parser.add_argument(
        "--crossref",
        default="precinctlevelelectionresults2012gen/2012GenPrecinctCrossReference.xlsx",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    vtd_path = data_dir / args.vtd10
    precinct_dir = data_dir / args.precinct_dir
    crossref_path = data_dir / args.crossref
    if not vtd_path.exists():
        raise FileNotFoundError(vtd_path)
    if not precinct_dir.exists():
        raise FileNotFoundError(precinct_dir)

    print(f"Loading VTD10 from {vtd_path.name}...")
    vtd = gpd.read_file(vtd_path)
    if vtd.crs is None:
        vtd = vtd.set_crs(4269)
    vtd = vtd.to_crs(4326)
    vtd_index = build_vtd_index(vtd)
    print(f"[OK] Indexed {len(vtd):,} VTDs across {len(vtd_index)} counties")

    crossref = load_precinct_crossref(crossref_path)

    print("Loading 2012 precinct statewide contests...")
    votes = load_2012_precinct_votes(precinct_dir)
    vtd_votes, coverage = map_votes_to_vtd(votes, vtd_index, crossref)
    if vtd_votes.empty:
        raise RuntimeError("No precinct votes could be matched to VTDs")

    targets = [
        ("congressional", data_dir / "fl_congressional_districts.geojson", data_dir / "district_contests"),
        (
            "congressional",
            data_dir / "fl_proposed_congressional_districts.geojson",
            data_dir / "district_contests_proposed_congressional",
        ),
        (
            "congressional",
            data_dir / "fl_congressional_districts_2026.geojson",
            data_dir / "district_contests_2026_congressional",
        ),
        ("state_house", data_dir / "fl_state_house_districts.geojson", data_dir / "district_contests"),
        ("state_senate", data_dir / "fl_state_senate_districts.geojson", data_dir / "district_contests"),
    ]

    for scope, geojson, out_dir in targets:
        if not geojson.exists():
            print(f"[WARN] Missing {geojson.name}; skipping {scope} -> {out_dir.name}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Assigning VTDs to {geojson.name} for {out_dir.name}...")
        vtd_to_district = assign_vtds_to_districts(vtd, geojson)
        print(f"[OK] {len(vtd_to_district):,} VTDs mapped to {vtd_to_district['district'].nunique()} districts")

        entries = []
        for contest_type in sorted(vtd_votes["contest_type"].unique()):
            payload, manifest = allocate_contest_to_districts(
                vtd_votes=vtd_votes,
                vtd_to_district=vtd_to_district,
                contest_type=contest_type,
                coverage_pct=coverage,
                source="FL DOS 2012 precinct text via Census VTD2010 geometry",
                allocation_source=f"vtd10_spatial:{geojson.name}",
            )
            payload["meta"]["scope"] = scope
            filename = f"{scope}_{contest_type}_2012.json"
            write_json(out_dir / filename, payload)
            entry = {
                "scope": scope,
                "file": filename,
                **manifest,
            }
            entries.append(entry)
            print(
                f"[OK] {out_dir.name}/{filename}: "
                f"D={entry['dem_total']:,} R={entry['rep_total']:,} "
                f"coverage={entry['match_coverage_pct']:.2f}%"
            )
        merge_manifest(out_dir, entries)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
