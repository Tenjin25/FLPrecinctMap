#!/usr/bin/env python3
"""
Build actual Florida legislative district contest slices.

This script adds chamber-native district election slices:
  - state_house_state_house_<year>.json
  - state_senate_state_senate_<year>.json

Sources:
  1) FL precinct-level county text files (auto-discovered folders)
  2) FL DOS county-level election text files (tab-delimited with headers)

Output:
  - data/district_contests/*.json (chamber-native slices)
  - data/district_contests/manifest.json (updated/merged)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


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

OFFICE_TO_SCOPE = {
    "state representative": "state_house",
    "state senator": "state_senate",
}

RACECODE_TO_SCOPE = {
    "STR": "state_house",
    "STS": "state_senate",
}


def parse_year_from_date(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None
    return int(dt.year)


def parse_year_from_text(value: object) -> Optional[int]:
    s = str(value or "")
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def normalize_party_bucket(party_code: object) -> str:
    p = str(party_code or "").strip().upper()
    if p in {"DEM", "D"}:
        return "dem"
    if p in {"REP", "R"}:
        return "rep"
    return "other"


def clean_text(value: object) -> str:
    s = str(value or "").strip()
    return re.sub(r"\s+", " ", s)


def parse_district_num(value: object) -> str:
    s = str(value or "").strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return ""
    return str(int(m.group(1)))


def should_skip_candidate(name: object) -> bool:
    s = clean_text(name).lower()
    if not s:
        return True
    if "undervote" in s or "under vote" in s:
        return True
    if "overvote" in s or "over vote" in s:
        return True
    return False


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


def district_result_row(
    dem_votes: int,
    rep_votes: int,
    other_votes: int,
    dem_candidate: str,
    rep_candidate: str,
) -> Dict[str, object]:
    total_votes = int(dem_votes) + int(rep_votes) + int(other_votes)
    margin = int(rep_votes) - int(dem_votes)
    margin_pct = (margin / total_votes * 100.0) if total_votes > 0 else 0.0
    winner = "REP" if rep_votes > dem_votes else ("DEM" if dem_votes > rep_votes else "TIE")
    return {
        "dem_votes": int(dem_votes),
        "rep_votes": int(rep_votes),
        "other_votes": int(other_votes),
        "total_votes": int(total_votes),
        "margin": int(margin),
        "margin_pct": round(float(margin_pct), 6),
        "winner": winner,
        "dem_candidate": dem_candidate,
        "rep_candidate": rep_candidate,
        "color": margin_color(winner, abs(float(margin_pct))),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


def select_precinct_files(folder: Path) -> List[Path]:
    candidates = [p for p in sorted(folder.glob("*PctResults*.txt")) if p.is_file()]
    if not candidates:
        candidates = [p for p in sorted(folder.glob("*.txt")) if p.is_file()]
    if not candidates:
        return []

    by_county: Dict[str, List[Path]] = {}
    for path in candidates:
        county = path.name.split("_", 1)[0].strip().upper() or path.stem.upper()
        by_county.setdefault(county, []).append(path)

    selected: List[Path] = []
    for county in sorted(by_county.keys()):
        files = sorted(by_county[county])
        recount = [f for f in files if "recount" in f.name.lower()]
        selected.append(sorted(recount if recount else files)[-1])
    return selected


def discover_precinct_sources(data_dir: Path) -> List[Tuple[int, Path, List[Path]]]:
    out: List[Tuple[int, Path, List[Path]]] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        files = select_precinct_files(child)
        if not files:
            continue
        # Keep only precinct result folders.
        name_l = child.name.lower()
        if "pctresults" not in " ".join(p.name.lower() for p in files[:3]) and "precinct" not in name_l:
            continue

        year = parse_year_from_text(child.name)
        if year is None:
            for f in files:
                year = parse_year_from_text(f.name)
                if year is not None:
                    break
        if year is None:
            continue
        out.append((year, child, files))
    return out


def read_precinct_file(path: Path, year: int) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    df["office_key"] = df["office_desc"].map(lambda v: clean_text(v).lower())
    df = df[df["office_key"].isin(OFFICE_TO_SCOPE.keys())].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    df["scope"] = df["office_key"].map(OFFICE_TO_SCOPE)
    df["district"] = df["district_desc"].map(parse_district_num)
    missing_dist = df["district"] == ""
    if missing_dist.any():
        df.loc[missing_dist, "district"] = df.loc[missing_dist, "race_code"].map(parse_district_num)

    df["candidate_name"] = df["candidate_name"].map(clean_text)
    df = df[~df["candidate_name"].map(should_skip_candidate)]
    df = df[df["district"] != ""]
    if df.empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df = df[df["votes"] > 0]
    df["party_bucket"] = df["party_code"].map(normalize_party_bucket)
    df["year"] = int(year)

    return df[["year", "scope", "district", "party_bucket", "candidate_name", "votes"]]


def load_precinct_year_files(files: Iterable[Path], year: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for txt in files:
        try:
            part = read_precinct_file(txt, year)
        except Exception as exc:
            print(f"[WARN] Failed reading {txt.name}: {exc}")
            continue
        if not part.empty:
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])
    return pd.concat(rows, ignore_index=True)


def build_candidate_name_from_dos(df: pd.DataFrame) -> pd.Series:
    first = df.get("CanNameFirst", "").map(clean_text)
    last = df.get("CanNameLast", "").map(clean_text)
    both = (first + " " + last).str.strip()
    candidate = both.mask((first == "") & (last != ""), last)
    candidate = candidate.mask((last == "") & (first != ""), first)
    candidate = candidate.mask((first == "") & (last == ""), "")
    return candidate


def load_dos_legislative_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin1", low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    if "RaceCode" not in df.columns or "CanVotes" not in df.columns:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    df["RaceCode"] = df["RaceCode"].astype(str).str.strip().str.upper()
    df = df[df["RaceCode"].isin(RACECODE_TO_SCOPE.keys())].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    year: Optional[int] = None
    if "ElectionDate" in df.columns:
        year = parse_year_from_date(df["ElectionDate"].iloc[0])
    if year is None:
        year = parse_year_from_text(path.name)
    if year is None:
        raise ValueError(f"Could not infer year for DOS file: {path}")

    df["scope"] = df["RaceCode"].map(RACECODE_TO_SCOPE)
    df["district"] = df.get("Juris1num", "").map(parse_district_num)
    df["candidate_name"] = build_candidate_name_from_dos(df)
    df = df[~df["candidate_name"].map(should_skip_candidate)]
    df = df[df["district"] != ""]
    if df.empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])

    df["votes"] = pd.to_numeric(df["CanVotes"], errors="coerce").fillna(0).astype(int)
    df = df[df["votes"] > 0]
    df["party_bucket"] = df.get("PartyCode", "").map(normalize_party_bucket)
    df["year"] = int(year)
    return df[["year", "scope", "district", "party_bucket", "candidate_name", "votes"]]


def load_all_legislative_records(data_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    precinct_years_seen: set[int] = set()

    for year, folder, files in discover_precinct_sources(data_dir):
        part = load_precinct_year_files(files, year)
        if not part.empty:
            frames.append(part)
            precinct_years_seen.add(int(year))

    dos_dir = data_dir / "DOS files"
    if dos_dir.exists():
        for txt in sorted(dos_dir.glob("*.txt")):
            try:
                part = load_dos_legislative_file(txt)
            except Exception as exc:
                print(f"[WARN] Skipping DOS file {txt.name}: {exc}")
                continue
            if part.empty:
                continue
            year = int(part["year"].iloc[0])
            # Prefer precinct-level files when available for a given year.
            if year in precinct_years_seen:
                continue
            frames.append(part)

    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=["year", "scope", "district", "party_bucket", "candidate_name", "votes"])
    out = pd.concat(non_empty, ignore_index=True)
    out["district"] = out["district"].map(parse_district_num)
    out = out[out["district"] != ""]
    return out


def top_party_candidate(
    candidate_totals: pd.DataFrame,
    district: str,
    party_bucket: str,
    fallback: str,
) -> str:
    subset = candidate_totals[
        (candidate_totals["district"] == district) & (candidate_totals["party_bucket"] == party_bucket)
    ].copy()
    if subset.empty:
        return fallback
    subset = subset.sort_values(["votes", "candidate_name"], ascending=[False, True])
    name = clean_text(subset.iloc[0]["candidate_name"])
    return name or fallback


def aggregate_scope_year(records: pd.DataFrame, scope: str, year: int) -> Optional[Tuple[dict, dict]]:
    data = records[(records["scope"] == scope) & (records["year"] == int(year))].copy()
    if data.empty:
        return None

    candidate_totals = (
        data.groupby(["district", "party_bucket", "candidate_name"], as_index=False)["votes"]
        .sum()
    )
    district_party = (
        data.groupby(["district", "party_bucket"], as_index=False)["votes"]
        .sum()
    )

    pivot = district_party.pivot(index="district", columns="party_bucket", values="votes").fillna(0)
    for col in ("dem", "rep", "other"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.reset_index()

    def district_sort_key(d: str):
        try:
            return (0, int(d))
        except Exception:
            return (1, str(d))

    pivot["district"] = pivot["district"].map(lambda v: str(v).strip())
    pivot = pivot[pivot["district"] != ""].copy()
    if pivot.empty:
        return None
    pivot = pivot.sort_values("district", key=lambda s: s.map(district_sort_key))

    result_map: Dict[str, dict] = {}
    for _, row in pivot.iterrows():
        district = str(row["district"]).strip()
        dem_votes = int(round(float(row.get("dem", 0))))
        rep_votes = int(round(float(row.get("rep", 0))))
        other_votes = int(round(float(row.get("other", 0))))
        dem_candidate = top_party_candidate(candidate_totals, district, "dem", "Democrat")
        rep_candidate = top_party_candidate(candidate_totals, district, "rep", "Republican")
        result_map[district] = district_result_row(
            dem_votes=dem_votes,
            rep_votes=rep_votes,
            other_votes=other_votes,
            dem_candidate=dem_candidate,
            rep_candidate=rep_candidate,
        )

    contest_type = scope
    payload = {
        "meta": {
            "scope": scope,
            "contest_type": contest_type,
            "year": int(year),
            "source": "Florida legislative election files (precinct text + DOS)",
            "allocation_method": "actual_legislative",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "general": {
            "results": result_map,
        },
    }

    dem_total = int(sum(v["dem_votes"] for v in result_map.values()))
    rep_total = int(sum(v["rep_votes"] for v in result_map.values()))
    other_total = int(sum(v["other_votes"] for v in result_map.values()))

    manifest_entry = {
        "scope": scope,
        "contest_type": contest_type,
        "year": int(year),
        "file": f"{scope}_{contest_type}_{year}.json",
        "rows": len(result_map),
        "dem_total": dem_total,
        "rep_total": rep_total,
        "other_total": other_total,
        "major_party_contested": bool(dem_total > 0 and rep_total > 0),
        "match_coverage_pct": 100.0,
    }
    return payload, manifest_entry


def merge_manifest(manifest_path: Path, new_entries: Iterable[dict]) -> dict:
    current_files: List[dict] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
        current_files = list(existing.get("files") or [])

    filtered = []
    for entry in current_files:
        scope = str(entry.get("scope") or "").strip()
        contest_type = str(entry.get("contest_type") or "").strip()
        if (scope, contest_type) in {("state_house", "state_house"), ("state_senate", "state_senate")}:
            continue
        filtered.append(entry)

    merged = filtered + list(new_entries)
    merged.sort(key=lambda e: (str(e.get("scope") or ""), str(e.get("contest_type") or ""), int(e.get("year") or 0)))
    return {"files": merged}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build actual FL legislative district contest slices.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None, help="Default: <data-dir>/district_contests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (data_dir / "district_contests").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_legislative_records(data_dir)
    if records.empty:
        print(json.dumps({"warning": "No legislative records found.", "output_dir": str(out_dir)}, indent=2))
        return 0

    written_files = 0
    manifest_new: List[dict] = []
    summary_rows = []
    scopes = ["state_house", "state_senate"]
    years = sorted(set(records["year"].astype(int).tolist()))

    for scope in scopes:
        for year in years:
            built = aggregate_scope_year(records, scope=scope, year=year)
            if built is None:
                continue
            payload, manifest_entry = built
            out_file = out_dir / manifest_entry["file"]
            write_json(out_file, payload)
            written_files += 1
            manifest_new.append(manifest_entry)
            summary_rows.append(
                {
                    "scope": scope,
                    "year": int(year),
                    "rows": int(manifest_entry["rows"]),
                    "dem_total": int(manifest_entry["dem_total"]),
                    "rep_total": int(manifest_entry["rep_total"]),
                    "other_total": int(manifest_entry["other_total"]),
                }
            )

    manifest_path = out_dir / "manifest.json"
    merged_manifest = merge_manifest(manifest_path, manifest_new)
    write_json(manifest_path, merged_manifest)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "records_loaded": int(len(records)),
                "years_loaded": years,
                "slices_written": int(written_files),
                "manifest_entries_added": int(len(manifest_new)),
                "summary": summary_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

