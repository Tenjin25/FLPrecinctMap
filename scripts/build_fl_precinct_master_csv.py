#!/usr/bin/env python3
"""
Build comprehensive CSV exports from Florida precinct text files.

Inputs:
  - Any data/precinctlevelelectionresults*gen*/ directory containing *.txt files.
    (year inferred from directory name)

Outputs (default: data/derived):
  - fl_precinct_results_<year>_long.csv (one per discovered year)
  - fl_precinct_results_all_years_long.csv
  - fl_precinct_legislative_all_years_long.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


RAW_COLUMNS = [
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

OUTPUT_COLUMNS = [
    "year",
    "county_code",
    "county_name",
    "precinct_code",
    "precinct_name",
    "office_desc",
    "district_desc",
    "district_num",
    "race_code",
    "candidate_name",
    "party_code",
    "party_bucket",
    "office_scope",
    "candidate_id",
    "candidate_code",
    "votes",
    "election_date",
    "election_name",
    "source_file",
]


def clean_text(value: object) -> str:
    s = str(value or "").strip()
    return re.sub(r"\s+", " ", s)


def normalize_party_bucket(party_code: object) -> str:
    p = str(party_code or "").strip().upper()
    if p in {"DEM", "D"}:
        return "dem"
    if p in {"REP", "R"}:
        return "rep"
    return "other"


def office_scope(office_desc: object) -> str:
    s = clean_text(office_desc).lower()
    if s == "state representative":
        return "state_house"
    if s == "state senator":
        return "state_senate"
    return ""


def parse_district_num(district_desc: object) -> str:
    s = str(district_desc or "").strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return ""
    return str(int(m.group(1)))


def parse_year_from_text(value: object) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


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


def discover_precinct_layout(data_dir: Path) -> List[Tuple[int, Path, List[Path]]]:
    found: List[Tuple[int, Path, List[Path]]] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        files = select_precinct_files(child)
        if not files:
            continue

        year = parse_year_from_text(child.name)
        if year is None:
            for f in files:
                year = parse_year_from_text(f.name)
                if year is not None:
                    break
        if year is None:
            continue

        found.append((int(year), child, files))
    return found


def read_precinct_txt(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        dtype=str,
        encoding="latin1",
        low_memory=False,
    )
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in RAW_COLUMNS:
        if col == "votes":
            continue
        df[col] = df[col].map(clean_text)

    df["year"] = int(year)
    df["source_file"] = path.name
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df["district_num"] = df["district_desc"].map(parse_district_num)
    df["party_bucket"] = df["party_code"].map(normalize_party_bucket)
    df["office_scope"] = df["office_desc"].map(office_scope)
    return df[OUTPUT_COLUMNS].copy()


def append_csv(path: Path, frame: pd.DataFrame, header_written: bool) -> bool:
    if frame.empty:
        return header_written
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a",
        index=False,
        header=not header_written,
        encoding="utf-8",
    )
    return True


def ensure_removed(paths: Iterable[Path]) -> None:
    for p in paths:
        if p.exists():
            p.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comprehensive precinct CSV exports.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="derived", help="Output dir under data-dir unless absolute.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (data_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_precinct_layout(data_dir)
    if not discovered:
        print(
            json.dumps(
                {
                    "output_dir": str(out_dir),
                    "warning": "No precinctlevelelectionresults*gen directories found.",
                },
                indent=2,
            )
        )
        return 0

    years = sorted(set(year for year, _, _ in discovered))
    year_paths: Dict[int, Path] = {
        year: out_dir / f"fl_precinct_results_{year}_long.csv" for year in years
    }
    combined_path = out_dir / "fl_precinct_results_all_years_long.csv"
    legislative_path = out_dir / "fl_precinct_legislative_all_years_long.csv"

    all_outputs = [*year_paths.values(), combined_path, legislative_path]
    ensure_removed(all_outputs)

    header_written: Dict[Path, bool] = {p: False for p in all_outputs}
    files_processed = 0
    total_rows = 0
    legislative_rows = 0
    rows_by_year: Dict[int, int] = {year: 0 for year in years}

    for year, folder, files in discovered:
        for txt in files:
            frame = read_precinct_txt(txt, year=year)
            if frame.empty:
                continue

            rows = int(len(frame))
            total_rows += rows
            rows_by_year[year] = rows_by_year.get(year, 0) + rows
            files_processed += 1

            year_path = year_paths[year]
            header_written[year_path] = append_csv(year_path, frame, header_written[year_path])
            header_written[combined_path] = append_csv(combined_path, frame, header_written[combined_path])

            legislative = frame[frame["office_scope"] != ""]
            legislative_rows += int(len(legislative))
            header_written[legislative_path] = append_csv(
                legislative_path, legislative, header_written[legislative_path]
            )

    year_outputs = {str(year): str(path) for year, path in sorted(year_paths.items())}
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "years_discovered": years,
                "files_processed": files_processed,
                "rows_total": total_rows,
                "rows_by_year": rows_by_year,
                "legislative_rows": legislative_rows,
                "outputs": {
                    "per_year": year_outputs,
                    "combined": str(combined_path),
                    "legislative": str(legislative_path),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
