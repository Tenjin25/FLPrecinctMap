#!/usr/bin/env python3
"""Publish Election Watch county totals and district races for the 2026 general.

Input is the statewide Election Watch county results text file. Its USR/STR/STS
rows already identify both county and district, so their district totals are
exact sums of the published rows, not precinct allocations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SOURCE_URL = "https://flelectionfiles.floridados.gov/enightfilespublic/20261103_ElecResultsFL.txt"
STATEWIDE = {"USS": "us_senate", "GOV": "governor", "ATG": "attorney_general",
             "CFO": "treasurer", "AGR": "agriculture_commissioner"}
DISTRICTS = {"USR": ("congressional", "us_house"),
             "STR": ("state_house", "state_house"),
             "STS": ("state_senate", "state_senate")}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def color(winner: str, pct: float) -> str:
    if winner == "TIE":
        return "#f7f7f7"
    palette = ("#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a",
               "#fcae91", "#fee8c8", "#f7f7f7") if winner == "REP" else (
               "#08306b", "#08519c", "#3182bd", "#6baed6", "#9ecae1",
               "#c6dbef", "#e1f5fe", "#f7f7f7")
    return palette[next(i for i, threshold in enumerate((40, 30, 20, 10, 5.5, 1, .5, 0)) if pct >= threshold)]


def candidate_name(row: dict[str, str]) -> str:
    first = row["CanNameFirst"].strip()
    middle = row["CanNameMiddle"].strip()
    last = row["CanNameLast"].strip()
    if row["RaceCode"] == "GOV":
        # The text export places the governor's surname in CanNameFirst and the
        # running mate's surname in CanNameLast. Use the Election Watch page's
        # full gubernatorial names for the two major-party map labels.
        if (first, last) == ("Donalds", "Avila"):
            return "Byron Donalds"
        if (first, last) == ("Jolly", "Graham"):
            return "David Jolly"
        return first + " / " + last
    return " ".join(part for part in (first, middle, last) if part and part != "/")


def result(rows: list[dict[str, str]]) -> dict[str, object]:
    totals = defaultdict(int)
    names: dict[str, dict[str, int]] = {"DEM": defaultdict(int), "REP": defaultdict(int)}
    for row in rows:
        party = row["PartyCode"].strip().upper()
        bucket = party if party in ("DEM", "REP") else "OTHER"
        votes = int(row["CanVotes"].replace(",", ""))
        totals[bucket] += votes
        if bucket in names:
            names[bucket][candidate_name(row)] += votes
    dem, rep, other = (totals[x] for x in ("DEM", "REP", "OTHER"))
    total = dem + rep + other
    margin = rep - dem
    pct = round(100 * margin / total, 6) if total else 0.0
    winner = "REP" if margin > 0 else "DEM" if margin < 0 else "TIE"
    return {"dem_votes": dem, "rep_votes": rep, "other_votes": other,
            "total_votes": total, "dem_candidate": max(names["DEM"], key=names["DEM"].get) if names["DEM"] else "",
            "rep_candidate": max(names["REP"], key=names["REP"].get) if names["REP"] else "",
            "margin": margin, "margin_pct": pct, "winner": winner,
            "color": color(winner, abs(pct))}


def manifest_entry(scope: str | None, contest: str, path: Path, results: dict) -> dict:
    entry = {"contest_type": contest, "year": 2026, "file": path.name, "rows": len(results),
             "dem_total": sum(r["dem_votes"] for r in results.values()),
             "rep_total": sum(r["rep_votes"] for r in results.values()),
             "other_total": sum(r["other_votes"] for r in results.values()),
             "major_party_contested": True, "source": SOURCE_URL,
             "allocation_method": "official_county_district_sums" if scope else "official_county_totals"}
    if scope:
        entry = {"scope": scope, **entry, "match_coverage_pct": 100.0}
    else:
        entry["granularity"] = "county"
    return entry


def merge_manifest(path: Path, additions: list[dict]) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": []}
    keys = {(x.get("scope"), x["contest_type"], x["year"]) for x in additions}
    files = [x for x in current["files"] if (x.get("scope"), x["contest_type"], x["year"]) not in keys]
    files.extend(additions)
    if any(x.get("scope") for x in files):
        files.sort(key=lambda x: (x.get("scope", ""), x["contest_type"], x["year"]))
    else:
        files.sort(key=lambda x: (x["contest_type"], x["year"]))
    current["files"] = files
    write_json(path, current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/20261103_ElecResultsFL.txt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    source, data = args.source, args.data_dir
    with source.open(encoding="latin1", newline="") as fh:
        raw = list(csv.DictReader(fh, delimiter="\t"))
    if not raw or {x["ElectionDate"] for x in raw} != {"11/03/2026"}:
        raise ValueError("Expected only November 3, 2026 general-election rows")
    counties = {x["CountyName"].strip().upper() for x in raw}
    if len(counties) != 67:
        raise ValueError(f"Expected 67 counties; found {len(counties)}")

    long_rows = []
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        code = row["RaceCode"].strip()
        if code not in STATEWIDE and code not in DISTRICTS:
            continue
        district = str(int(row["Juris1num"].strip())) if code in DISTRICTS else ""
        county = row["CountyName"].strip().upper()
        office = row["RaceName"].strip()
        groups[(code, district, county)].append(row)
        long_rows.append({"county": county, "precinct": "", "office": office,
                          "district": district, "party": row["PartyCode"].strip(),
                          "candidate": candidate_name(row), "votes": int(row["CanVotes"].replace(",", ""))})

    export_dir = data / "derived" / "fl_2026_openelections"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "20261103__fl__general__county.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["county", "precinct", "office", "district", "party", "candidate", "votes"])
        writer.writeheader()
        writer.writerows(sorted(long_rows, key=lambda x: (x["office"], x["district"], x["county"], x["party"], x["candidate"])))

    county_entries, main_district_entries, congressional_entries = [], [], []
    for code, contest in STATEWIDE.items():
        county_results = {county: result(groups[(code, "", county)]) for county in sorted(counties)}
        if any(not x["total_votes"] for x in county_results.values()):
            raise ValueError(f"Missing county result for {code}")
        path = data / "contests" / f"{contest}_2026.json"
        rows = [{"county": county, **value} for county, value in county_results.items()]
        write_json(path, {"meta": {"source": SOURCE_URL, "granularity": "county", "year": 2026},
                          "rows": rows, "county_totals": county_results})
        county_entries.append(manifest_entry(None, contest, path, county_results))

    for code, (scope, contest) in DISTRICTS.items():
        numbers = sorted({district for c, district, _ in groups if c == code}, key=int)
        district_results = {}
        for district in numbers:
            district_rows = [row for (c, d, _), rows in groups.items() if c == code and d == district for row in rows]
            district_results[district] = result(district_rows)
        folder = "district_contests_2026_congressional" if scope == "congressional" else "district_contests"
        path = data / folder / f"{scope}_{contest}_2026.json"
        write_json(path, {"meta": {"scope": scope, "contest_type": contest, "year": 2026,
                                    "source": SOURCE_URL, "allocation_method": "official_county_district_sums",
                                    "district_lines": "2026" if scope == "congressional" else "2022/2024 legislative"},
                          "general": {"results": district_results}})
        entry = manifest_entry(scope, contest, path, district_results)
        (congressional_entries if scope == "congressional" else main_district_entries).append(entry)

    merge_manifest(data / "contests" / "manifest.json", county_entries)
    merge_manifest(data / "district_contests" / "manifest.json", main_district_entries)
    merge_manifest(data / "district_contests_2026_congressional" / "manifest.json", congressional_entries)
    summary = {"source": SOURCE_URL, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_file": source.name, "county_count": len(counties), "candidate_rows": len(long_rows),
               "districts": {x["scope"]: x["rows"] for x in main_district_entries + congressional_entries},
               "note": "No precinct geometry or vote allocation used for these district race totals."}
    write_json(export_dir / "county_results_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
