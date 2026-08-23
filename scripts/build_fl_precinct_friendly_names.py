#!/usr/bin/env python3
"""Build NCPrecinctMap-style friendly precinct labels for Florida.

The Florida Division of Elections' official precinct-level results include a
county-scoped precinct identifier and a polling-location label.  This script
prunes those labels to the precinct codes present in the live geometry and
writes the same ``counties -> precinct_code -> display_name`` shape consumed
by the NCPrecinctMap front end.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Optional, Set, Tuple


OFFICIAL_SOURCE_URL = "https://dos.fl.gov/media/708761/2024-gen-outputofficial1.zip"


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip()


def normalize_code(value: object) -> str:
    return clean_text(value).upper()


def compact_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_code(value))


def integer_code_value(value: object) -> Optional[int]:
    raw = normalize_code(value)
    match = re.fullmatch(r"(\d+)(?:\.0+)?", raw)
    return int(match.group(1)) if match else None


def canonicalize_friendly_name(raw_name: object, display_code: str) -> str:
    cleaned = clean_text(raw_name)
    if re.fullmatch(r"(?:PRECINCT|PCT)\s+[A-Z0-9 .\-/]+", cleaned, flags=re.IGNORECASE):
        return f"Precinct {display_code}"
    return cleaned


def load_display_precincts(geojson_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    county_names: Dict[str, str] = {}
    codes_by_county: Dict[str, Set[str]] = defaultdict(set)
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        county_code = normalize_code(props.get("county_code"))
        county_name = normalize_code(props.get("county_nam"))
        precinct_code = normalize_code(props.get("prec_id"))
        if not county_code or not county_name or not precinct_code:
            continue
        county_names[county_code] = county_name
        codes_by_county[county_code].add(precinct_code)
    return county_names, dict(codes_by_county)


def resolve_display_code(raw_code: object, display_codes: Iterable[str]) -> Optional[str]:
    raw = normalize_code(raw_code)
    if not raw:
        return None
    codes = set(display_codes)
    if raw in codes:
        return raw

    raw_compact = compact_code(raw)
    compact_matches = [code for code in codes if compact_code(code) == raw_compact]
    if len(compact_matches) == 1:
        return compact_matches[0]

    raw_number = integer_code_value(raw)
    if raw_number is not None:
        numeric_matches = [
            code for code in codes
            if integer_code_value(code) is not None and integer_code_value(code) == raw_number
        ]
        if len(numeric_matches) == 1:
            return numeric_matches[0]
    return None


def build_friendly_names(source_zip: Path, geometry_path: Path) -> Tuple[dict, dict]:
    county_names, codes_by_county = load_display_precincts(geometry_path)
    candidates: DefaultDict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_keys: Set[Tuple[str, str]] = set()
    resolved_code_cache: Dict[Tuple[str, str], Optional[str]] = {}

    with zipfile.ZipFile(source_zip) as archive:
        entries = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".txt") and "_recount" not in name.lower()
        )
        for entry_name in entries:
            with archive.open(entry_name) as raw_file:
                text_file = io.TextIOWrapper(raw_file, encoding="latin1", newline="")
                for row in csv.reader(text_file, delimiter="\t"):
                    # Official field layout: county code/name are 1/2, precinct ID is 6,
                    # and precinct polling location is 7 (one-based positions).
                    if len(row) < 7:
                        continue
                    county_code = normalize_code(row[0])
                    raw_precinct_code = normalize_code(row[5])
                    friendly_name = clean_text(row[6])
                    if not county_code or not raw_precinct_code or not friendly_name:
                        continue
                    source_keys.add((county_code, raw_precinct_code))
                    source_key = (county_code, raw_precinct_code)
                    if source_key not in resolved_code_cache:
                        resolved_code_cache[source_key] = resolve_display_code(
                            raw_precinct_code,
                            codes_by_county.get(county_code, set()),
                        )
                    display_code = resolved_code_cache[source_key]
                    if not display_code:
                        continue
                    friendly_name = canonicalize_friendly_name(friendly_name, display_code)
                    candidates[(county_code, display_code)][friendly_name] += 1

    counties: Dict[str, Dict[str, str]] = {}
    matched_geometry_keys: Set[Tuple[str, str]] = set()
    for (county_code, precinct_code), names in sorted(candidates.items()):
        county_name = county_names.get(county_code, county_code)
        # Prefer the most frequently reported official label.  Stable alphabetical
        # tie-breaking keeps rebuilds deterministic.
        best_name = sorted(names.items(), key=lambda item: (-item[1], item[0].upper()))[0][0]
        counties.setdefault(county_name, {})[precinct_code] = best_name
        matched_geometry_keys.add((county_code, precinct_code))

    geometry_keys = {
        (county_code, precinct_code)
        for county_code, codes in codes_by_county.items()
        for precinct_code in codes
    }
    stats = {
        "counties": len(counties),
        "friendly_names": len(matched_geometry_keys),
        "geometry_precincts": len(geometry_keys),
        "coverage_pct": round(
            (len(matched_geometry_keys) / len(geometry_keys) * 100.0) if geometry_keys else 0.0,
            2,
        ),
        "unmatched_geometry_precincts": len(geometry_keys - matched_geometry_keys),
        "official_source_precinct_keys": len(source_keys),
    }
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_from": [
            str(source_zip.as_posix()),
            str(geometry_path.as_posix()),
        ],
        "source_url": OFFICIAL_SOURCE_URL,
        "counties": {
            county: dict(sorted(code_map.items()))
            for county, code_map in sorted(counties.items())
        },
    }
    return payload, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Florida precinct friendly-name index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--source-zip", default="2024GenOutputOfficial.zip")
    parser.add_argument("--geometry", default="fl_precinct_centroids.geojson")
    parser.add_argument("--output", default="precinct_friendly_names.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    source_zip = Path(args.source_zip).expanduser()
    geometry_path = Path(args.geometry).expanduser()
    output_path = Path(args.output).expanduser()
    if not source_zip.is_absolute():
        source_zip = data_dir / source_zip
    if not geometry_path.is_absolute():
        geometry_path = data_dir / geometry_path
    if not output_path.is_absolute():
        output_path = data_dir / output_path

    if not source_zip.exists():
        parser.error(f"missing official precinct archive: {source_zip}")
    if not geometry_path.exists():
        parser.error(f"missing precinct geometry: {geometry_path}")

    payload, stats = build_friendly_names(source_zip, geometry_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
