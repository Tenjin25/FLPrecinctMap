#!/usr/bin/env python3
"""Allocate RDH/VEST fl_2012 precinct results onto district layers via spatial join."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def merge_manifest(out_dir: Path, new_entries: list[dict]) -> None:
    path = out_dir / "manifest.json"
    existing: list[dict] = []
    if path.exists():
        existing = list(json.loads(path.read_text(encoding="utf-8")).get("files") or [])
    keyed = {
        (e.get("scope"), e.get("contest_type"), int(e.get("year"))): e
        for e in existing
        if e.get("year") is not None
    }
    for entry in new_entries:
        keyed[(entry["scope"], entry["contest_type"], int(entry["year"]))] = entry
    merged = sorted(keyed.values(), key=lambda r: (r["scope"], r["contest_type"], int(r["year"])))
    path.write_text(json.dumps({"files": merged}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    py = sys.executable
    data = Path("data")
    jobs = [
        {
            "name": "current",
            "out": data / "district_contests",
            "scopes": "congressional,state_house,state_senate",
            "cd": data / "fl_congressional_districts.geojson",
            "hd": data / "fl_state_house_districts.geojson",
            "sd": data / "fl_state_senate_districts.geojson",
        },
        {
            "name": "proposed",
            "out": data / "district_contests_proposed_congressional",
            "scopes": "congressional",
            "cd": data / "fl_proposed_congressional_districts.geojson",
            "hd": data / "fl_state_house_districts.geojson",
            "sd": data / "fl_state_senate_districts.geojson",
        },
        {
            "name": "2026",
            "out": data / "district_contests_2026_congressional",
            "scopes": "congressional",
            "cd": data / "fl_congressional_districts_2026.geojson",
            "hd": data / "fl_state_house_districts.geojson",
            "sd": data / "fl_state_senate_districts.geojson",
        },
    ]

    for job in jobs:
        tmp = data / f"_tmp_2012_{job['name']}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        cmd = [
            py,
            "scripts/build_fl_district_contests.py",
            "--data-dir",
            "data",
            "--years",
            "2012",
            "--scopes",
            job["scopes"],
            "--allocation-method",
            "spatial",
            "--shapefile-template",
            "fl_2012.zip",
            "--congressional-geojson",
            str(job["cd"]),
            "--state-house-geojson",
            str(job["hd"]),
            "--state-senate-geojson",
            str(job["sd"]),
            "--output-dir",
            str(tmp),
        ]
        print(f"RUN {job['name']} ...")
        subprocess.check_call(cmd)
        man = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
        job["out"].mkdir(parents=True, exist_ok=True)
        for entry in man["files"]:
            src = tmp / entry["file"]
            dst = job["out"] / entry["file"]
            shutil.copy2(src, dst)
            print(
                f"  {entry['file']}: cov={entry.get('match_coverage_pct')} "
                f"D={entry['dem_total']:,} R={entry['rep_total']:,}"
            )
        merge_manifest(job["out"], man["files"])
        shutil.rmtree(tmp)
        print(f"merged into {job['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
