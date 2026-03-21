#!/usr/bin/env python3
"""
Build county-level contest slices for the FL map app.

Primary source:
  - VEST precinct shapefiles (2014+ in this repository)

Supplemental source:
  - FL DOS county-level aligned text files (*Election-aligned.txt)

Outputs:
  data/contests/{contest_type}_{year}.json
  data/contests/manifest.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import geopandas as gpd
import pandas as pd


VOTE_COL_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{3}[A-Z][A-Z0-9]{3}$")

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

DOS_RACECODE_TO_CONTEST = {
    "PRE": "president",
    "USS": "us_senate",
    "GOV": "governor",
    "ATG": "attorney_general",
    "CFO": "treasurer",
    "TRE": "treasurer",
    "AGR": "agriculture_commissioner",
    "SEC": "secretary_of_state",
    "CMP": "comptroller",
    "EDU": "commissioner_of_education",
}

CORE_CONTEST_TYPES = {
    "president",
    "us_senate",
    "governor",
    "attorney_general",
    "treasurer",
    "agriculture_commissioner",
    "secretary_of_state",
    "comptroller",
    "commissioner_of_education",
}

CANDIDATE_CODE_TO_NAME = {
    "G14AGRDHAM": "Thaddeus Hamilton",
    "G14AGRRPUT": "Adam Putnam",
    "G14ATGDSHE": "George Sheldon",
    "G14ATGRBON": "Pam Bondi",
    "G14CFODRAN": "William Rankin",
    "G14CFORATW": "Jeff Atwater",
    "G14GOVDCRI": "Charlie Crist",
    "G14GOVRSCO": "Rick Scott",
    "G16PREDCLI": "Hillary Clinton",
    "G16PRERTRU": "Donald Trump",
    "G16USSDMUR": "Patrick Murphy",
    "G16USSRRUB": "Marco Rubio",
    "G18AGRDFRI": "Nikki Fried",
    "G18AGRRCAL": "Matt Caldwell",
    "G18ATGDSHA": "Sean Shaw",
    "G18ATGRMOO": "Ashley Moody",
    "G18CFODRIN": "Jeremy Ring",
    "G18CFORPAT": "Jimmy Patronis",
    "G18GOVDGIL": "Andrew Gillum",
    "G18GOVRDES": "Ron DeSantis",
    "G18USSDNEL": "Bill Nelson",
    "G18USSRSCO": "Rick Scott",
    "G20PREDBID": "Joe Biden",
    "G20PRERTRU": "Donald Trump",
    "G22AGRDBLE": "Naomi Blemur",
    "G22AGRRSIM": "Wilton Simpson",
    "G22ATGDAYA": "Aramis Ayala",
    "G22ATGRMOO": "Ashley Moody",
    "G22CFODHAT": "Adam Hattersley",
    "G22CFORPAT": "Jimmy Patronis",
    "G22GOVDCRI": "Charlie Crist",
    "G22GOVRDES": "Ron DeSantis",
    "G22USSDDEM": "Val Demings",
    "G22USSRRUB": "Marco Rubio",
    "G24PREDHAR": "Kamala Harris",
    "G24PRERTRU": "Donald Trump",
    "G24USSDMUC": "Debbie Mucarsel-Powell",
    "G24USSRSCO": "Rick Scott",
}


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


def clean_text(value: object) -> str:
    s = str(value or "").strip()
    return re.sub(r"\s+", " ", s)


def parse_year_from_text(value: object) -> Optional[int]:
    s = str(value or "")
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def parse_year_from_date(value: object) -> Optional[int]:
    text = clean_text(value)
    if not text:
        return None
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None
    return int(dt.year)


def normalize_party_bucket(party_code: object) -> str:
    p = clean_text(party_code).upper()
    if p in {"DEM", "D"}:
        return "dem"
    if p in {"REP", "R"}:
        return "rep"
    return "other"


def should_skip_candidate(name: object) -> bool:
    s = clean_text(name).lower()
    if not s:
        return True
    if s in {"-", "/"}:
        return True
    if "undervote" in s or "under vote" in s:
        return True
    if "overvote" in s or "over vote" in s:
        return True
    return False


def normalize_county_token(value: object) -> str:
    s = clean_text(value).upper()
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def infer_contest_from_office_desc(value: object) -> Optional[str]:
    s = clean_text(value).lower()
    if not s:
        return None
    if "president of the united states" in s:
        return "president"
    if "united states senator" in s:
        return "us_senate"
    if "secretary of state" in s:
        return "secretary_of_state"
    if "comptroller" in s:
        return "comptroller"
    if "commissioner of education" in s:
        return "commissioner_of_education"
    if "attorney general" in s:
        return "attorney_general"
    if "commissioner of agriculture" in s or "agriculture commissioner" in s:
        return "agriculture_commissioner"
    if "chief financial officer" in s or s == "treasurer":
        return "treasurer"
    if "governor" in s:
        return "governor"
    return None


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def build_candidate_name_from_aligned(df: pd.DataFrame) -> pd.Series:
    first = get_series(df, "CanNameFirst").map(clean_text)
    last = get_series(df, "CanNameLast").map(clean_text)
    office_desc = get_series(df, "OfficeDesc").map(lambda v: clean_text(v).lower())

    both = (first + " " + last).str.strip()
    candidate = both.mask((first == "") & (last != ""), last)
    candidate = candidate.mask((last == "") & (first != ""), first)
    candidate = candidate.mask((first == "") & (last == ""), "")

    # Presidential rows in this source are typically ticket-formatted.
    pres_mask = office_desc.str.contains("president of the united states", regex=False)
    pres_ticket = first
    pres_ticket = pres_ticket.mask(
        (first != "") & (last != "") & (first.str.lower() != last.str.lower()),
        first + " / " + last,
    )
    candidate = candidate.mask(pres_mask, pres_ticket)
    return candidate.map(clean_text)


def build_canonical_county_name_map(county_geojson: Path) -> Dict[str, str]:
    counties = gpd.read_file(county_geojson)[["NAME20"]].copy()
    counties = counties[counties["NAME20"].notna()]
    out: Dict[str, str] = {}
    for raw in counties["NAME20"].tolist():
        canonical = clean_text(raw).upper()
        token = normalize_county_token(canonical)
        if token:
            out[token] = canonical
    # Historical alias (Miami-Dade was Dade).
    out["DADE"] = "MIAMI-DADE"
    return out


def canonicalize_county_name(raw: object, canonical_map: Dict[str, str]) -> str:
    token = normalize_county_token(raw)
    if not token:
        return ""
    return canonical_map.get(token, clean_text(raw).upper())


def top_party_candidate(candidate_totals: pd.DataFrame, party_bucket: str, fallback: str) -> str:
    subset = candidate_totals[candidate_totals["party_bucket"] == party_bucket].copy()
    if subset.empty:
        return fallback
    subset = subset.sort_values(["votes", "candidate_name"], ascending=[False, True])
    name = clean_text(subset.iloc[0]["candidate_name"])
    return name or fallback


def build_county_row_payload(
    county: str,
    dem_votes: int,
    rep_votes: int,
    other_votes: int,
    dem_candidate: str,
    rep_candidate: str,
) -> dict:
    dv = int(dem_votes)
    rv = int(rep_votes)
    ov = int(other_votes)
    tv = dv + rv + ov
    margin = rv - dv
    margin_pct = (margin / tv * 100.0) if tv > 0 else 0.0
    winner = "REP" if rv > dv else ("DEM" if dv > rv else "TIE")
    return {
        "county": county,
        "dem_votes": dv,
        "rep_votes": rv,
        "other_votes": ov,
        "total_votes": tv,
        "dem_candidate": dem_candidate,
        "rep_candidate": rep_candidate,
        "margin": margin,
        "margin_pct": round(margin_pct, 6),
        "winner": winner,
        "color": margin_color(winner, abs(margin_pct)),
    }


def load_aligned_county_file(path: Path, canonical_county_map: Dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin1", low_memory=False)
    except Exception as exc:
        print(f"[WARN] Failed reading aligned county file {path.name}: {exc}")
        return pd.DataFrame(
            columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"]
        )

    if df.empty:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    df.columns = [clean_text(c) for c in df.columns]
    if "CanVotes" not in df.columns or "CountyName" not in df.columns:
        print(f"[WARN] Skipping {path.name}: missing CanVotes/CountyName columns.")
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    year: Optional[int] = None
    if "ElectionDate" in df.columns:
        for value in get_series(df, "ElectionDate"):
            year = parse_year_from_date(value)
            if year is not None:
                break
    if year is None:
        year = parse_year_from_text(path.name)
    if year is None:
        print(f"[WARN] Skipping {path.name}: could not infer year.")
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    race_code = get_series(df, "RaceCode").map(lambda v: clean_text(v).upper())
    contest_type = race_code.map(DOS_RACECODE_TO_CONTEST)
    office_mapped = get_series(df, "OfficeDesc").map(infer_contest_from_office_desc)
    contest_type = contest_type.fillna(office_mapped)
    df["contest_type"] = contest_type
    df = df[df["contest_type"].isin(CORE_CONTEST_TYPES)].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    df["county"] = get_series(df, "CountyName").map(
        lambda v: canonicalize_county_name(v, canonical_county_map)
    )
    df = df[df["county"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    df["votes"] = pd.to_numeric(get_series(df, "CanVotes"), errors="coerce").fillna(0).astype(int)
    df = df[df["votes"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    df["party_bucket"] = get_series(df, "PartyCode").map(normalize_party_bucket)
    df["candidate_name"] = build_candidate_name_from_aligned(df)
    df = df[~df["candidate_name"].map(should_skip_candidate)].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])

    df["year"] = int(year)
    return df[["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"]]


def load_all_aligned_county_records(data_dir: Path, canonical_county_map: Dict[str, str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*Election-aligned.txt")):
        part = load_aligned_county_file(path, canonical_county_map)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["year", "contest_type", "county", "party_bucket", "candidate_name", "votes"])
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


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


def infer_candidate_name(
    df: pd.DataFrame,
    columns: List[str],
    party_label: str,
) -> str:
    if not columns:
        return party_label
    totals = {}
    for col in columns:
        totals[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    top_col = max(totals, key=totals.get)
    return CANDIDATE_CODE_TO_NAME.get(top_col, party_label)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def build_county_code_map(shapefile_zip: Path, county_geojson: Path) -> Dict[str, str]:
    precincts = gpd.read_file(shapefile_zip)[["COUNTY", "geometry"]].copy()
    counties = gpd.read_file(county_geojson)[["NAME20", "geometry"]].copy()
    precincts = precincts[precincts["COUNTY"].notna() & precincts.geometry.notna()]
    counties = counties[counties["NAME20"].notna() & counties.geometry.notna()]

    if precincts.crs is None:
        precincts = precincts.set_crs(4326)
    if counties.crs is None:
        counties = counties.set_crs(4326)
    if precincts.crs != counties.crs:
        counties = counties.to_crs(precincts.crs)

    joined = gpd.sjoin(precincts, counties, how="left", predicate="intersects")
    joined = joined[joined["NAME20"].notna()]
    mode_map = (
        joined.groupby("COUNTY")["NAME20"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {str(k).strip().upper(): str(v).strip().upper() for k, v in mode_map.items()}


def build() -> None:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    contests_dir = data_dir / "contests"
    contests_dir.mkdir(parents=True, exist_ok=True)

    years = [2014, 2016, 2018, 2020, 2022, 2024]
    county_geojson = data_dir / "tl_2020_12_county20.geojson"
    county_map = build_county_code_map(
        data_dir / "fl_2024.zip",
        county_geojson,
    )
    canonical_county_map = build_canonical_county_name_map(county_geojson)

    manifest_entries = []
    results_by_year: Dict[str, Dict[str, dict]] = {}
    built = 0
    built_keys: Set[Tuple[str, int]] = set()

    for year in years:
        shp = data_dir / f"fl_{year}.zip"
        if not shp.exists():
            continue
        df = gpd.read_file(shp, ignore_geometry=True)
        if "COUNTY" not in df.columns:
            continue
        df["county_name"] = df["COUNTY"].astype(str).str.strip().str.upper().map(county_map)
        df = df[df["county_name"].notna()]

        contests = parse_contest_columns(df.columns)
        for contest_type, cols in sorted(contests.items()):
            dem_candidate = infer_candidate_name(df, cols["dem"], "Democrat")
            rep_candidate = infer_candidate_name(df, cols["rep"], "Republican")

            dem = sum_numeric(df, cols["dem"])
            rep = sum_numeric(df, cols["rep"])
            other = sum_numeric(df, cols["other"])
            total = dem + rep + other

            rows = pd.DataFrame(
                {
                    "county": df["county_name"],
                    "dem_votes": dem,
                    "rep_votes": rep,
                    "other_votes": other,
                    "total_votes": total,
                }
            )
            rows = rows.groupby("county", as_index=False)[["dem_votes", "rep_votes", "other_votes", "total_votes"]].sum()
            rows = rows.sort_values("county")
            if rows.empty:
                continue

            payload_rows = []
            county_results = {}
            for _, r in rows.iterrows():
                dv = int(round(float(r["dem_votes"])))
                rv = int(round(float(r["rep_votes"])))
                ov = int(round(float(r["other_votes"])))
                tv = dv + rv + ov
                margin = rv - dv
                margin_pct = (margin / tv * 100.0) if tv > 0 else 0.0
                winner = "REP" if rv > dv else ("DEM" if dv > rv else "TIE")
                row_payload = {
                    "county": r["county"],
                    "dem_votes": dv,
                    "rep_votes": rv,
                    "other_votes": ov,
                    "total_votes": tv,
                    "dem_candidate": dem_candidate,
                    "rep_candidate": rep_candidate,
                    "margin": margin,
                    "margin_pct": round(margin_pct, 6),
                    "winner": winner,
                    "color": margin_color(winner, abs(margin_pct)),
                }
                payload_rows.append(row_payload)
                county_results[r["county"]] = {
                    "dem_votes": dv,
                    "rep_votes": rv,
                    "other_votes": ov,
                    "total_votes": tv,
                    "dem_candidate": dem_candidate,
                    "rep_candidate": rep_candidate,
                    "margin": margin,
                    "margin_pct": round(margin_pct, 6),
                    "winner": winner,
                    "competitiveness": {"color": row_payload["color"]},
                }

            filename = f"{contest_type}_{year}.json"
            write_json(contests_dir / filename, {"rows": payload_rows})
            built += 1
            built_keys.add((contest_type, int(year)))

            dem_total = int(sum(r["dem_votes"] for r in payload_rows))
            rep_total = int(sum(r["rep_votes"] for r in payload_rows))
            other_total = int(sum(r["other_votes"] for r in payload_rows))
            manifest_entries.append(
                {
                    "year": year,
                    "contest_type": contest_type,
                    "file": filename,
                    "rows": len(payload_rows),
                    "dem_total": dem_total,
                    "rep_total": rep_total,
                    "other_total": other_total,
                    "major_party_contested": bool(dem_total > 0 and rep_total > 0),
                }
            )
            contest_key = f"{contest_type}_{year}"
            results_by_year.setdefault(str(year), {}).setdefault(contest_type, {})[contest_key] = {
                "contest_type": contest_type,
                "year": int(year),
                "results": county_results,
            }

    aligned_records = load_all_aligned_county_records(data_dir, canonical_county_map)
    if not aligned_records.empty:
        grouped = aligned_records.groupby(["year", "contest_type"], as_index=False)
        for (year, contest_type), subset in grouped:
            key = (str(contest_type), int(year))
            if key in built_keys:
                continue

            candidate_totals = (
                subset.groupby(["party_bucket", "candidate_name"], as_index=False)["votes"]
                .sum()
            )
            dem_candidate = top_party_candidate(candidate_totals, "dem", "Democrat")
            rep_candidate = top_party_candidate(candidate_totals, "rep", "Republican")

            county_party = subset.groupby(["county", "party_bucket"], as_index=False)["votes"].sum()
            pivot = county_party.pivot(index="county", columns="party_bucket", values="votes").fillna(0)
            for col in ("dem", "rep", "other"):
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot.reset_index().sort_values("county")
            if pivot.empty:
                continue

            payload_rows = []
            county_results = {}
            for _, r in pivot.iterrows():
                row_payload = build_county_row_payload(
                    county=str(r["county"]).strip().upper(),
                    dem_votes=int(round(float(r.get("dem", 0)))),
                    rep_votes=int(round(float(r.get("rep", 0)))),
                    other_votes=int(round(float(r.get("other", 0)))),
                    dem_candidate=dem_candidate,
                    rep_candidate=rep_candidate,
                )
                payload_rows.append(row_payload)
                county_results[row_payload["county"]] = {
                    "dem_votes": row_payload["dem_votes"],
                    "rep_votes": row_payload["rep_votes"],
                    "other_votes": row_payload["other_votes"],
                    "total_votes": row_payload["total_votes"],
                    "dem_candidate": row_payload["dem_candidate"],
                    "rep_candidate": row_payload["rep_candidate"],
                    "margin": row_payload["margin"],
                    "margin_pct": row_payload["margin_pct"],
                    "winner": row_payload["winner"],
                    "competitiveness": {"color": row_payload["color"]},
                }

            filename = f"{contest_type}_{year}.json"
            write_json(contests_dir / filename, {"rows": payload_rows})
            built += 1
            built_keys.add(key)

            dem_total = int(sum(r["dem_votes"] for r in payload_rows))
            rep_total = int(sum(r["rep_votes"] for r in payload_rows))
            other_total = int(sum(r["other_votes"] for r in payload_rows))
            manifest_entries.append(
                {
                    "year": int(year),
                    "contest_type": str(contest_type),
                    "file": filename,
                    "rows": len(payload_rows),
                    "dem_total": dem_total,
                    "rep_total": rep_total,
                    "other_total": other_total,
                    "major_party_contested": bool(dem_total > 0 and rep_total > 0),
                }
            )
            contest_key = f"{contest_type}_{year}"
            results_by_year.setdefault(str(year), {}).setdefault(str(contest_type), {})[contest_key] = {
                "contest_type": str(contest_type),
                "year": int(year),
                "results": county_results,
            }

    manifest_entries.sort(key=lambda e: (e["contest_type"], int(e["year"])))
    write_json(contests_dir / "manifest.json", {"files": manifest_entries})
    write_json(
        data_dir / "fl_elections_aggregated.json",
        {
            "state": "FL",
            "source": "VEST (University of Florida) + FL DOS aligned county files",
            "results_by_year": results_by_year,
        },
    )

    print(
        json.dumps(
            {
                "contests_built": built,
                "manifest_entries": len(manifest_entries),
                "aggregated_path": str(data_dir / "fl_elections_aggregated.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
