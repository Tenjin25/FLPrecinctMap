# FLPrecinctMap

Interactive Florida election mapping app focused on:
- County, congressional, State House, and State Senate views
- Current vs proposed congressional lines
- Precinct-aware crosswalk allocations
- Real legislative district election results (House/Senate) across multiple years

The app is a static `index.html` + data files project, with Python scripts to build/refresh datasets.

## Current Scope

- Main UI title: **Sunshine State Ballot Atlas**
- Supports district views:
  - Congressional
  - State House
  - State Senate
- Legislative district contest slices currently built for:
  - `2012, 2014, 2016, 2018, 2020, 2022, 2024`

## Repository Layout

- `index.html`: main application
- `data/`: input and generated map/election data
- `scripts/`: data build and validation scripts

Key generated outputs used by the app:
- `data/contests/*.json`: statewide/county contest slices
- `data/district_contests/*.json`: district contest slices + `manifest.json`
- `data/district_contests_proposed_congressional/*.json`: proposed-congressional slices
- `data/crosswalks/*.csv`: precinct/block/district weighting tables
- `data/fl_precinct_centroids.geojson`: precinct centroid layer

## Requirements

- Python 3.10+ (tested with Python 3.14)
- Python packages:
  - `pandas`
  - `geopandas`

Install example:

```powershell
python -m pip install pandas geopandas
```

## Mapbox Token Setup (Required)

`index.html` now expects a token from `window.MAPBOX_TOKEN`.

Current config:
- `index.html` uses `mapboxToken: (window.MAPBOX_TOKEN || '')`

Set it before the main script, for example near the top of `<body>`:

```html
<script>
  window.MAPBOX_TOKEN = "YOUR_MAPBOX_PUBLIC_TOKEN";
</script>
```

If token is empty, basemap tiles will not load.

## Run Locally

From repo root:

```powershell
python -m http.server 8000
```

Then open:
- `http://localhost:8000/index.html`

## Data Build Workflows

### 1) County Contest Slices

Builds:
- `data/contests/*.json`
- `data/fl_elections_aggregated.json`

Run:

```powershell
python scripts/build_fl_county_contests.py
```

### 2) Precinct Centroids

Builds:
- `data/fl_precinct_centroids.geojson`

Run:

```powershell
python scripts/build_fl_precinct_centroids.py --data-dir data
```

### 3) VTD2010 Crosswalks (2012/2014 reconciliation + districts)

Builds:
- `data/crosswalks/vtd10_to_vtd20_weights.csv`
- `data/crosswalks/vtd10_to_congressional_current_weights.csv`
- `data/crosswalks/vtd10_to_congressional_proposed_weights.csv`
- `data/crosswalks/vtd10_to_state_house_weights.csv`
- `data/crosswalks/vtd10_to_state_senate_weights.csv`

Run:

```powershell
python scripts/build_fl_vtd10_crosswalks.py --data-dir data --output-dir crosswalks
```

### 4) District Contest Slices (allocation-based)

General script:

```powershell
python scripts/build_fl_district_contests.py --help
```

Typical spatial run:

```powershell
python scripts/build_fl_district_contests.py --allocation-method spatial --data-dir data
```

Proposed congressional run:

```powershell
python scripts/build_fl_district_contests.py `
  --allocation-method spatial `
  --scopes congressional `
  --congressional-geojson data/fl_proposed_congressional_districts.geojson `
  --output-dir data/district_contests_proposed_congressional
```

### 5) Actual Legislative Elections (House/Senate district-native)

Builds chamber-native contest slices:
- `state_house_state_house_<year>.json`
- `state_senate_state_senate_<year>.json`

Run:

```powershell
python scripts/build_fl_actual_legislative_contests.py --data-dir data
```

This updates:
- `data/district_contests/manifest.json`

### 6) Validate Proposed Congressional Slices

Run:

```powershell
python scripts/validate_proposed_congressional_data.py --data-dir data
```

### 7) Comprehensive Precinct CSV Exports

Builds long-form CSVs from discovered precinct text folders:
- `data/derived/fl_precinct_results_<year>_long.csv`
- `data/derived/fl_precinct_results_all_years_long.csv`
- `data/derived/fl_precinct_legislative_all_years_long.csv`

Run:

```powershell
python scripts/build_fl_precinct_master_csv.py --data-dir data --output-dir derived
```

## Adding More Years

For automatic pickup in the legislative and CSV scripts:
- Add folders under `data/` containing county-level precinct text files (`*PctResults*.txt`)
- Ensure year appears in folder name or file names (for detection)
- Re-run:
  - `python scripts/build_fl_actual_legislative_contests.py --data-dir data`
  - `python scripts/build_fl_precinct_master_csv.py --data-dir data --output-dir derived`

Notes:
- Recount files are preferred when both base and recount exist for a county.
- If both precinct-level and DOS data exist for the same year, precinct-level data is preferred.

## UI/Data Notes

- Florida office filtering is applied (CFO instead of Treasurer; no FL SoS/Labor/Insurance/Superintendent statewide contest entries).
- Governor/Lt. Governor treated as joint ticket.
- District line toggle supports congressional current vs proposed mapping.
- District carryover crosswalk loader supports both:
  - `area_weight` style files
  - `weight` style files (`from_vtd10` + `district`)

## Troubleshooting

- Basemap not loading:
  - Set `window.MAPBOX_TOKEN` with a valid public token.
- Contest appears in dropdown but not on map:
  - Check `data/district_contests/manifest.json` for matching `scope`, `contest_type`, `year`.
- Crosswalk mismatch:
  - Rebuild with `build_fl_vtd10_crosswalks.py` and verify files in `data/crosswalks/`.
- Proposed congressional consistency:
  - Run `validate_proposed_congressional_data.py`.

## Script Reference

See:
- `scripts/README.md`
