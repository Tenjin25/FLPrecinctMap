## County Slice Builder

Build county-level statewide contest slices:

```powershell
py scripts/build_fl_county_contests.py
```

This writes:
- `data/contests/*.json`
- `data/contests/manifest.json`
- `data/fl_elections_aggregated.json`

Source behavior:
- Prefers VEST precinct shapefiles when available.
- Backfills missing county years from `data/*Election-aligned.txt` DOS county files.
- Includes historic statewide offices present in aligned files (for example `secretary_of_state`, `comptroller`, `commissioner_of_education`).
- Florida Cabinet reform for these offices was approved on November 3, 1998 (Amendment 8) and took effect on January 7, 2003.
- DOS aligned files are county-level only and do not directly produce district or precinct slices.

## District Slice Builder

Use `build_fl_district_contests.py` to generate `data/district_contests/*.json` and `manifest.json` from VEST Florida shapefiles.

### 1) Blockfile + Crosswalk mode (recommended)

```powershell
py scripts/build_fl_district_contests.py `
  --allocation-method block `
  --block-id-year 2020 `
  --congressional-block-crosswalk data/crosswalks/congressional_block_to_district.csv `
  --state-house-block-crosswalk data/crosswalks/state_house_block_to_district.csv `
  --state-senate-block-crosswalk data/crosswalks/state_senate_block_to_district.csv `
  --blockfile-template "blockfiles/fl_blocks_{year}.csv"
```

If your blockfiles are 2010 GEOIDs, add:

```powershell
--block-id-year 2010 --nhgis-2010-2020 data/nhgis_blk2010_blk2020_12.zip
```

### 2) Precomputed precinct weights mode

```powershell
py scripts/build_fl_district_contests.py `
  --allocation-method precinct_weights `
  --congressional-weights data/crosswalks/congressional_precinct_weights.csv `
  --state-house-weights data/crosswalks/state_house_precinct_weights.csv `
  --state-senate-weights data/crosswalks/state_senate_precinct_weights.csv
```

### 3) Spatial fallback mode

```powershell
py scripts/build_fl_district_contests.py --allocation-method spatial
```

### Dry run

```powershell
py scripts/build_fl_district_contests.py --allocation-method spatial --dry-run
```

## Actual Legislative District Contests

Build chamber-native State House / State Senate slices from precinct text files and DOS files:

```powershell
py scripts/build_fl_actual_legislative_contests.py --data-dir data
```

This writes files like:
- `data/district_contests/state_house_state_house_2020.json`
- `data/district_contests/state_senate_state_senate_2020.json`

and updates:
- `data/district_contests/manifest.json`

## Comprehensive Precinct CSV Exports

Build long-form CSVs from all discovered precinct result folders:

```powershell
py scripts/build_fl_precinct_master_csv.py --data-dir data --output-dir derived
```

Outputs include:
- `data/derived/fl_precinct_results_<year>_long.csv`
- `data/derived/fl_precinct_results_all_years_long.csv`
- `data/derived/fl_precinct_legislative_all_years_long.csv`

## Precinct Map + Friendly Names

Build the county-scoped friendly-name index from Florida's official 2024
precinct-results archive, then rebuild centroids and simplified polygons so the
same names are embedded in the map features:

```powershell
py scripts/build_fl_precinct_friendly_names.py
py scripts/build_fl_precinct_centroids.py
```

The friendly-name JSON intentionally matches NCPrecinctMap's
`counties -> precinct_code -> display_name` schema. The official archive is
available from the Florida Division of Elections precinct-level results page.

## DRA-Backed VTD2010 Crosswalk Pipeline

Use `build_fl_vtd10_crosswalks_from_dra.py` to rebuild the Florida VTD2010 crosswalks with DRA's corrected Florida 2020 VTD GeoJSON from GitHub plus Census/TIGER geometry downloads.

```powershell
& .\.venv\Scripts\python.exe scripts/build_fl_vtd10_crosswalks_from_dra.py
```

This writes:
- `data/crosswalks/vtd10_to_vtd20_weights.csv`
- `data/crosswalks/vtd10_to_congressional_current_weights.csv`
- `data/crosswalks/vtd10_to_congressional_proposed_weights.csv`
- `data/crosswalks/vtd10_to_state_house_weights.csv`
- `data/crosswalks/vtd10_to_state_senate_weights.csv`

Cached source downloads are stored under:
- `data/sources/dra_vtd_data/`
