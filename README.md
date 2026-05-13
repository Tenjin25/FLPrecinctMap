# Sunshine State Ballot Atlas (FLPrecinctMap)

Interactive Florida election map published as a static site on GitHub Pages.

## What This Project Is

`FLPrecinctMap` is a front-end-only election explorer built from:
- `index.html` (app UI + map logic)
- static JSON/CSV/GeoJSON data files in `data/`

No backend service is required for production deployment.

## Live Deployment

- Hosted on GitHub Pages from this repository
- Static asset delivery only (`html`, `json`, `csv`, `geojson`, etc.)

## Main Features

- Statewide and county election views
- Congressional, State House, and State Senate views
- District line comparison (current vs proposed congressional lines)
- Precinct-aware overlays and crosswalk-backed district allocation views
- Trend and competitiveness summaries

## Repository Layout

- `index.html`: Main application
- `data/`: Election data, map layers, crosswalks, and generated outputs
- `scripts/`: Data build/refresh scripts
- `NCMap.html`: Design/reference companion file

## Required Configuration

Set your Mapbox public token before app initialization:

```html
<script>
  window.MAPBOX_TOKEN = "YOUR_MAPBOX_PUBLIC_TOKEN";
</script>
```

If `window.MAPBOX_TOKEN` is empty, basemap tiles will not render.

## Data Outputs Used by the App

Key files and folders:
- `data/contests/*.json`
- `data/district_contests/*.json`
- `data/district_contests_proposed_congressional/*.json`
- `data/crosswalks/*.csv`
- `data/fl_precinct_centroids.geojson`
- `data/fl_elections_aggregated.json`

## Updating Data (Optional)

If you need to rebuild or refresh data, use the scripts in `scripts/`.

Common workflow:
1. Build or refresh contest/crosswalk outputs.
2. Verify generated files under `data/`.
3. Commit updated static assets.
4. Push to `main` to publish through GitHub Pages.

Detailed script documentation:
- `scripts/README.md`

## Notes

- Close-race display uses adaptive precision so very small non-zero leads are not shown as ties.
- Candidate legacy shorthand normalization is included (for example, `GORE` -> `Al Gore`, `BUSH` -> `George W. Bush`).

## Recent Comprehensive Update (May 2026)

This project received a full UI/app integration update to make the Florida atlas the active production index while preserving the richer controls and interaction model from the NC companion build.

### 1. App Base + Florida Hookup Transplant

- `index.html` was rebuilt from the `NCMap.html` structure as the functional base.
- Florida-specific data/config hooks were transplanted from `index - Copy.html`, including:
  - Florida county/district layer paths
  - Florida contest and district contest directories
  - Florida crosswalk paths
  - Florida map center/bounds and district-line behavior

### 2. Florida Control Surface + Button Layout

- Main controls were aligned to the Florida UX:
  - Quick Jump buttons: Panhandle, North FL, Central FL, Tampa Bay, Southwest, South FL
  - District line toggles: `Current` and `Proposed` (2026 control removed from UI wiring)
- Residual NC-facing visible labels were cleaned from the active Florida experience.

### 3. Theme and Visual Alignment

- Theme token and control styling were aligned to the Florida copy implementation.
- The controls title pill (`.controls-title-pill`) and related header styles were corrected to match the copy-file look and gradient/shadow treatment.

### 4. Candidate Label Logic Alignment

- Trend/winner label logic in `index.html` was aligned with the copy-file behavior:
  - Candidate callouts now use direct contest row candidate fields through `shortCandidateLabel(...)`.
  - Winner summary strings use `winnerMarginLabelShort(...)` in trend contexts.

### 5. Stabilization + Error-Reduction Hardening

- Added path guards to avoid invalid fetches:
  - `loadJSON(path)` now throws when path is missing/blank.
  - `loadCSV(path)` now throws when path is missing/blank.
- Optional loaders were made explicitly conditional on configured paths:
  - county demographics
  - county population estimates
  - precinct demographics
- NC OpenElections presidential CSV fallback is disabled in this Florida build:
  - `const PRESIDENT_OPEN_ELECTIONS_CSV = {};`

### 6. Support Files Added for Missing Optional Resources

The following files were added as minimal placeholders to prevent fetch failures and keep optional features non-fatal when full source datasets are not yet populated:

- `data/county_demographics_2020_dp1.json`
- `data/district_descriptions.json`
- `data/fl_congressional_districts.csv`
- `data/fl_state_house_districts.csv`
- `data/fl_state_senate_districts.csv`
- `data/fl_district_results_2022_lines.json`
- `data/fl_voting_precincts.geojson`

These are scaffolding assets and can be replaced with full production datasets at any time without changing app code.
