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
