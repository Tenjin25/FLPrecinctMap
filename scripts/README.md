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
