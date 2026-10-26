# Packaging the model outputs for the data deposit

These scripts turn the model outputs into the compact, CF-compliant files that make up the
SCALES-Global data deposit. They read the outputs in place and write the package to
`data/outputs/zenodo/`.

## Prerequisites

The model outputs must already exist:

- annual grids `data/outputs/grids/surplus_N_<year>_0p05.nc` (scripts 26 and 27b);
- national tables `data/outputs/mass_balance_N_<year>.csv` and `data/outputs/timeseries_N_global.csv`
  (scripts 17 and 27b);
- Monte Carlo outputs in `data/outputs/uncertainty_mc/` (scripts 46 and 47).

## Build the package

From the repository root:

```bash
python zenodo_prep/repackage_grids.py selftest     # quick check on synthetic data
python zenodo_prep/consolidate_tables.py selftest
bash zenodo_prep/run_all.sh
```

## Contents of `data/outputs/zenodo/`

```
full/           per-variable time-stacked cubes for all grid variables (1961-2023)
                  <var>_1961_2023.nc, dimensions (time, lat, lon)
                  component masses (*_kg) and cropland_ha: float32, lossless
                  surplus rates (*_kgha): int16 packed with scale/offset (~0.1 kg/ha step)
lite/           the surplus-rate cubes plus cropland_ha and cell_area_ha
uncertainty/    gridded uncertainty for 1961, 1965, ..., 2020 and 2023
tables/
  national_budget_N_1961_2023.csv     one row per (ISO3, year): budget terms, surpluses, NUE,
                                      and surplus uncertainty (mean, sd, 95% interval, CV)
  global_timeseries_N_1961_2023.csv   global annual totals with the Monte Carlo band
  per_year_national_N_1961_2023.zip  the per-year national CSVs (zipped; Zenodo allows 100 files per record)
```

All NetCDF files follow CF-1.10 on EPSG:4326, with a `crs` grid-mapping variable, a mid-year `time`
coordinate, an integer `year` coordinate, units and long names.

## Notes

- Component masses are stored losslessly, so per-cell values sum to the national budget. Only the
  surplus-rate maps are packed to int16; the packing range is each variable's true minimum and maximum.
- The per-hectare cropland surplus is set to missing in cells with less than 0.1% of their area in
  cropland, or with an absolute rate above 30,000 kg N ha-1 yr-1 (`repackage_grids.py`,
  `MASK_MIN_CROPLAND_FRAC` and `MASK_ABS_RATE_CAP`). Mass variables are not affected.
- The coefficient of variation is unstable for countries whose surplus is close to zero; the standard
  deviation and the 2.5th and 97.5th percentiles are the robust summary there.
- `upload_to_zenodo.py` uploads the package to Zenodo (see its docstring; needs a personal access token).
- `run_all.sh` also copies `DEPOSIT_README.md` into the package as `README.md` and writes the data dictionary
  (`build_data_dictionary.py`).
- Re-running is safe; delete `data/outputs/zenodo/` to start from scratch.
