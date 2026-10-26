# SCALES-Global: annual 0.05° global nitrogen budgets and surpluses, 1961–2023

**Authors:** Kimberly J. Van Meter (The Pennsylvania State University) and Nandita B. Basu (University of Waterloo).
**Corresponding author:** Kimberly Van Meter (vanmeterlab@gmail.com).
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

This record accompanies the data descriptor *SCALES-Global* (Van Meter and Basu, *Scientific Data*, in review).

## What the dataset is

SCALES-Global (Scalable Computation of Anthropogenic Land Element Surpluses) is an annual global nitrogen (N)
budget for 1961–2023. National budgets are built from FAOSTAT activity data and published coefficients, and each
budget term is downscaled to a 0.05° grid (about 5.5 km) with process-specific spatial proxies that conserve the
national totals. The budget covers synthetic fertilizer, biological N fixation, atmospheric deposition, livestock
manure, human-derived N (food waste, night soil, biosolids and wastewater), crop removal and forage removal from
grazing land. Three surplus views are provided:

- **cropland surplus** (`surplus_cropland_kgha`), per hectare of cropland;
- **landscape surplus** (`surplus_landscape_kgha`), per hectare of land, including grazing land and terrestrial
  human-N pathways;
- **total anthropogenic surplus** (`surplus_full_kgha`), the landscape surplus plus wastewater N discharged to
  surface water.

Uncertainty is estimated with a coefficient-level Monte Carlo analysis (10,000 realizations per year) of the national
budgets and propagated to every grid cell.

## Contents

| Folder (file prefix) | Contents |
|---|---|
| `full/` | Every grid variable as a time-stacked NetCDF file (`<var>_1961_2023.nc`; time x lat x lon): component N fluxes (kg N per cell per year), cropland area and fraction, and the three surplus views (kg N ha⁻¹ yr⁻¹). |
| `lite/` | Mapping subset: the three surplus views, `cropland_ha` and a geodesic `cell_area_ha`. |
| `uncertainty/` | Gridded surplus uncertainty for 1961, 1965, …, 2020 and 2023: Monte Carlo mean, standard deviation, its coefficient and activity-data parts, and the half-width of the 95% interval, for each surplus view. |
| `tables/national_budget_N_1961_2023.csv` | Country budgets, one row per country and year: all component fluxes, the three surplus views and their uncertainty. |
| `tables/global_timeseries_N_1961_2023.csv` | Global annual totals by component (Tg N) with the Monte Carlo interval of each surplus view. |
| `tables/per_year_national_N_1961_2023.zip` | The national budget tables, one file per year. |
| `DATA_DICTIONARY.md`, `data_dictionary.json` | Variable-by-variable documentation generated from the files. |

## File naming

Zenodo stores files as a flat list, so the folder is encoded as a file-name prefix joined by a double underscore:
for example `full/bnf_kg_1961_2023.nc` is stored as `full__bnf_kg_1961_2023.nc`.

## Format

Grids are CF-1.10 NetCDF-4 on EPSG:4326 (longitude −180 to 180, latitude −90 to 90), with a `crs` grid-mapping
variable, a mid-year time axis and an integer `year` coordinate. Mass variables are stored as float32 without loss, so
per-cell values sum to the national budgets; surplus rates are packed to int16 with a per-variable scale and offset,
which CF readers apply automatically.

```python
import xarray as xr
ds = xr.open_dataset("lite__surplus_full_kgha_1961_2023.nc")
ds["surplus_full_kgha"].sel(year=2020)
```

## Versions

Version 2 replaces all files of version 1. It incorporates the revisions made for the data descriptor: a
within-country pattern for cropland deposition, the HYDE 3.3 cropland extent for all cropland terms, crop-group
fertilizer placed on the same crop map as crop removal, a ceiling on manure applied per hectare of cropland,
forage removal evaluated on the grid, and a coefficient-level Monte Carlo uncertainty analysis.

## Citation

Van Meter, K. J. and Basu, N. B. SCALES-Global: annual 0.05° global nitrogen budgets and surpluses, 1961–2023
(version 2). Zenodo (2026). [DOI of this version]

## Code

The model code is available at [GitHub repository] and archived at [code DOI].
