# SCALES-Global

Code for **SCALES-Global**, an annual (1961–2023) global nitrogen budget and nitrogen-surplus dataset at
0.05° resolution. SCALES (Scalable Computation of Anthropogenic Land Element Surpluses) builds national
nitrogen budgets from FAOSTAT activity data and published coefficients, and downscales each budget term
to the grid with process-specific spatial proxies while conserving national totals. Three surplus views
are produced:

- **cropland surplus**, per hectare of cropland, comparable with the FAO Cropland Nutrient Budget;
- **landscape surplus**, per hectare of land, adding grazing-land inputs and forage removal and
  terrestrial human-N pathways;
- **total anthropogenic surplus**, the landscape surplus plus wastewater N discharged to surface water.

Uncertainty is estimated by a coefficient-level Monte Carlo analysis of the national budgets and
propagated to the grid.

The dataset is archived on Zenodo: [DATA DOI]. The method is described in:

> Van Meter, K. J. & Basu, N. B. SCALES-Global: [title]. *Scientific Data* (in review).

## Repository layout

```
config/config.yaml        model settings, data paths and FAOSTAT codes
coefficients/             coefficient and ancillary tables (see coefficients/README_coefficients.md)
src/                      model modules (national budget, downscaling, Monte Carlo)
scripts/                  numbered scripts: data preparation, model runs, uncertainty, validation
scripts/tests/            unit tests on synthetic data (no input data needed)
zenodo_prep/              packaging of the outputs for the data deposit
data/validation/          small reference table used by the validation suite
run_pipeline.py           command-line entry point for single-year national budgets
```

Input data are not included (about 130 GB). They are downloaded into `data/raw/` with the scripts below.
All outputs are written to `data/outputs/`. Both locations can be changed under `paths` in
`config/config.yaml`.

## Installation

Python 3.10 or later.

```bash
conda create -n scales python=3.11
conda activate scales
pip install -r requirements.txt
python scripts/tests/test_uncertainty_mc.py      # quick check; needs no input data
```

## Input data

| Folder in `data/raw/` | Dataset | How to obtain |
|---|---|---|
| `faostat/` | FAOSTAT domains RFN, RFB, QCL, EMN, RL, ESB, OA, FBS, FBSH, MK | `scripts/01_download_faostat.py` |
| `ludemann_crop_coeffs/` | Crop nutrient concentrations, Ludemann et al. (Dryad, doi:10.5061/dryad.n2z34tn0x) | manual download, then `scripts/02_download_crop_coeffs.py` |
| `deposition/` | ISIMIP3a N deposition, NOy and NHx (doi:10.48364/ISIMIP.759077) | `scripts/04_download_deposition.py` |
| `vishwakarma2023/` | National cropland N deposition, Vishwakarma et al. (2023) (Dryad, doi:10.5061/dryad.msbcc2g1x) | manual download |
| `cropgrids/` | CROPGRIDS v1.08, Tang et al. (figshare, doi:10.6084/m9.figshare.22491997) | `scripts/06_download_cropgrids.py` |
| `npkgrids/` | NPKGRIDS v1.08 (figshare, doi:10.6084/m9.figshare.24616050) | `scripts/09_download_npkgrids.py` |
| `coello_fertilization/` | Crop-group fertilizer, Coello et al. (2025), *Sci. Data* 12:44 | manual download (see the paper's data availability statement) |
| `lu_tian_fertilizer/` | Lu and Tian (2016) N and P fertilizer (doi:10.1594/PANGAEA.863323) | `scripts/21_download_lu_tian.py` |
| `zhang_manure/` | Gridded manure N, Zhang et al. (2017) (PANGAEA 871980) | manual download, then `scripts/12c_build_zhang_manure.py` |
| `gridded_livestock/`, `glw/` | Gridded Livestock of the World (GLW3, GLW4) | `scripts/07_download_glw.py`, `07b_download_glw3.py`, `23_download_glw_epochs.py` |
| `hyde/` | HYDE 3.3 cropland, pasture, rangeland and population | `scripts/32_download_hyde.py` |
| `gpw/` | Global Pasture Watch grassland, Parente et al. (2024) (doi:10.1038/s41597-024-04139-6), aggregated to 0.05° | `scripts/gee_export_gpw_grassland_tiled.py` (Google Earth Engine) |
| `landcover/` | MODIS MCD12C1 v061 land cover | `scripts/12_download_landcover.py` (NASA Earthdata login) |
| `population_grid/` | GHS-POP R2023A population epochs | `scripts/08_download_population_grid.py`, `22_download_ghs_epochs.py` |
| `wastewater/` | Jones et al. (2021) wastewater production, collection and treatment (doi:10.1594/PANGAEA.918731); OECD wastewater treatment | `scripts/42_download_wastewater.py` |
| `gdp/` | Maddison Project Database GDP per capita | manual download (optional; FAOSTAT MK is used otherwise) |

Inputs used only by the validation suite:

| Folder in `data/raw/` | Dataset | How to obtain |
|---|---|---|
| `europe_batool2022/` | European soil N surplus, Batool et al. (2022) (doi:10.5281/zenodo.6581441) | `scripts/14_download_europe.py` |
| `hani/` | HaNi, Tian et al. (2022) (doi:10.1594/PANGAEA.942069) | `scripts/29_download_hani.py` |
| `NNI_brehob_et_al/` | EPA Next-Generation National Nutrient Inventory (doi:10.23719/1532436) | manual download |
| `NANI_LI_2024/` | Global NANI, Li et al. (2024), file `NANI2.tif` | manual download |
| `us_counties/` | US Census county boundaries (cb_2021_us_county_500k) | manual download |

The Tuholske et al. (2021) national sewered-N values used in validation (their 1 km effluent rasters,
KNB doi:10.5063/F76B09, summed by country) are included in `data/validation/`.

## Running the model

All commands run from the repository root. Steps must be run in this order.

**1. Derived inputs** (once, after the downloads):

```bash
python scripts/05_build_dep_ox.py
python scripts/31b_build_dep_total_byyear.py
python scripts/31c_build_dep_wl_byyear.py
python scripts/31d_build_dep_noncrop_byyear.py
python scripts/24_build_population_history.py
python scripts/45_build_treatment.py
python scripts/38_build_activity_cv.py
python scripts/12c_build_zhang_manure.py --pool both
python scripts/12d_build_coello_national_groups.py --nutrient N
```

These scripts write the tables in `coefficients/` that are already included in this repository (for
example `dep_wl_country_byyear.csv`, `sewer_connection.csv`, `wastewater_treatment.csv`), so they only
need to be run to rebuild those tables from source.

**2. National budgets, 1961–2023:**

```bash
python scripts/17_run_timeseries.py --start 1961 --end 2023
```

**3. Gridded budgets** (about one minute per year):

```bash
python scripts/26_run_historical_grids.py --start 1961 --end 2023
python scripts/27b_reconcile_national_with_grids.py --start 1961 --end 2023
python scripts/27_grid_summary.py
```

Script 27b makes the national tables consistent with the grids for the two terms computed on the grid
(forage removal and deposition on all land), so it must run after scripts 17 and 26.

**4. Uncertainty:**

```bash
python scripts/46_run_uncertainty_mc.py --start 1961 --end 2023     # national Monte Carlo, 10,000 draws
python scripts/47_gridded_uncertainty_cov.py --start 1961 --end 2023 # gridded uncertainty
python scripts/48_figure6_uncertainty.py --year 2023                 # uncertainty figure
```

Coefficient uncertainties are set in `coefficients/uncertainty_spec.csv` and
`coefficients/uncertainty_region_cv.csv`.

**5. Technical validation:**

```bash
python scripts/55_validation_suite.py
```

This writes Table 11, the per-test comparison tables and the validation figure to
`data/outputs/validation/`.

**6. Packaging for the data deposit** (optional): see `zenodo_prep/README_zenodo_prep.md`.

## Outputs

| File | Contents |
|---|---|
| `data/outputs/mass_balance_N_<year>.csv` | national budget terms, surpluses and land areas, one row per country |
| `data/outputs/timeseries_N_global.csv` | global annual totals |
| `data/outputs/grids/surplus_N_<year>_0p05.nc` | gridded budget terms (kg N per cell) and surpluses (kg N ha⁻¹) |
| `data/outputs/uncertainty_mc/uncertainty_N_<year>.csv` | national surplus uncertainty |
| `data/outputs/uncertainty_mc/uncertainty_N_timeseries.csv` | global surplus uncertainty, 1961–2023 |
| `data/outputs/uncertainty_mc/grids/uncertainty_N_<year>_0p05.nc` | gridded surplus uncertainty |
| `data/outputs/validation/` | validation tables and figure |

## Tests

```bash
for t in scripts/tests/test_*.py; do python "$t"; done
```

The tests use synthetic inputs and check mass conservation, the cropland-area formulation, the manure
ceiling, the fertilizer weighting and the Monte Carlo sampling.

## License

Code: MIT License (see `LICENSE`). The dataset on Zenodo is released under CC BY 4.0. Input datasets
remain under their own licenses.

## Citation

Please cite the data descriptor and the Zenodo dataset (see `CITATION.cff`).

## Contact

Kimberly J. Van Meter, Department of Geography, The Pennsylvania State University.
