# Coefficient and ancillary tables

All tables are read by the model through `config/config.yaml`. Lines beginning with `#` are comments.

| File | Contents |
|---|---|
| `cropland_fraction.csv` | Fraction of national fertilizer applied to cropland, cf(i), for the 21 countries listed in FAO Cropland Nutrient Budget Table 1; cf = 1 elsewhere |
| `fertilizer_volatilization_global.csv` | NH3 volatilization fraction and N content by fertilizer product |
| `crop_uptake.csv` | Crop N removal coefficients, c_N(j), from Ludemann et al. |
| `bnf_params.csv` | Legume fixation parameters (%Ndfa from Peoples et al. 2021; whole-plant factors) |
| `pasture_uptake.csv` | Pasture N-uptake rates for managed pasture and rangeland (forage supply ceiling) |
| `animal_product_N.csv` | N content of grazing-derived animal products |
| `livestock_excretion.csv`, `livestock_tam.csv`, `ipcc_region_map.csv` | IPCC Tier-1 excretion inputs (used only with `manure.method: bottom_up`) |
| `ipcc_nex_region_crosswalk.csv` | Country to IPCC excretion region |
| `glw_excretion.csv`, `species_map.csv` | Livestock species weights for the manure spatial pattern |
| `manure_volatilization_global.csv` | Manure NH3 volatilization fractions |
| `landcover_classes.csv` | MODIS land-cover classes used for grazing land |
| `dep_wl_country_byyear.csv` | National cropland deposition, Vishwakarma et al. (2023) WL product |
| `dep_total_country_byyear.csv`, `dep_noncrop_country_byyear.csv`, `dep_ox_country.csv` | ISIMIP3a deposition aggregated to countries |
| `population_country.csv` | National population (FAOSTAT) |
| `food_consumer_loss.csv` | Retail and household food-loss fractions, w_c (Morée et al. 2013) |
| `gdp_percapita.csv` | GDP per capita |
| `sewer_connection.csv`, `sewer_connection_anchors.csv`, `region_factors.csv` | Sewer-connection fraction c(i,t), its observed anchors and regional factors |
| `wastewater_treatment.csv` | Treated fraction and treatment N removal |
| `human_recycle.csv`, `human_recycle_country.csv` | Night-soil recycling classes and country trajectories |
| `biosolids_routes.csv`, `biosolids_routes_byyear.csv`, `biosolids_beta.csv` | Sewage-sludge disposal routes |
| `cropgrids_country_crosswalk.csv`, `cropgrids_item_crosswalk.csv`, `coello_cropgrids_crosswalk.csv` | Crosswalks between FAOSTAT, CROPGRIDS and Coello crop and country codes |
| `activity_cv_by_flag.csv`, `activity_cv_summary.csv` | Activity-data CVs by FAOSTAT flag (Table 9) |
| `uncertainty_spec.csv`, `uncertainty_region_cv.csv` | Coefficient CVs for the Monte Carlo analysis (Table 8), with sources in the `notes` column |
