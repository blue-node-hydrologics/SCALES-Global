// ============================================================================
// Global Pasture Watch (GPW) 30-m grassland -> 0.05deg grassland FRACTION layer
// for SCALES-Global grassland-manure downscaling.
//
// Parente et al. 2024, Sci Data (doi:10.1038/s41597-024-04139-6); CC-BY-4.0.
// Asset: projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c  (band dominant_class:
//   0 = Other, 1 = Cultivated grassland, 2 = Natural/Semi-natural grassland), annual 2000-2022.
//
// WHAT THIS DOES
//   Builds binary 30-m masks for each grassland class, then averages them to the SCALES 0.05deg grid
//   -> the FRACTION of each cell that is cultivated (pasture) and natural/semi-natural (rangeland).
//   Fraction x cell area = land AREA per cell — the grazing-land analog of CropGRIDS `croparea`.
//   band pasture   -> narrow base: grassland FERTILIZER (managed pasture only)
//   band rangeland -> with pasture, the broad base (pasture+rangeland) for manure PRP, forage, BNF.
//
// OUTPUT  GPW_grassland_frac_0p05_<YEAR>.tif  (2-band Uint16 = fraction x 10000; band1 pasture,
//   band2 rangeland), grid crsTransform [0.05,0,-180,0,-0.05,90], 7200 x 3600 (lat 90..-90 north-up)
//   — identical to the model grid (edges at -180/90; centres -179.975/89.975).
//
// STEPS
//   1. Paste into the Earth Engine Code Editor (https://code.earthengine.google.com), Run.
//   2. In the Tasks tab, Run the export -> lands in your Google Drive folder "GPW_export".
//   3. Move the .tif to  data/raw/gpw/  in the repo. scripts/12c reads it from there.
// ============================================================================

var YEAR = 2022;                 // latest GPW year; SCALES freezes this pattern (Zhang gives annual signal)

var gpw = ee.ImageCollection('projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c')
            .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01').first();
var cls = gpw.select('dominant_class');            // 0 Other, 1 Cultivated, 2 Natural/Semi-natural

// TWO bands so the model can separate the land bases:
//   pasture   = cultivated grassland (class 1)  -> narrow base: grassland FERTILIZER
//   rangeland = natural/semi-natural (class 2)  -> broad base (pasture+rangeland): manure/forage/BNF
var two = cls.eq(1).addBands(cls.eq(2)).rename(['pasture', 'rangeland']);

// bestEffort:true + small maxPixels makes reduceResolution aggregate from a COARSE image PYRAMID of the
// 0/1 masks (whose pyramid already IS the fraction) instead of materialising all 30 m pixels globally
// -> avoids "Number of pixels requested from Image.eq exceeds 2^31". maxPixels 64 -> ~1 km pyramid.
var frac = two
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64, bestEffort: true})
  .reproject({crs: 'EPSG:4326', crsTransform: [0.05, 0, -180, 0, -0.05, 90]})
  .unmask(0);

Map.setCenter(-93.5, 42, 5);
Map.addLayer(frac.select('pasture'),   {min: 0, max: 1, palette: ['ffffff', 'ffcd73']}, 'cultivated (pasture) fraction');
Map.addLayer(frac.select('rangeland'), {min: 0, max: 1, palette: ['ffffff', 'ff9916']}, 'natural (rangeland) fraction');

Export.image.toDrive({
  image: frac.multiply(10000).round().toUint16(),          // store fraction x 10000 (0..10000)
  description: 'GPW_grassland_frac_0p05_' + YEAR,
  folder: 'GPW_export',
  fileNamePrefix: 'GPW_grassland_frac_0p05_' + YEAR,
  crs: 'EPSG:4326',
  crsTransform: [0.05, 0, -180, 0, -0.05, 90],
  dimensions: '7200x3600',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});
// After running: Tasks tab -> Run 'GPW_grassland_frac_0p05_2022' -> Drive/GPW_export/*.tif
//                -> move to data/raw/gpw/ , then: python scripts/12c_build_zhang_manure.py --mass-check
