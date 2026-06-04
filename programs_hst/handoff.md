# programs_hst — handoff for stellar catalog work

## State as of v30 SN-finder commit (2026-05-26)

Two existing scripts in this directory:

- `02_readcatalog_hst.py` — reads the HST COSMOS tile metadata catalog.
  Inspect first; the tile list and footprint info will drive the
  stellar-catalog tile-loop.
- `03_make_sn_cutouts_hst.py` — was used during SN finding to produce
  per-SN HST F814W cutouts.  Useful as a template for the
  per-star-cutout step if we want diagnostic PNGs later.

Nothing here yet for stellar catalog work.

----

## What we learned in programs_webpage that applies directly

1. **DAOStarFinder is the right tool for point-source detection.**
   `photutils.detection.DAOStarFinder(fwhm=fwhm_px, threshold=3*bg_std)`
   on a background-subtracted cutout.  Returns peak, sharp, rnd1, rnd2,
   flux, mag per candidate.
2. **PSF FWHM for HST F814W = 0.134″.**  Use as `fwhm_as`; convert to
   pixels via `fwhm_px = fwhm_as / pix_scale` (pix_scale ≈ 30 mas).
3. **PSF-matched aperture is SNR-optimal**: `aper_r = 1×FWHM`,
   ring `2-3.5×FWHM`.  Don't use the fixed 0.20″ aperture
   (that's 1.5× too large for F814W's 0.134″ FWHM — diluting SNR).
4. **Morphology gates (CLAUDE.md):** `0.40 ≤ sharp ≤ 0.85`,
   `|rnd1| ≤ 0.5`, `|rnd2| ≤ 0.5`.  These separate stars (pass) from
   galaxies (fail G2 upper) and from CRs/saturation (fail G2 lower).
5. **Don't materialise the full tile.**  Slice the memmap first, then
   `.astype(float64)` on the small slice.  This kept us from reading
   500 MB tiles when we only needed an 80 KB box.  Pattern in
   `recut_3_hst.py:cut()`.
6. **FITS caching across script lifetime.**  `_FITS_CACHE[path] =
   fits.open(path, memmap=True)` keeps the file open; subsequent
   accesses are free.  `_WCS_CACHE` likewise.

----

## Proposed next-step plan for the HST stellar catalog

### 10_run_dao_per_tile_hst.py
- For each of the 173 HST tiles in COSMOS_v2.0:
  - Open `acs_I_030mas_<tile>_sci.fits` (cached).
  - Estimate background via `sigma_clipped_stats` on the FULL tile
    (sub-sample for speed if needed — e.g., 256×256 grid).
  - Run DAOStarFinder at threshold = 5σ across the entire tile.
  - Apply morphology gates G2-G4.
  - For each survivor, compute PSF-matched aperture phot + AB mag.
  - Write per-tile CSV: `csvfiles_stellar_hst/tile_<NNN>_stars.csv`
    columns: `tile, x, y, ra, dec, f814w_mag, f814w_snr, peak, sharp,
              rnd1, rnd2, sat_flag`.
- Saturation flag: peak > some threshold (start with `peak > 60000` on
  the drizzled product; calibrate vs known-bright stars from Gaia).
- Parallelisation: 173 tiles is well-suited for `multiprocessing.Pool`
  IF each worker opens its own FITS handles (don't share `_FITS_CACHE`
  across processes — it's not pickle-safe).

### 11_merge_tiles_hst.py
- Concatenate per-tile CSVs into one HST stellar catalog.
- Deduplicate sources within overlap regions (catalog match within
  0.2″ → keep the higher-SNR row).
- Output: `csvfiles_stellar_hst/hst_stars_v01.csv` (or wherever the
  user wants the final catalogs — see also the cross-mission step
  in the parent project).

### 12_validate_against_gaia.py
- Cross-match the HST catalog against Gaia DR3 stars in the COSMOS
  field (Gaia has astrometry and proper motion for mag < 21).
- Quantify: positional offset histogram (HST 2003 vs Gaia 2016 epoch
  → ~10-100 mas systematic offset), recovery completeness vs mag.
- This is the empirical validation of the morphology criteria for
  HST.  If recovery is < 80% for mag 19-21 stars, the thresholds need
  loosening.

### 13_export_for_cross_mission.py
- Output a slim catalog with columns matching the JWST/Euclid format
  for the eventual cross-mission match.
- Required minimum columns: `id, ra, dec, mag_F814W, snr_F814W,
  sharp, rnd1, rnd2, sat_flag, epoch=2003-2007`.

----

## Open questions / decisions for the user

- **Output path**: should single-mission stellar CSVs live in
  `csvfiles_stellar/` (parallel to `csvfiles_sn/`) or in per-mission
  subdirs like `csvfiles_stellar_hst/`?
- **Versioning**: same `v##` convention as SN work — start at v01 for
  the first HST stellar catalog.
- **Saturation cutoff**: prefer header field if reliable, else empirical
  cut from bright-end Gaia comparison.
- **PM tolerance for cross-mission**: 0.2-0.5″ is my rough estimate;
  user may have a tighter or looser preference.

----

## Files NOT to touch (under master programs_webpage control)

- `recut_3_hst.py`, `04_polish_labels.py`, `make_v##_*.py`,
  `csvfiles_sn/*.csv` — these are the SN-finder canonical assets.
  Stellar work imports from `recut_3_hst.py` but doesn't modify it.
