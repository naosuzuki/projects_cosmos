# programs_jwst — handoff for stellar catalog work

## State as of v30 SN-finder commit (2026-05-26)

Two existing scripts:

- `02_readcatalog_jwst.py` — reads COSMOS-Web tile metadata.  Tile
  inventory + footprint info for the catalog tile loop.
- `03_make_sn_cutouts_jwst.py` — produced per-SN JWST cutouts for the
  SN-finder webpage.  Template for diagnostic PNG generation.

Nothing here yet for stellar catalog work.

----

## What we learned in programs_webpage that applies directly

1. **DAOStarFinder + per-band morphology gates** correctly identify
   point sources across the four JWST bands.  The 11 SN test set
   established that `sharp ∈ [0.4, 0.85]`, `|rnd1| ≤ 0.5`, `|rnd2| ≤ 0.5`
   separates real point sources from galaxies and blends.
2. **F115W ≠ always best band.**  Across the 11 SN test set the
   morphology-criteria best band landed on F115W (6), F150W (3), and
   F277W (rejected due to host hijack in 1 case).  For stars the
   distribution will be different — saturation flips bright stars from
   F115W to F277W/F444W — but the multi-band evaluate-then-pick logic
   is the same.
3. **PSF-matched aperture is critical.**  The fixed 0.20″ aperture
   used by recut_3_hst.aper_photometry's default is 3.5× too big for
   F115W (FWHM=0.057″) — diluting SNR by ~3-10×.  Always pass
   per-band PSF FWHM.  Master §2 has the table.
4. **`peak/bg_std` is a HOSTILE picker.**  Don't use it as the best-band
   selection metric — the closest DAO candidate to a SN/star coord can
   be a neighbour (host) in redder bands.  Use `snr_aper`
   (PSF-matched aperture S/N at the catalog position) instead.  Master
   §2.5 documents the bug.
5. **Single-band DAO scan covers the whole tile in ~150 ms** on warm
   cache for 30 mas drizzle products of ~5000×5000 px.  Four bands
   per tile × ~18 tiles in COSMOS-Web = 72 scans = ~11 s wall.  Very
   manageable for a single-process run; multiprocessing is optional.
6. **FITS handle caching** (`_FITS_CACHE`) keeps repeated band/tile
   accesses essentially free.  Open all FITS at the start of each
   tile's work; close at end of tile if memory tight.

----

## Proposed next-step plan for the JWST stellar catalog

### 10_run_dao_per_tile_jwst.py
- For each COSMOS-Web tile (A1-A12, B1-B6):
  - For each band F115W, F150W, F277W, F444W:
    - Open `<v1.0 i2d>` (SCI + ERR HDUs) with FITS cache.
    - Background estimate via `sigma_clipped_stats` on a sub-sampled
      grid (full tile 5000² → use 500² stride).
    - Run DAOStarFinder at threshold = 5σ on the whole tile.
    - Apply morphology gates G2-G4.  Don't apply G1 yet — that's the
      per-source position gate; for catalog mode each survivor IS the
      position.  G1 becomes relevant only at cross-band matching.
    - Compute PSF-matched aperture phot + AB mag using header
      `PIXAR_SR` + 1 MJy/sr ZP conversion.
  - Cross-band match within the tile (radius 0.10″) to build per-source
    rows with `mag_F115W, snr_F115W, sharp_F115W, …` for each band.
  - Apply multi-band confirmation: pass requires
    `≥1 band with snr_aper ≥ 5σ AND ≥2 bands with snr_aper ≥ 3σ`,
    each passing G2-G4.
  - Saturation flag: peak == NaN OR peak > BAND-specific limit.
  - Write `csvfiles_stellar_jwst/tile_<tile>_stars.csv`.

### 11_merge_tiles_jwst.py
- Concatenate per-tile CSVs.
- Deduplicate sources in tile overlaps (match within 0.15″ → keep
  highest combined SNR row).
- Output: `csvfiles_stellar_jwst/jwst_stars_v01.csv`.

### 12_validate_against_gaia.py
- Cross-match against Gaia DR3 stars in COSMOS field.
- Recovery completeness vs mag (per band).
- Astrometric offset between JWST-derived RA/Dec and Gaia DR3 at JWST
  epoch (proper-motion-propagated from Gaia 2016).  Expected JWST
  internal accuracy ~10-30 mas after Gaia tying.
- If recovery < 80% for mag 18-22 stars, loosen G2/G3/G4 and retest.

### 13_export_for_cross_mission.py
- Slim catalog for the cross-mission match step.
- Columns: `id, ra_jwst, dec_jwst, epoch=2024, mag_F115W, mag_F150W,
  mag_F277W, mag_F444W, snr_*, sharp_*, rnd1_*, rnd2_*, sat_flag,
  n_bands_detect`.

----

## Cross-mission notes (relevant to JWST as anchor)

- JWST has the best astrometric reference (tied to Gaia by COSMOS-Web
  pipeline) — use JWST positions as the cross-mission **reference
  frame**.  HST (epoch 2003) and Euclid (2024) positions should be
  matched **into** the JWST coordinate system.
- Proper-motion tolerance for HST → JWST cross-match: ~0.2-0.5″ for
  typical disk stars over 20 yr baseline.  Looser for high-PM stars.
- Cross-match radius for JWST → Euclid: smaller (~0.15″) since both
  are 2024 epoch.

----

## Open questions / decisions for the user

- **Tile-loop parallelism**: 18 tiles × 4 bands = 72 sub-tasks.
  Multiprocessing.Pool with 6 workers gets a ~6× speedup.  Worth doing
  or single-process is fine?
- **i2d vs sci.fits**: prefer v1.0 i2d (has ERR HDU) but some tiles
  may only have v0.8 sci.  Fall back to v0.8 with sigma-clipped noise
  estimate per band per tile?
- **Saturation threshold**: peak NaN is the most reliable JWST
  saturation signature, but the pipeline may sometimes leave a finite
  value.  Calibrate during the validation step.

----

## Files NOT to touch (under master programs_webpage control)

- `recut_3_hst.py`, `04_polish_labels.py`, `make_v##_*.py`,
  `csvfiles_sn/*.csv` — SN-finder canonical assets.  Stellar work
  imports from `recut_3_hst.py` but doesn't modify it.
