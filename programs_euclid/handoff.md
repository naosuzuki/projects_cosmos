# programs_euclid — handoff for stellar catalog work

## State as of v30 SN-finder commit (2026-05-26)

Two existing scripts:

- `02_readcatalog_euclid.py` — reads Euclid MER + IDR tile metadata.
  Tile inventory and footprint info.
- `03_make_sn_cutouts_euclid.py` — produced per-SN Euclid VIS+NISP
  cutouts for the SN-finder webpage.

Nothing here yet for stellar catalog work.

----

## What we learned in programs_webpage that applies directly

1. **Euclid PSF is BROAD (0.19″ VIS, 0.5-0.6″ NISP).**  The broad NISP
   PSF means:
   - More sources cleared as PSF-like that are actually marginal
     galaxies → G2 upper limit critical.
   - Aperture photometry needs a much larger ring (out to ~2″ for
     NISP-H) — make sure local environment doesn't pollute it.
   - Cross-band morphology consistency is the main defence:
     a true point source has consistent sharp/rnd1/rnd2 across all
     four bands (VIS + 3 NISP).
2. **VIS is the anchor.**  Wherever VIS shows a PSF-like source,
   trust it more than NISP-only detections.
3. **63924 lesson**: NISP-Y had clean DAO morphology (sharp=0.45,
   rnd1=+0.12, rnd2=-0.19) while NISP-H rnd2=-0.83 was a hard
   reject — the SN+host were merging in NISP-H's broader PSF.
   Same effect will apply to faint stars near galaxies in NISP-H.
4. **Background is already subtracted.**  `BGSUB-MOSAIC` in the
   filename means the global background is removed.  `bg_med ≈ 0` after
   `sigma_clipped_stats`; use `bg_std` only for DAO threshold.
5. **Pixel scale 100 mas** (vs 30 mas for HST/JWST) — Euclid tiles are
   ~5× smaller in pixel count for the same sky area.  DAO scans are
   correspondingly faster (~20 ms per band per tile, warm cache).
6. **Per-band aperture sizes** (1×FWHM):
   - VIS: 0.194″ → ~2 px → tiny.  Use `aper_r = max(0.20″, FWHM)`
     to keep at least 2 px diameter.
   - NISP-Y/J/H: 0.5-0.6″ → 5-6 px.  Normal.

----

## Proposed next-step plan for the Euclid stellar catalog

### 10_run_dao_per_tile_euclid.py
- For each Euclid DR1 COSMOS tile (60 tiles VIS-only that overlap
  COSMOS):
  - For each band VIS, NIR-Y, NIR-J, NIR-H:
    - Open `EUC_MER_BGSUB-MOSAIC-<band>-TILE<tile>-*.fits` (cached).
    - Compute `bg_std` via `sigma_clipped_stats` on a sub-sampled grid
      (file is pre-background-subtracted, so `bg_med ≈ 0`).
    - DAOStarFinder at threshold = 5σ_std across the whole tile.
    - Apply morphology gates G2-G4.
    - Aperture phot: `aper_r = 1×FWHM` (special case VIS: floor at
      0.20″), ring `2-3.5×FWHM`.
  - Cross-band match within tile (radius 0.20″ — looser than JWST
    because of broader Euclid PSFs).
  - Confirmation rule:
    - VIS at ≥5σ + ≥1 NISP at ≥3σ (preferred case — most stars), OR
    - ≥2 NISP bands at ≥5σ if VIS is non-detection (rare, treat
      as candidate; flag for manual check).
  - Saturation: VIS saturates ~mag 16; NISP saturates ~mag 13-14.
    Flag stars with peak above mission-supplied saturation level
    (check `SATURATE` keyword in header).
  - Write `csvfiles_stellar_euclid/tile_<tile>_stars.csv`.

### 11_merge_tiles_euclid.py
- Concatenate per-tile CSVs.
- Deduplicate in overlaps (match within 0.20″ → keep highest
  combined VIS+NISP SNR).
- Output: `csvfiles_stellar_euclid/euclid_stars_v01.csv`.

### 12_validate_against_mer_classification.py
- Cross-match the catalog against the Euclid MER catalogue's
  `phz_classification=1` flag (Euclid's own star classification).
- Recovery completeness vs mag.  Per-band purity.
- Quantify the NISP-only star fraction (expect to be small and
  contaminated).

### 13_validate_against_gaia.py
- Same as HST/JWST: Gaia DR3 cross-match for astrometric truth and
  star truth label.

### 14_export_for_cross_mission.py
- Slim catalog matching JWST/HST format.
- Columns: `id, ra_euclid, dec_euclid, epoch=2024, mag_VIS, mag_Y,
  mag_J, mag_H, snr_*, sharp_*, rnd1_*, rnd2_*, sat_flag,
  n_bands_detect`.

----

## Cross-mission notes (relevant to Euclid)

- Euclid VIS astrometry is tied to Gaia DR3 by the MER pipeline.
  Expected accuracy ~10-30 mas (slightly worse than JWST).
- Cross-match to JWST (both 2024 epoch): tolerance ~0.15″.
- Cross-match to HST (2003 epoch): tolerance ~0.3-0.5″ for PM.
- Euclid's 100 mas pixel scale means JWST + HST WCS resampling is
  fine (no aliasing issues going from 30 mas → 100 mas).

----

## Open questions / decisions for the user

- **VIS vs NISP epoch**: the SN-finder rule was "skip VIS because
  different epoch from NISP".  For DR1 stellar catalog work, this
  needs reconfirming — are MER VIS+NISP tile mosaics contemporaneous
  within ≤1 day, or could there be a several-month offset?  Stellar
  catalog can tolerate days of offset (PMs are sub-mas for that
  baseline) but not years.
- **Saturation handling**: same question as HST — header field vs
  empirical cut.  Euclid MER should have a definitive value.
- **NISP-only candidates**: keep them with a flag, or reject outright?
  My recommendation: keep with `confirmed=False` flag for now;
  filter at the cross-mission merge step.

----

## Files NOT to touch (under master programs_webpage control)

- `recut_3_hst.py`, `04_polish_labels.py`, `make_v##_*.py`,
  `csvfiles_sn/*.csv` — SN-finder canonical assets.  Stellar work
  imports from `recut_3_hst.py` but doesn't modify it.
