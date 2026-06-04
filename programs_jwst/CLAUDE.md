# programs_jwst — JWST stellar catalog (point sources)

**Master criteria & conventions:**
`/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
Read that first — this file is a directory-scoped extension that
carries forward only what's relevant for JWST point-source / stellar
catalog work.

----

## Scope

Detect stars (point sources) in the JWST COSMOS-Web NIRCam 30mas
mosaics across **four bands** (F115W, F150W, F277W, F444W) and produce
a per-tile point-source catalog that will be merged with HST and Euclid
catalogs into a unified cross-mission stellar catalog.

JWST is the **sharpest PSF and deepest single-epoch coverage** of the
three missions — the morphology criteria from the master CLAUDE.md
were validated primarily on JWST sources, and JWST will likely be the
anchor for cross-mission matching (best astrometry + best
star/galaxy separation).

----

## Inviolable principles (carry-over from master §1)

1. **PSF / point-source morphology is the PRIMARY discriminator.** Never
   pick sources by S/N alone.  The bug we hit on 246188 (F277W
   peak/bg_std=1157 was the host, not the SN) applies equally to
   stellar catalog work: bright neighbours in red bands can hijack
   compactness measurements.
2. **Per-source band selection, not per-telescope.**  For each star,
   evaluate every band individually — apply morphology gates
   per band, declare detected only when the gates pass.  Don't assume
   F115W is always best; saturation flips it for bright stars.
3. **No overwrites of versioned directories.**
4. **Sign conventions**: N=up, E=LEFT; +RA = east =
   `+offset_arcsec / 3600 / cos(dec_rad)`.
5. **Commit & push when a version is approved.**

----

## Morphology criteria for stars (from master §2, scoped to JWST)

Per-band eligibility (must pass ALL four):

| gate | rule                              | rejects                          |
|------|-----------------------------------|----------------------------------|
| G1   | `sep ≤ max(0.10″, 1.5×FWHM_band)` between DAO peak and reference position | mis-matches & blends |
| G2   | `0.40 ≤ sharp ≤ 0.85`             | galaxies, CRs, saturation       |
| G3   | `|rnd1| ≤ 0.50`                   | asymmetric blobs                |
| G4   | `|rnd2| ≤ 0.50`                   | diagonal asymmetry              |

Detection: pass G1-G4 AND `snr_aper ≥ 5σ` in at least one band,
`≥ 3σ` in ≥ 2 bands (same 1×5σ + 2×3σ rule we used for SN
confirmation).  PSF-matched aperture: `aper_r = 1×FWHM_band`,
ring `2-3.5×FWHM_band`.

**Saturation warning**: JWST NIRCam F115W saturates around mag 17 in
COSMOS-Web v1.0 i2d products — much brighter than the SN regime
(mag 24-27).  The morphology of saturated stars is BAD (sharp falls
below 0.4, peak pixels NaN'd by the pipeline).  G2 should reject
saturated stars naturally — confirm this during validation.

----

## JWST-specific facts

- **Pixel scale**: 30 mas (`mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v1.0_i2d.fits`).
- **PSF FWHM** (empirical, 2026-05-26):
  - F115W = 0.057″
  - F150W = 0.057″
  - F277W = 0.130″
  - F444W = 0.160″
  - F115W and F150W are PSF-limited (Nyquist-sampled at 30 mas drizzle).
- **Tile naming**: `A1`–`A12`, `B1`–`B6` (visit-based).
- **HDU layout**: v1.0 i2d files have `SCI` and `ERR` HDUs (named).
  v0.8 sci.fits is HDU 0 only (no `ERR`).  Use the v1.0 i2d when
  available for proper per-pixel SNR; fall back to v0.8 with
  `sigma_clipped_stats` for noise.
  Path resolution: `recut_3_hst.resolve_jwst_path(tile, band)`.
- **N-up rotation**: JWST drizzled products use spacecraft roll angle,
  not N=up.  For VISUAL displays use `north_up_jwst=True` in
  `recut_3_hst.cut()`.  For CATALOG/PHOTOMETRY work the roll doesn't
  matter — DAO and aperture phot operate on pixels in the native
  frame; WCS resolves to RA/Dec correctly either way.
- **Epoch**: 2023-2024.  Use as cross-mission reference epoch (most
  recent of the three, best astrometry from Gaia-tied calibration).

----

## Multi-band logic for stellar catalog

A star is detected if **EITHER**:

  - it passes G1-G4 in ≥ 2 bands with `snr_aper ≥ 3σ` in each AND
    ≥ 1 of those bands has `snr_aper ≥ 5σ`, OR
  - it's saturated (peak NaN'd) in F115W but recoverable in F277W or
    F444W — handle as a separate "saturated bright star" case with a
    relaxed morphology gate.

Best-band selection (for the per-star metadata): highest `snr_aper`
among bands passing G1-G4.  Same picker as the SN work, same
anti-hijack defence (sep_sn gate).

----

## Reusable building blocks from programs_webpage/

- `recut_3_hst.py` — `_FITS_CACHE`, `_WCS_CACHE`, `_open_cached()`,
  `_wcs_cached()`, `aper_photometry()`, `cut()`, `resolve_jwst_path()`,
  the per-band lambda helpers (`_jwst_band_path_fn`).
- `make_v29.py` — `scan_dao()`, `measure_band()`, the per-band loop
  with morphology gates, eligibility logic.  Adapt the per-source loop
  to a per-tile loop for catalog mode.

Don't rewrite the FITS-cache / slice-then-astype trick from scratch.

----

## Existing files in this directory

| file                              | purpose                                          |
|-----------------------------------|--------------------------------------------------|
| `02_readcatalog_jwst.py`          | reads COSMOS-Web tile metadata catalog          |
| `03_make_sn_cutouts_jwst.py`      | per-SN JWST cutouts (used by SN webpage)        |

Stellar-catalog scripts should be numbered **`10_`–`19_`**.  See
`handoff.md` for the next-step plan.

----

## Open items

- Validate G2-G4 thresholds against JWST star control (~50 stars, mag
  18-22, `phz_classification=1` from Euclid MER catalogue) per band.
- Confirm saturation behaviour: is `peak = NaN` the signature, or does
  the pipeline propagate a finite (but garbage) value?  Inspect bright
  Gaia stars in F115W to find out.
- Decide whether F115W and F150W get treated as a "blue pair" for SED
  classification (they share PSF, ~similar depth) or kept independent.
