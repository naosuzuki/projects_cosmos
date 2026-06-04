# programs_hst — HST stellar catalog (point sources)

**Master criteria & conventions:**
`/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
Read that first — this file is a directory-scoped extension that
carries forward only what's relevant for HST point-source / stellar
catalog work.

----

## Scope

Detect stars (point sources, PSF-like compact morphology) in the HST
COSMOS_v2.0 30mas ACS F814W mosaics and produce a per-tile point-source
catalog that will later be merged with JWST and Euclid catalogs into a
unified cross-mission stellar catalog.

HST contributes **one band** (F814W) — no multi-band band-picker logic
needed (unlike JWST/Euclid).

----

## Inviolable principles (carry-over from master §1)

1. **PSF / point-source morphology is the PRIMARY discriminator.** Never
   pick sources by S/N alone. The master §2.5 bug-watch applies in full
   to stellar catalog work too: stars and galaxies separate primarily
   on `sharp`/`rnd1`/`rnd2`, not on brightness.
2. **No overwrites of versioned directories.** v01, v02, … never edit
   in place.
3. **Sign conventions**: N=up, E=LEFT in image; +RA = east =
   `+offset_arcsec / 3600 / cos(dec_rad)`.
4. **Commit & push when a version is approved.** Canonical artefacts
   (per-tile point-source CSVs, unified HST stellar catalog) go in
   git; per-version diagnostic PNGs may stay untracked.
5. **Knowledge transfer**: if the stellar work expands into a sibling
   directory (e.g. `programs_stellar/`), bring the criteria with it
   via a scoped CLAUDE.md there too.

----

## Morphology criteria for stars (same as master §2, scoped to F814W)

For a single-band catalog, the per-band eligibility gates collapse to:

| gate | rule | rejects                                      |
|------|------|----------------------------------------------|
| G1   | `sep < 0.20″` between DAO peak and catalog position when matching | mis-matches |
| G2   | `0.40 ≤ sharp ≤ 0.85`  | galaxies (>0.85), CRs/saturation (<0.40)  |
| G3   | `|rnd1| ≤ 0.50`        | asymmetric blobs                           |
| G4   | `|rnd2| ≤ 0.50`        | diagonal asymmetry                          |

**HST-specific tuning needs validation** against bright unsaturated stars
(mag 19-21 in F814W, Euclid MER `phz_classification=1`).  Master §2.6
star-control sample applies here directly.

Significance: PSF-matched aperture S/N at the catalog position,

```
aper_r   = 1.0 × FWHM    (= 0.134")   ring 2.0–3.5×FWHM
```

A star is "detected" if it passes G2-G4 AND `snr_aper ≥ 5σ` in F814W.

----

## HST-specific facts

- **Pixel scale**: 30 mas (drizzled product `acs_I_030mas_<tile>_sci.fits`).
- **PSF FWHM** (empirical, 2026-05-26): 0.134″ ± 0.018″.
- **Filter**: F814W only.  No multi-band confirmation possible from HST
  alone — confirmation must come from JWST or Euclid.
- **Saturation**: F814W saturates around mag 18 on the COSMOS_v2.0
  mosaic depending on exposure depth.  Flag any source with peak
  brighter than the saturation level (read header `SATURATE` or
  inspect bright-end of magnitude histogram).
- **Epoch**: ACS COSMOS observations 2003-2007.  Proper-motion offsets
  vs JWST (2024) and Euclid (2024-2025) of order **0.2-0.5″** for
  high-PM stars (≥ 10 mas/yr × ~20 yr).  Cross-mission matching needs
  this tolerance.
- **Tile naming**: 3-digit tile numbers, e.g. `052`, `040`, `076`, `089`.
  ACS COSMOS footprint = 173 tiles.
- **FITS path**:
  `/Volumes/exdisk1/data/HST/COSMOS_v2.0/acs_I_030mas_<tile>_sci.fits`
  (some still gz; sci+wht present).
- **WCS**: J2000 ICRS, drizzled to common grid.

----

## Reusable building blocks from programs_webpage/

These are imported / referenced, not copied:

- `recut_3_hst.py` — `_FITS_CACHE`, `_WCS_CACHE`, `_open_cached()`,
  `_wcs_cached()`, `aper_photometry()`, `cut()`, `resolve_jwst_path()`.
  Reuse via `sys.path.insert(...)` + `import recut_3_hst as R`.
- `make_v29.py` (`scan_dao()`, `measure_band()`) — the multi-band DAO
  scan + PSF-matched aperture pattern, scoped to single band for HST.

DON'T rewrite the FITS-cache or slice-from-memmap-before-astype tricks
from scratch — import from `recut_3_hst.py`.

----

## Existing files in this directory

| file                          | purpose                                            |
|-------------------------------|----------------------------------------------------|
| `02_readcatalog_hst.py`       | reads the COSMOS HST catalog (per-tile metadata)    |
| `03_make_sn_cutouts_hst.py`   | generates per-SN cutout PNGs (used by SN webpage)  |

Stellar-catalog scripts should be numbered **`10_`–`19_`** to leave
room for catalog/cutout work (00-09) and not collide.  See `handoff.md`
for the concrete next-step plan.

----

## Open items

- Validate G2-G4 thresholds against HST star control sample (master §2.6).
- Define HST saturation flag (header field vs empirical bright-end cut).
- Pick a per-tile vs whole-mosaic architecture for the catalog
  (per-tile is friendlier to parallelisation and matches existing
  cutout architecture).
