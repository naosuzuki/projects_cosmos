# programs_euclid — Euclid stellar catalog (VIS + NISP point sources)

**Master criteria & conventions:**
`/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
Read that first — this file is a directory-scoped extension that
carries forward only what's relevant for Euclid point-source / stellar
catalog work.

----

## Scope

Detect stars (point sources) in Euclid DR1 COSMOS mosaics:

- **VIS** — single broad optical band (~550-900 nm), 100 mas pixel,
  FWHM 0.194″.  The sharpest Euclid product — primary for stellar
  detection.
- **NISP-Y, NISP-J, NISP-H** — three near-IR bands, 100 mas pixel,
  FWHM 0.52-0.57″.  Much broader PSF; many fewer resolved stars,
  many more star-galaxy blends.

Produce a per-tile Euclid stellar catalog (VIS + NISP combined) that
merges with HST and JWST catalogs into the unified cross-mission
stellar catalog.

**VIS is the anchor band** within Euclid — NISP detections should be
confirmed by VIS where possible.  NISP-only stars are rare and
suspect.

----

## Inviolable principles (carry-over from master §1)

1. **PSF / point-source morphology is the PRIMARY discriminator.**  In
   NISP especially, the broad PSF means many galaxies look star-like
   in NISP alone — morphology gates must be applied per band and
   cross-checked against VIS where available.
2. **Per-band morphology evaluation.**  Don't roll up NISP-Y/J/H into
   one "NISP" measurement.  Each NISP band has its own PSF, its own
   noise floor, its own host-contamination behaviour.
3. **VIS ≠ NISP epoch in general** — but for DR1 COSMOS the
   VIS+NISP observations are contemporaneous within each visit.  This
   is OPPOSITE to the SN-finder rule (where we skipped VIS because it
   was a different epoch than NISP); for **stellar catalog** work the
   epochs are aligned, so VIS+NISP all four bands are usable together.
   Re-confirm during data exploration.
4. **No overwrites of versioned directories.**
5. **Sign conventions**: N=up, E=LEFT.
6. **Commit & push when a version is approved.**

----

## Morphology criteria for stars (master §2 scoped to Euclid)

Per-band eligibility (must pass ALL four):

| gate | rule                                | rejects                       |
|------|-------------------------------------|-------------------------------|
| G1   | `sep ≤ max(0.10″, 1.5×FWHM_band)`   | mis-matches & blends         |
| G2   | `0.40 ≤ sharp ≤ 0.85`               | galaxies, CRs, saturation    |
| G3   | `|rnd1| ≤ 0.50`                     | asymmetric blobs             |
| G4   | `|rnd2| ≤ 0.50`                     | diagonal asymmetry            |

Detection: pass G1-G4 AND `snr_aper ≥ 5σ` in VIS OR in ≥ 1 NISP band
with ≥ 2 bands at ≥ 3σ.  PSF-matched aperture:
`aper_r = 1×FWHM_band`, ring `2-3.5×FWHM_band`.

**G2 in NISP is fragile.**  At FWHM 0.5-0.6″ and 100 mas pixels,
star-galaxy separation is mediocre.  The validation step (master §2.6)
matters more here than for HST/JWST.

----

## Euclid-specific facts

- **VIS**:
  - Pixel scale: 100 mas
  - PSF FWHM: 0.194″ ± 0.006″
  - Single broad band ~ 550-900 nm; deepest single-exposure Euclid product
  - Path: `/Volumes/exdisk1/data/Euclid/COSMOS_DR1/EUC_MER_BGSUB-MOSAIC-VIS_TILE<tile>-*.fits`
- **NISP-Y**:
  - Pixel scale: 100 mas
  - PSF FWHM: 0.524″ ± 0.096″
  - Path: `EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE<tile>-*.fits`
- **NISP-J**:
  - FWHM: 0.537″
  - Path: `EUC_MER_BGSUB-MOSAIC-NIR-J_TILE<tile>-*.fits`
- **NISP-H**:
  - FWHM: 0.567″
  - Path: `EUC_MER_BGSUB-MOSAIC-NIR-H_TILE<tile>-*.fits`
- **Tile naming**: 9-digit numbers (e.g. `101542818`, `101545698`).
  COSMOS DR1 has 60 tiles VIS-only that fit COSMOS footprint.
- **Background**: the files are already background-subtracted (the
  `BGSUB-MOSAIC` in the filename).  Skip global bg subtraction; only
  estimate `bg_std` locally for DAO threshold.
- **Epoch**: 2024-2025 (DR1 = Reference Release 1).

----

## Cross-band logic within Euclid

A star is detected if **EITHER**:

  - VIS passes G1-G4 with `snr_aper ≥ 5σ` (anchor case — most stars),
  - OR ≥ 2 NISP bands pass G1-G4 with `snr_aper ≥ 5σ` and at least one
    passes ≥ 5σ — NISP-only stars are accepted only with this stricter
    rule (since galaxies are easy to confuse in NISP).

The 63924 SN case taught us that NISP-Y often has the cleanest
star/SN morphology of the three NISP bands — it's the bluest, so
galaxies are LESS reddened relative to the star.  For the catalog
this means NISP-Y is the most useful NISP band for star/galaxy
separation; NISP-H is the worst.

----

## Reusable building blocks from programs_webpage/

- `recut_3_hst.py` — caches, helpers; `_eu_band_path_fn(band)`,
  `resolve_euclid_path(tile, band)`.
- `make_v29.py` — `scan_dao()`, `measure_band()` patterns.

----

## Existing files in this directory

| file                              | purpose                                          |
|-----------------------------------|--------------------------------------------------|
| `02_readcatalog_euclid.py`        | reads Euclid MER tile metadata + IDR catalog    |
| `03_make_sn_cutouts_euclid.py`    | per-SN Euclid VIS+NISP cutouts (SN webpage)     |

Stellar-catalog scripts should be numbered **`10_`–`19_`**.  See
`handoff.md`.

----

## Open items

- Validate G2-G4 thresholds against Euclid star control (Euclid MER
  catalogue's own `phz_classification=1` is the natural truth label).
- Quantify NISP star/galaxy separation efficiency — expect it to be
  the weakest of the three missions.
- Confirm that BGSUB-MOSAIC means the global background really is
  removed (i.e., don't double-subtract).
