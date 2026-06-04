# programs_cnnsn — CNN-based SN classifier (extension of the master pipeline)

**Master criteria & conventions:**
`/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
Read that first. It has the working SN-extraction baseline (v32) — the
G1-G4 morphology gates, the multi-band confirmation rules, the bug
patterns, the inviolable principles. This file is a directory-scoped
extension; **it does not duplicate the master, only points back and
records the cnnsn-specific scope**.

----

## Scope of this directory

The master baseline (`make_v29.py` + `recut_3_hst.py`) recovers all 17
known SN with zero misidentifications using hand-engineered §2 morphology
gates. Per master §6, the **long-term goal** is a CNN-based classifier
that replaces or augments those gates while reusing the same
measurement infrastructure.

This directory is where that classifier lives. **It is NOT a from-scratch
rewrite of SN discovery.** The discovery/measurement layer is mature; we
plug into it.

### What stays in the master pipeline (do NOT reimplement here)

- FITS path resolution: `R._hst_path_fn`, `R.resolve_jwst_path`,
  `R.resolve_euclid_path`. These already handle the A10 F115W fallback,
  Euclid `superseded_may2025/` exclusion, and the v0.8/v1.0/.gz priority.
- Memmap slicing trick: slice `sci_hdu.data[y0:y1, x0:x1]` FIRST, then
  `.astype(np.float64)`. Otherwise astropy materialises a 30 GB tile.
- WCS/FITS caches: `R._FITS_CACHE`, `R._WCS_CACHE`, `R._open_cached`,
  `R._wcs_cached`.
- PSF-matched aperture photometry: `R.aper_photometry` (or the inline
  versions in `make_v29.py` / `find_v31_positions.py`).
- DAOStarFinder scan + sigma-clipped background — exactly as in
  `make_v29.measure_band`.

### What this directory adds

A learned classifier that takes the per-band measurements emitted by
`make_v29.measure_band` (plus optionally the FITS cutout pixels) and
decides "real SN" / "not SN", replacing or augmenting:

- §2.1 per-band eligibility (G1-G4)
- §2.3 multi-band confirmation rule (HST/JWST/Euclid thresholds)
- §2.4 best-band selection

The CNN doesn't discover new SN positions on its own; it takes a
catalog of candidate positions (initially the 17 known + the control
samples in §2.6 of the master) and classifies them.

----

## Data sources (authoritative; mirrored from the master)

| Survey | Base directory | Notes |
|---|---|---|
| HST ACS | `/Volumes/exdisk1/data/HST/COSMOS_v2.0/` | `acs_I_030mas_<NNN>_sci.fits` + `_wht.fits`, 81 fields. |
| JWST COSMOS-Web | `/Volumes/exdisk1/data/JWST/COSMOS_v0.8/` | `mosaic_nircam_<filter>_*_v1.0_i2d.fits`. A10 F115W: use 3 files in `scidir/` (sci v0_8, wht v1.0, err v1.0). `R.resolve_jwst_path` already handles. |
| Euclid DR1 | `/Volumes/exdisk1/data/Euclid/COSMOS_DR1/` | 60 tiles × 4 bands. May-2025 superseded versions in `superseded_may2025/` — do not use. |
| Ground truth | `/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn17_v32.csv` + `lookup_sn17_v32.csv` | 17 SNe (3 HST, 13 JWST, 1 EUCLID). v32 is current; bump `<N>` for SN additions, `<V>` for revisions. |
| Tile footprint index | `/Users/suzuki/github/projects_cosmos/csvfiles/tile_lookup.fits` | Built by `programs_webpage/build_tile_lookup.py`. Maps RA/Dec → which tile, for any new candidate position. |

## Inherited operational rules (from master)

- **Version every iteration.** `v01/`, `v02/`, … never overwrite.
- **No commits without user OK.** Sign-off triggers commit + push.
- **PSF morphology > S/N** as the discriminating signal — the CNN
  inputs should preserve PSF shape information.
- **`Cutout2D(mode="partial")` can silently suppress DAO peaks.** Use
  the raw memmap slice when DAO output matters; Cutout2D is only safe
  for rendering / human-eyes inputs.

----

## Training data plan (what we need before training)

1. **Positives:** 17 known SNe from `tbl_sn17_v32.csv`. Use the
   per-source best-band cutout (and/or all bands of the source's
   telescope) at the catalog SN RA/Dec. Augment with rotation/flip;
   maybe small position jitter (≤0.05″ per master §2.6 #4).
2. **Negatives — three control samples per master §2.6:**
   - ~50 unsaturated stars (mag 19–21) — should LOOK like SN to the
     CNN (PSF morphology), but live on no host. Tests over-broadening
     of the positive class.
   - ~50 host galaxies away from any SN — should fail G2 (sharp>0.85).
     Tests rejection of extended sources.
   - ~50 random catalog-blank positions — should fail G1. Tests the
     noise floor.
3. **Validation: hold-out cross-validation** — leave-one-out across
   the 17 known SNe so we still get a usable held-out set. Plus the
   three control samples as separate test sets.

## Architecture sketch (subject to revision)

```
programs_cnnsn/
├── CLAUDE.md                     this file
├── data/                         (to populate) cached cutouts, labels
├── src/
│   ├── extract_cutouts.py        wraps R.measure_band → multi-band FITS
│   │                              cutout tensors at given RA/Dec
│   ├── build_controls.py         generate the 3 control samples (§2.6)
│   ├── model.py                  CNN architecture
│   ├── train.py                  training loop, LOO-CV
│   └── eval.py                   evaluate on controls + held-out
└── outputs/                      gitignored; cutout caches, weights
```

## Open items (must resolve before training)

1. **Cutout schema decision** — fixed pixel size per band (e.g. 64×64
   regardless of pixel scale → physical box varies), or fixed angular
   box (e.g. 3″ × 3″ → pixel size varies per band)? Affects whether
   the CNN sees consistent PSF size or consistent physical context.
2. **Per-band channel stacking** — all bands of all telescopes as
   channels (with NaN for missing), or per-telescope models? The 17
   training SNe are unbalanced (3 HST / 13 JWST / 1 EUCLID), favouring
   a per-telescope start.
3. **Tabular features alongside images** — feed the §2 measurements
   (`snr_aper`, `sharp`, `rnd1`, `rnd2`, `sep_sn`) as auxiliary inputs?
   These already encode most of the discrimination signal and would
   shrink the training-data requirement.
4. **Control-sample generation** must be done first, before any model
   training. Spec'd by master §2.6.
5. **Build `tile_lookup.fits` if not current** — `build_tile_lookup.py`
   in `programs_webpage/` is the canonical builder; this CLAUDE.md
   doesn't replicate it.

## Versioning

Per master §1#4: every CNN iteration that produces classifier outputs
goes in `outputs/cnn_v<NN>/` and is referenced from a top-level
`outputs/index.html`-style ledger. Don't overwrite v01 with v02.

## Why this scoping (per master §1#8)

The master CLAUDE.md is the single source of truth for SN-identification
algorithmic decisions (PSF-first, snr_aper picker, G1-G4 gates, the bug
patterns). Duplicating those rules here invites drift. This file only
records: (a) what's specific to the CNN-extension scope, (b) what data
this directory consumes, (c) the project-specific open items. For
algorithmic decisions, defer upstream.
