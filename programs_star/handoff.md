# programs_star — handoff for unified stellar catalog work

## State as of v30 SN-finder commit (2026-05-26)

Empty directory.  This file documents the proposed architecture and
the dependencies on the three sibling directories.

----

## Upstream dependencies

This directory **cannot start** until at least one of the per-mission
catalogs from the sibling directories is available:

| dependency                                                    | needed by | status as of 2026-05-26 |
|---------------------------------------------------------------|-----------|--------------------------|
| `csvfiles_stellar_hst/hst_stars_v##.csv`                      | `12_crossmatch_hst_*`, `15_proper_motion` | not started |
| `csvfiles_stellar_jwst/jwst_stars_v##.csv`                    | `11_load_inputs` onwards                  | not started |
| `csvfiles_stellar_euclid/euclid_stars_v##.csv`                | `11_load_inputs` onwards                  | not started |
| Prior Euclid×ACS comparison output (`~/data/HSC_SNIa/euclid_acs_comparison/`) | `15_proper_motion` (validation) | done in earlier session |

The right order to bring directories online:
1. **programs_jwst/** first — JWST is the reference frame and the
   morphology criteria were validated primarily there.
2. **programs_hst/** second — needed for PM baseline (17-22 yr to JWST).
3. **programs_euclid/** third — broadest PSF + lowest astrometric
   precision; brings VIS + NISP photometry for SED work.
4. **programs_star/** last — depends on all three.

----

## What we learned in programs_webpage that applies directly

1. **Morphology gates (G1-G4) are validated only on the 11 SN test
   set** so far.  Master §2.6 validation campaign (star + galaxy +
   random-position control samples) is the prerequisite for trusting
   them at scale.  programs_star benefits from this because we get a
   large star sample to validate against here.
2. **Per-source band selection beats per-telescope.**  When picking a
   "reference band" per star, use the same rule we used for SN:
   highest `snr_aper` among bands passing morphology gates, NOT
   `peak/bg_std`.
3. **Host hijack analogue for stars**: bright neighbours in red bands
   can hijack centroid measurements.  At 100 mas pixel scale (Euclid
   NISP), even moderate density blends point sources.  Always check
   the per-band `sep_sn` for consistency across bands.
4. **PSF FWHM table** (master §5):
   F814W=0.134, F115W=0.057, F150W=0.057, F277W=0.130, F444W=0.160,
   VIS=0.194, NIR-Y=0.524, NIR-J=0.537, NIR-H=0.567 (all arcsec).
5. **Commit-and-push on milestone close** (master §1#7).  The unified
   catalog CSVs are canonical → always commit; PM quiver plots and
   colour-colour PNGs are diagnostic → user decides.

----

## Proposed next-step plan for the unified stellar catalog

### 10_load_inputs.py
- Read the three per-mission CSVs from `csvfiles_stellar_<mission>/`.
- Sanity-check column schemas match the documented baseline.
- Report counts: total / passing-morphology / saturated per mission
  per band.
- Output: in-memory tables for the downstream scripts (no CSV write).

### 11_crossmatch_jwst_euclid.py
- Match JWST ↔ Euclid VIS within **0.15″** (contemporaneous epochs).
- For each match: combined ID, ra/dec from JWST (reference frame),
  all JWST + Euclid magnitudes joined into a single row.
- Flag unmatched sources separately:
  - JWST-only (no Euclid counterpart) → typically faint or saturated
    in Euclid
  - Euclid-only → typically saturated or outside JWST footprint
- Output: `csvfiles_star/match_jwst_euclid_v01.csv`.

### 12_crossmatch_hst_to_jwsteuclid.py
- For each JWST/Euclid row from step 11, search HST catalog within
  **0.40″** (PM-tolerant for 17-22 yr baseline).
- Where multiple HST candidates are within radius, pick the closest
  AND record the second-closest separation as `ambig_arcsec` (master
  §2 ambiguity flag analogue).
- For known-high-PM Gaia stars, use Gaia's PM vector to predict the
  HST→JWST offset and tighten the match window.
- Output: `csvfiles_star/match_hst_jwsteuclid_v01.csv` — now 3-mission
  joined rows.

### 13_build_unified_catalog.py
- Combine the matched table from step 12 with the JWST-only / Euclid-only /
  HST-only orphans (with appropriate flags).
- Compute aggregate fields:
  - `n_missions_detect` (1, 2, or 3)
  - `n_bands_detect` (1-8)
  - `is_saturated_anywhere` (OR of per-mission sat_flag)
  - `best_morph_band` (band with highest `snr_aper` AND passing G1-G4
    — picker rule from master §2.4)
- Output: `csvfiles_star/star_unified_v01.csv`.

### 14_validate_gaia.py
- Cross-match the unified catalog against Gaia DR3 stars in the
  COSMOS field.
- Quantify:
  - positional offset histograms (per mission, per band)
  - recovery completeness vs G mag
  - false-positive rate (Gaia-stars-not-in-catalog and
    catalog-stars-not-in-Gaia)
- Output: `csvfiles_star/gaia_validation_v01.csv` + diagnostic PNGs.

### 15_proper_motion.py
- For stars detected at HST (epoch 2003-2007) AND JWST/Euclid (epoch
  2024), measure tangential PM.
- Compare with Gaia DR3 PMs where available — sanity check.
- Re-use methodology from the earlier Euclid×ACS work
  (`~/data/HSC_SNIa/euclid_acs_comparison/`).
- Output: `csvfiles_star/proper_motion_v01.csv`.

### 16_stellar_classification.py
- For non-saturated stars detected in ≥ 4 bands, build colour-colour
  table.
- Apply a decision-tree classifier (or simple colour cuts) to assign
  spectral class M/K/G/F/A/B.
- Output: `csvfiles_star/spectral_class_v01.csv`.

### 17_export_v01.py
- Produce the master deliverable: `csvfiles_star/star_master_v01.csv`
  with all columns from steps 13/14/15/16 joined by `star_id`.

----

## External truth datasets to use

- **Gaia DR3** — positions, PMs, parallaxes for stars to G ~ 21 across
  COSMOS.  The primary external truth.  Download CSV via
  `astroquery.gaia` for the COSMOS RA/Dec box.
- **Euclid MER `phz_classification`** — Euclid's own star/galaxy/QSO
  flag.  Use as a second-opinion validator for Euclid-detected stars.
- **COSMOS2025 photo-z catalog** — Hubeny et al. (?); has its own
  source list.  Possible cross-check on faint sources where Gaia is
  incomplete.

----

## Open questions / decisions for the user

- **Output directory** — `csvfiles_star/` to match the
  `csvfiles_sn/` pattern from master CLAUDE.md §3?  Alternative:
  `csvfiles_stellar/` (more descriptive but mismatches `_sn`/`_star`
  parallel).  Recommendation: **`csvfiles_star/`** for symmetry.
- **Versioning** — start at `v01` (since this is fresh work, not
  iterating on v30).
- **PM measurement method** — linear fit of (RA, Dec) vs epoch with
  free intercept and slope, weighted by per-mission positional
  uncertainty?  Or Bayesian with Gaia prior?  Simple linear fit is
  fine for v01.
- **Spectral classification** — colour-cut decision tree (fast, no
  models) vs SED fit to PARSEC isochrones (slower, model-dependent)?
  Recommendation: **colour cuts for v01**, isochrone fit for v02.
- **What to do with non-detection in some bands**: propagate
  upper-limit mag, or leave NaN?  Standard astronomy practice is to
  store upper limit + flag.

----

## Prior art to reuse

- `~/data/HSC_SNIa/euclid_acs_comparison/` — HST↔Euclid match for
  COSMOS done in earlier sessions.  933k pairs, 17k PM stars, frame
  offset measured.  **Read its README first** before re-doing.
- `~/github/projects_euclid/programs/` — the scripts that produced the
  above.  Reuse `astropy.coordinates.SkyCoord.match_to_catalog_sky`
  patterns.

----

## Files NOT to touch (under master programs_webpage control)

- `recut_3_hst.py`, `04_polish_labels.py`, `make_v##_*.py`,
  `csvfiles_sn/*.csv` — SN-finder canonical assets.  Stellar work
  imports utilities from `recut_3_hst.py` (in upstream sibling
  directories) but doesn't modify it.
