# programs_star — unified cross-mission stellar catalog

**Master criteria & conventions:**
`/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
Read that first — this file is a directory-scoped extension that
carries forward only what's relevant for cross-mission stellar
catalog construction.

----

## Scope

`programs_star/` is the **aggregator** for the stellar catalog project.
It does **not** do raw FITS / DAO work — that happens in the three
sibling directories:

```
programs_hst/      → per-tile F814W point-source catalog
programs_jwst/     → per-tile 4-band F115/F150/F277/F444W catalog
programs_euclid/   → per-tile VIS + NISP-Y/J/H catalog
       ↓                    ↓                    ↓
        ──────────► programs_star/ ◄────────────
              cross-mission match
              proper-motion measurement
              SED / colour classification
              external validation (Gaia DR3)
              unified catalog export
```

The deliverables of this directory are:

1. **Unified stellar catalog** — one row per star with positions
   (per-mission, epoch-tagged), magnitudes (8+ bands across 3 missions),
   morphology stats, PM (when measurable), Gaia cross-match status.
2. **Proper-motion catalog** — for stars detected at ≥ 2 epochs (HST
   2003-2007 vs JWST/Euclid 2024), measure tangential proper motion.
3. **Stellar-class catalog** — colour-based spectral classification
   using the 8+ band photometry.

This directory's outputs feed back to the SN-finder work (master §2.6
validation), since point-source criteria need a star control sample.

----

## Inviolable principles (carry-over from master §1)

1. **PSF / morphology gates were applied UPSTREAM.**  By the time data
   reaches `programs_star/`, each candidate has already passed
   G1-G4 in its own mission.  Don't re-derive morphology here.  DO
   propagate the morphology numbers (sharp, rnd1, rnd2 per band) into
   the unified catalog so downstream consumers (us, SN-finder
   validation, the user) can apply tighter cuts if needed.
2. **Epoch matters.**  HST is 2003-2007, JWST is 2023-2024, Euclid is
   2024-2025.  Cross-matching MUST be PM-tolerant for HST↔JWST
   (≥ 17 yr baseline → up to ~0.3″ for typical disk stars, ~1″ for
   high-PM stars).  JWST↔Euclid is contemporaneous (~0″ baseline) so
   the tolerance can be much tighter.
3. **JWST is the reference frame.**  Tied to Gaia DR3 by the
   COSMOS-Web pipeline (~10-30 mas).  HST and Euclid positions get
   matched **into** JWST's coordinate system.
4. **No overwrites of versioned directories.**  Use the same `v##`
   convention.  Catalog versions go in `csvfiles_star/star_unified_v01.csv`
   (or similar, see §3 of master CLAUDE.md for the naming convention
   pattern: `<deliverable>_<N>_v<NN>.csv`).
5. **Commit & push when a version is approved.**  Canonical artefacts:
   the unified catalog CSVs in `csvfiles_star/`.  Diagnostic PNGs
   (colour-colour diagrams, PM quiver plots, etc.) may stay untracked.
6. **PSF / point-source morphology is the primary discriminator** — even
   though it's applied upstream, when we hit ambiguities (e.g.,
   sources detected only in NISP), defer to the more reliable
   morphology measurement (typically JWST F115W → JWST F150W →
   HST F814W → Euclid VIS → Euclid NISP, in decreasing PSF quality).

----

## Cross-mission matching specifics

### Match radii (working values; revise after validation)

| pair          | epoch baseline | tolerance | rationale                                |
|---------------|---------------:|----------:|------------------------------------------|
| JWST ↔ Euclid | ~0 yr          | 0.15″     | both ~2024; differences = WCS + centroid |
| HST  ↔ Euclid | 17-22 yr       | 0.40″     | adds PM for typical disk stars           |
| HST  ↔ JWST   | 17-21 yr       | 0.40″     | same                                     |

For known-high-PM stars (Gaia-flagged) loosen to 1.5″ and use the Gaia
PM vector to predict the offset before matching.

### Frame & coordinates

- All positions in **ICRS J2000**.
- Catalog stores per-mission RA/Dec separately, **with epoch tag** —
  don't blend positions.  Downstream queries pick the relevant epoch.
- "Reference RA/Dec" column = JWST coordinates when available, else
  Euclid VIS, else HST.

### Magnitudes

- Each mission catalog provides AB magnitudes computed with the
  upstream mission's standard (HST `ABMAG_ZP`, JWST MJy/sr→AB via
  `PIXAR_SR`, Euclid header `ZP`).
- Cross-mission colour terms (HST F814W vs JWST F115W, etc.) are
  filter-dependent but not band-corrected in v01.  Note in the doc;
  add a colour-correction step in a later version if needed for
  precision colour-colour analysis.

----

## Prior art relevant to this directory

From the user's project memory (already done in earlier sessions):

- **Euclid × ACS COSMOS comparison** (in `programs_euclid/` or
  `~/data/HSC_SNIa/euclid_acs_comparison/`): 933k matched pairs;
  ACS-Euclid frame offset +67/+5 mas (from galaxies); 17k stars have
  measurable PM (median 6.6 mas/yr, 95% @ 22 mas/yr).
- This means the HST↔Euclid match has been done before for the same
  field.  Re-use the methodology (and possibly the matched-pair table
  itself) instead of reinventing.  Check
  `~/data/HSC_SNIa/euclid_acs_comparison/` first.

----

## Reusable building blocks from upstream

- Per-mission catalog schemas (defined in each `programs_<mission>/
  handoff.md` §"Proposed next-step plan"): all 4 catalogs share a
  baseline of `(id, ra, dec, epoch, mag_<band>, snr_<band>,
  sharp_<band>, rnd1_<band>, rnd2_<band>, sat_flag, n_bands_detect)`.
- `astropy.coordinates.SkyCoord.match_to_catalog_sky()` for fast
  KDTree-based positional matching.  Use `nthneighbor=1` for unique
  pairs; `nthneighbor=2` to detect ambiguous matches and flag them.
- Don't build a per-tile loop here — programs_star reads catalogs
  that already aggregate over tiles.

----

## Open items for v01

- Decide on canonical column naming (`mag_F814W` vs `mag_hst_F814W`
  etc.).  Recommendation: prefix bands with mission when there's
  potential collision (e.g., Euclid VIS and a hypothetical HST WFC3
  would both be ~optical broadband).  For COSMOS we have a unique
  band → mission mapping so prefix is optional.
- External truth datasets to validate against:
  - Gaia DR3 (positions, PMs, full sky)
  - COSMOS2025 / COSMOS2020 photo-z catalogs
  - Euclid MER `phz_classification=1` (Euclid's own star flag)
- Decide on PM measurement methodology (linear-trend fit across
  available epochs, or Bayesian with Gaia prior).
- Spectral-class assignment: simple colour-cut decision tree
  (M/K/G/F/A) vs SED fitting (e.g., PARSEC isochrones).

----

## Known bugs in master_or_catalog_v01 (found 2026-05-27)

Discovered during SN-search v01 visual inspection
(`programs_webpage/make_v01_polished_web_opt.py` output reviewed
candidate by candidate).  All issues feed back into a planned `v02`
master regeneration.

### B1. `primary_id` is NOT globally unique
- HST `hst_id` is per-tile (numbered 1..N per tile), so the same
  `hst_id=325` represents different physical sources across overlapping
  HST tiles. `primary_id` of form `HSTID_<hst_id>` therefore collides.
- Quantified on v01: HST `primary_id` prefix has dup ratio **29.4×**
  (90,824 rows for only 3,086 unique HSTIDs).  EuclidID and JWST_numeric
  are 1.0× (unique).  Max collision: `HSTID_192` appears **121** times.
- Total: 89,934 of 1,132,959 rows (~8%) share a colliding `primary_id`
  with at least one other row.
- **Downstream impact**: per-row filters (e.g., "exclude rows with
  Gaia PM") leak — only the specific rows carrying Gaia info get
  excluded, leaving sister rows of the same physical source as
  non-Gaia candidates.  Found 28,701 such leaked rows.
- **Fix proposal (v02)**: append cluster index to `primary_id`
  (`HSTID_325_C0042`).  OR: do within-mission positional dedup BEFORE
  the cross-mission union-find (preferable but a deeper upstream
  change in `programs_hst/`).

### B2. `is_point_source` misses obvious stars
v01 marks `is_point_source=True` only when a source appears in one of
the 9 `refined_*` catalogs with the upstream ellipticity threshold
satisfied. This misses:

  - **Faint Gaia stars with no PM measurement.** Gaia source_id is
    present but `gaia_pmra/pmdec` are NaN (typical for G > 20).
    Currently 9,021 unique `primary_id`s carry a Gaia source_id, but
    only 7,660 have measured PM, leaving 1,361 confirmed Gaia stars
    unflagged.
  - **Euclid PHZ stars** (`cat_phz_classification ∈ {vis,nisp} == 2`)
    not detected in HST (e.g., faint or saturated → HST catalog drops
    them).  Several v01 SN candidates (cand_26, cand_39, etc.) had
    `cat_phz_classification_vis = 2` or `_nisp = 2` but
    `is_point_source = False`.
  - **HST mu_class stellar** (`cat_mu_class_hst == 1`) with low
    `cat_class_star_hst` (the two HST flags can disagree); current
    aggregation ignores `mu_class`.

### B3. Saturated stars missing from HST catalog → no PM measurement
The per-tile HST DAO pipeline in `programs_hst/` rejects saturated
sources (bleed trails, distorted PSFs).  Consequence:
  - These stars never get an `hst_id` in `cat_matched_HST.parquet`.
  - They cannot pair with their Euclid / JWST counterparts in the
    cross-mission PM analysis (HST_Euclid_NISP, HST_Euclid_VIS,
    HST_JWST), so `pmtot_*` is always NaN even when visible proper
    motion is obvious to the eye.

**Cases found in SN-search v01 visual inspection (2026-05-27):**
  - **cand_39**: NISP-detected, NISP PHZ classification=2 (STAR), HST
    imagery clearly shows a saturated star, but no `hst_id` → no PM.
  - **cand_98**: HST imagery and Euclid imagery both show the star,
    user sees "very big proper motion" between HST and Euclid. Master
    shows `cat_point_like_prob_vis=0.989` (STAR), `is_point_source=True`,
    but `cat_mag_F814W_hst=NaN`, `cat_ra_hst=NaN` (HST catalog missing).
    PM pairs all NaN — no cross-mission measurement possible.

**Fix proposal (v02 master, no programs_hst rerun needed)**:

Add column `hst_saturated_likely` = True when ALL of:
  - `cat_point_like_prob_vis > 0.9` OR `cat_point_like_prob_nisp > 0.9`
    OR `cat_phz_classification_vis == 2` OR `cat_phz_classification_nisp == 2`
  - `cat_ra_hst is NaN` (i.e. HST catalog missed this source)
  - Source position is INSIDE an HST tile footprint (covered but not
    catalog-detected → strong saturation evidence)

These sources MUST be flagged in `is_likely_star`.

Additional v02 column proposal `large_pm_likely` = True when:
  - `is_likely_star == True` (any of the above star flags), AND
  - one of:
      (a) `any pmtot_<pair> > 100 mas/yr` regardless of pm_flag (Gaia DR3
          high-PM stars come up here)
      (b) raw position delta between any two missions' cat_ra/cat_dec
          implies > 30 mas/yr (catches B5 cases where the PM measurement
          script understates motion)
      (c) `hst_saturated_likely == True` (likely a bright nearby star
          with large PM that saturated HST and Gaia)

This `large_pm_likely` flag goes downstream as a strong star-candidate
indicator even when explicit PM measurement is broken (B3, B5).

**Fix proposal (long-term)** in programs_hst:
  - Rerun the per-tile catalog with a relaxed saturation mask (or a
    secondary pass that admits sources flagged as saturated with their
    sat_flag set).  Then re-run programs_star steps 3–7.

### B4. New aggregate column `is_likely_star` (v02 master)

**FINAL implemented definition** (in `59_master_or_catalog_v02.py`, verified
to give 0/12 false-positives on the 12 known SNe present in master, and
~9% star fraction overall which is realistic for COSMOS):

```
is_likely_star = (
    NOT is_agn_qso                                     # not an AGN/QSO
    AND (
        gaia_source_id present                          # Gaia anchor (any)
        OR cat_class_star_hst > 0.8                     # SExtractor stellarity
        OR cat_phz_classification_vis  == 1             # Euclid PHZ: 1=STAR
        OR cat_phz_classification_nisp == 1             #   ↑ verified by counts
        OR cat_point_like_prob_vis  > 0.9
        OR cat_point_like_prob_nisp > 0.9
        OR cat_mag_VIS_vis  < 17                        # bright VIS → likely star
        OR cat_mag_F814W_hst < 18                       # bright F814W → likely star
        OR hst_saturated_likely                          # B3 condition
        OR pm_significant                                # see below
    )
)
```

**Encoding corrections made during implementation:**
1. **Euclid `phz_classification`: 1=STAR, 2=GALAXY, 3=QSO** (not the
   reverse I initially assumed). Verified from distribution: 58% of
   sources have phz=2, only ~3% have phz=1 — consistent with
   galaxy-dominated deep survey.
2. **HST `cat_mu_class_hst` encoding is inverted from naïve "1=star"**:
   86% of HST rows have mu_class=1 (must be "extended/galaxy"); only 12%
   have mu_class=2 (probably star). v02 does NOT use mu_class; relies
   on numeric cat_class_star_hst > 0.8 instead.

**EXCLUDED from is_likely_star** (deliberately):
- `is_point_source` — SNe ARE point sources by definition; including this
  would exclude real SN candidates. The CNN + morphology + cross-survey
  asymmetry rules downstream discriminate SN-like vs star-like.

**`pm_significant` definition** (tiered to avoid astrometric residuals):
- `pm_flag = "gaia_consistent"`: trust at 3σ + 5 mas/yr
- `pm_flag = "ok"`: stricter (10σ AND 20 mas/yr) — without Gaia anchor,
  cross-mission centroid residuals can fake ~5 mas/yr signals
- `pm_flag` other (None, inconsistent_with_gaia, etc.): ignored

Downstream SN-finder uses `is_likely_star OR is_agn_qso` as the single
star/AGN exclusion criterion.

### B5. `pmtot_*` cross-mission values look small even when raw position
deltas indicate large motion (cand_26: raw positions imply ~24 mas/yr,
but `pmtot_HST_Euclid_NISP = 5.7`). Centroid / sign convention may
differ between the cat positions and the PM-fit positions. Audit the
`56_step7_proper_motion.py` / `57_step7_pm_v2.py` centroids vs the
`cat_ra/cat_dec` values used in `59_master_or_catalog.py`.

### B6. PM info is COPIED to all colliding-hst_id rows (consequence of B1)
`59_master_or_catalog.py` lines 442-456 join PM values from
`refined_<pair>_with_pm_v02.parquet` by `hst_id_hst` only.  Because
B1 says `hst_id` is per-tile (not globally unique), the SAME pmtot
value is attached to ALL master rows sharing that hst_id, regardless
of whether they are the same physical source.

Concrete case (cand_65 inspection, 2026-05-27):
  - HSTID_105 has **84** rows in master.
  - Row idx=13938: real star, RA=149.7615, Dec=1.6372,
    cat_class_star=0.989, mag_F814W=19.63 → real PM 8.52 mas/yr.
  - Row idx=17575: galaxy, RA=149.832, Dec=1.715 (378″ away from idx=13938),
    cat_class_star=0.0009, mag_F814W=22.57 → wrongly carries pmtot=8.52.

**Fix in v02 master:** join PM info by `(hst_id, hst_tile)` tuple, not
`hst_id` alone.  Or first deduplicate hst rows by adding a globally
unique `hst_uid = f"{tile}_{hst_id}"` column and join on that.


----

## Step 3a — empirical PSF models (v04 `programs_star/v04/`)

Per-band empirical PSF models built with PSFEx, following Tanaka+2023
(COSMOS-Web NIRCam, same 30 mas mosaics).  Pilot tile = **A4**, validated
on all 4 JWST bands + Euclid VIS; HST ACS F814W adapter ready.

### Scripts (reproducible)
- `54_step3a_build_psf_model.py` — JWST builder (SExtractor pass-1 →
  star selection → PSFEx).  `_euclid.py` / `_hst.py` variants for the
  single-HDU mosaics (ZP from MAGZERO / PHOTFLAM·PHOTPLAM).
- `56_make_psf_qa_plots.py` (+ `_euclid.py`) — 6 QA plots per (band,tile).
- `55_make_psf_qa_site.py` — static QA site at `html/psf_qa/`.
- `57_run_jwst_psf_qa.sh` — **one-command reproducible driver** for all
  4 JWST bands on a tile (runs 54→56 per band, then 55).
- `deblend_sweep_A4.log`, `HANDOFF_psf_qa.md` — sweep record + handoff.

### "Good star" selection (LOCKED — validated on F115W A4)
- `CLASS_STAR>0.8`, `SNR_WIN>100`, `ELONGATION<1.5`, `ELLIPTICITY<0.20`
- `FWHM_IMAGE > 0.5 × PSF_FWHM` — kills 1-px CR/hot-pixel artifacts (a
  *second locus* at FR≈0.8 px that otherwise inflates the MAD; visually
  confirmed as cosmic rays).  `PSF_FWHM` = median FWHM of the SNR>1000
  subset (real stars dominate).
- `locus − 3·MAD < FLUX_RADIUS < locus + 2.5·std`  (iterative MAD locus)
- `masked_core == 0` (JWST i2d masks saturated cores to **0**, not NaN;
  SExtractor SATUR flag is NOT triggered — must test pixels == 0)
- `gap_masked`: >5 zero pixels in the VIGNET footprint (chip gaps)
- `NOT (n_1 ≥ 2 AND FLAGS < 2)` — neighbour contamination; the FLAGS<2
  conjunction is essential because bright stars' own diffraction spikes
  get deblended (FLAGS≥2) and would otherwise be cut as "neighbours".
- PSFEx: `SAMPLE_FLAGMASK 0x00fc` (allow neighbour+deblend bits),
  `SAMPLE_MAXELLIP 0.18`.

### TWO per-filter knobs (the hard-won lessons)
1. **`DEBLEND_MINCONT` per band** (in `configs/jwst_nircam_<band>.sex`):
   F115W=0.02, F150W/F277W/F444W=**0.05** (vs old 0.0005).  Aggressive
   deblending split bright stars' diffraction spikes into separate
   objects → spike pixels blanked (−1e30) in the VIGNET.  Higher MINCONT
   keeps spikes ATTACHED so they become part of the PSF model.  LW spikes
   need 100× the old value.  (Sweep metric: % bright FLAGS<2 + mean n_1;
   see `deblend_sweep_A4.log` + `pilot/deblend_knee_A4.png`.)
2. **`SAMPLE_FWHMRANGE` scaled to PSF FWHM** — computed dynamically in
   54_ as `[0.6, 2.5]×PSF_FWHM`, written to the meta JSON, re-read by 56_.
   The fixed config range (LW 2–6 px) cut THROUGH the F444W locus
   (PSF FWHM 5.9, bright stars to 8.7 px) and rejected EVERY bright star.
   This was the real cause of "bright stars on the locus rejected", NOT
   saturation/gaps/spikes.  Fix took F444W from 60→186 accepted.

### Final A4 results
| band | DEBLEND_MINCONT | FWHMrange (px) | locus (px) | cut (px) | cand | acc | χ² |
|------|---|---|---|---|---|---|---|
| F115W | 0.02 | 1.5–5.0  | 1.29 | 0.90 | 229 | 127 | 1.38 |
| F150W | 0.05 | 1.5–5.5  | 1.37 | 1.04 | 335 | 251 | 1.31 |
| F277W | 0.05 | 2.6–10.9 | 2.39 | 1.93 | 126 | 100 | 1.67 |
| F444W | 0.05 | 3.4–14.0 | 3.02 | 2.69 | 245 | 186 | 1.54 |

All 4 LW/SW models visibly reproduce the NIRCam diffraction-spike
pattern (the science goal).  χ² is SNR-weighted, so it rises when bright
spike stars are included — that is EXPECTED, not a regression.

### QA plot conventions (56_)
- mosaic cell size auto-detected (SW 101, LW 151, Euclid 51 px); panels
  forced square via `aspect='equal'` + figsize matched to data aspect.
- mag-vs-halflight: solid red = PSFEx-accepted, open red = candidate
  rejected; black-X saturated; blue artifacts.
- residual stretch = sqrt, 99.5% of own range; red frame = χ²>1.6.
