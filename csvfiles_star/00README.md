# COSMOS cross-mission star catalog — version history

This directory holds the per-mission and cross-mission catalogs built
from HST/ACS F814W, JWST COSMOS-Web, Euclid MER VIS, and Euclid MER NISP
data over the COSMOS field.  Each pipeline step adds value and bumps a
version number on the downstream files.

## File-naming convention

```
cat_matched_<MISSION>.{parquet,csv}              ← Step 3 (per-mission)
cat_matched_<MISSION>_with_gaia.{parquet,csv}    ← Step 4 (per-mission)

cross_<COMBO>.{parquet,csv}                      ← Step 5 (cross-mission intersection)
refined_<COMBO>.{parquet,csv}                    ← Step 6 (point-source flagged)
refined_<COMBO>_with_pm_v<NN>.{parquet,csv}      ← Step 7 (proper motion)
```

where `<MISSION>` ∈ {HST, JWST, Euclid_VIS, Euclid_NISP} and `<COMBO>`
is an underscore-joined subset of those four missions.

## Pipeline steps

| Step | Script | Output prefix | Description |
|---|---|---|---|
| 3 | `52_step3_catalog_match.py` | `cat_matched_*` | DAO detections matched within 1 mosaic pixel to each mission's published catalog (HST ACS i-phot, COSMOS-Web v1.1, MER DR1). Catalog ID is the primary key; unmatched DAO rows are dropped. |
| 4 | `53_step4_gaia_augment.py` | `cat_matched_*_with_gaia` | Gaia DR3 cross-match per mission, propagating Gaia positions to the mission epoch using each star's own PM (avoids high-PM mismatches). Bright Gaia stars (G<18) missing from a mission catalog (saturation) are added as `gaia_only_bright` rows so the bright end isn't lost. |
| 5 | `54_step5_cross_match.py` | `cross_*` | Intersection (AND) cross-match across 9 mission combinations (6 pairs + 2 triples + 1 four-way master). Each row is one unique star detected by EVERY listed mission. Gaia source_id is the gold-standard tie, with positional matching for catalog-only stars. `gaia_only_bright` rows do NOT count as a detection for the intersection. |
| 6 | `55_step6_refine_pointsource.py` | `refined_*` | Per-mission ellipticity threshold (95th-pctl of Gaia+QSO training set in each magnitude bin, capped at 0.30) combined with catalog star/PS flags to set `is_point_source`. Gaia-anchored stars and COSMOS-Web AGN/QSOs (LePhare type=2 ∪ flag_chandra) override to True. |
| 7 | `57_step7_pm_v2.py` | `refined_*_with_pm_v<NN>` | Apparent proper motion per source from the two missions' catalog positions in the user's sign convention (mission − reference), with per-source error (per-mission systematic + SNR-dependent centroid), divided by `epoch_m1 − epoch_m2`. Gaia-only-bright sources brought in as supplements where one mission saturates the star (`pm_method = gaia_m1` / `gaia_m2`). |

## Version history of `refined_<COMBO>_with_pm_v<NN>`

### v01 — 2026-05-27, first PM measurement with Gaia supplements

- Direction: `m1 − m2` where m2 is the reference (more recent / Gaia-tied).
- Sources: cat_cat (both catalog) + gaia_m1 + gaia_m2 supplements.
- Hard cap |pmtot| < 1000 mas/yr to remove obviously-broken cross-matches.
- Per-row error from per-mission σ_axis (HST 30 mas, JWST 10 mas, VIS 15 mas, NISP 30 mas) plus SNR-dependent centroid noise, divided by |Δt|.
- **Known issue**: a long tail (max ~940 mas/yr in HST × Euclid) came from Gaia source_id mismatches — the same Gaia ID linked to two unrelated physical sources (typically a Gaia-saturated bright star whose nearest catalog detection in one mission is a saturation-core artifact while the other mission picked up a different neighbour). Inspect via `pm_method`, `gaia_source_id`, and compare `pmtot` to `sqrt(gaia_pmra² + gaia_pmdec²)`.

### v02 — 2026-05-27, cleaned: bogus PMs NaN'd, real PMs kept

- Starts from v01.  All other columns unchanged.
- Adds **`pm_flag`** column with values:
    - `ok` — no Gaia PM available, |pmtot| within hard cap.
    - `gaia_consistent` — Gaia PM available, `|pmtot − gaia_pmtot| / max(gaia_pmtot, 20) < 0.5` (within 50% relative, with a 5 mas/yr noise floor on the denominator).
    - `inconsistent_with_gaia` — Gaia PM available, |pmtot| > 100 mas/yr, and disagrees with Gaia by more than 50%. **PM columns NaN'd in v02.**
    - `over_hard_cap` — |pmtot| > 600 mas/yr (no Gaia DR3 star in COSMOS exceeds 500). **PM columns NaN'd.**
    - `no_gaia_extreme` — no Gaia, |pmtot| > 600. **PM columns NaN'd.**
- Result: ~30–90 rows per combo NaN'd as bogus; the rest preserved.

Versions co-exist on disk so you can always reproduce v01-style scatter plots or audit a specific source. v02 is what to use for downstream analysis.

## Real high-PM survivors in v02

After cleaning, the 11 unique Gaia stars with |pmtot| > 100 mas/yr that survived (Gaia-consistent measurements):

```
gaia_source_id       gaia_G  measured_pmtot  gaia_pmtot  agreement
3836028810399121408    18.83          248.5       248.1   0.2%
3836369311111664128    17.35          199.3       206.2   3.4%
3836024481072083968    17.75          120.7        87.2  38%  (JWST×VIS, Δt=0.5 yr noisy)
3835994828618242048    16.17          119.9       118.5   1.1%
3836388208967847424    18.32          114.7       114.1   0.5%
3836286714595353600    16.62          114.0       116.3   1.9%
3836234659592087552    15.47          110.4        80.8  37%  (JWST×VIS, noisy)
3834725201925833728    18.43          108.8       101.9   6.8%
3836367386966323200    17.00          108.6       107.2   1.3%
3836261597626954752    16.06          107.1       111.2   3.7%
3836389140975735296    19.29          101.5        97.7   3.9%
```

The two JWST×Euclid_VIS rows (Δt = 0.5 yr) are noisy because of the short
baseline (NMAD ≈ 22 mas/yr per axis = 31 mas/yr in pmtot), so a 30–40%
disagreement with Gaia is still within the per-source uncertainty.

## Mission epochs

| Mission | Epoch |
|---|---:|
| HST/ACS F814W | 2005.0 |
| JWST COSMOS-Web | 2024.0 |
| Euclid VIS | 2024.5 |
| Euclid NISP | 2024.5 |

## Five intersection combinations with PM

| Combo | Δt (yr) | N (v02) | n_cat_cat | n_gaia_m1 | n_gaia_m2 |
|---|---:|---:|---:|---:|---:|
| HST − Euclid_VIS | −19.5 | 106,904 | 106,006 | 644 | 254 |
| HST − Euclid_NISP | −19.5 | 91,653 | 90,769 | 630 | 254 |
| HST − JWST | −19.0 | 54,553 | 54,254 | 17 | 282 |
| JWST − Euclid_VIS | −0.5 | 99,435 | 98,870 | 560 | 5 |
| JWST − Euclid_NISP | −0.5 | 86,505 | 85,996 | 504 | 5 |

(VIS × NISP is omitted from PM analysis because MER assigns one
RA/Dec per source across all Euclid bands; Δt = 0 by construction.)

## Diagnostic plots

PM scatter and per-category quiver plots are in
`../htmls/pm_v01/pm_<COMBO>_{scatter,quiver_all,quiver_stars,quiver_agn_qso}.png`.
Solid arrows are cat-cat; dotted arrows (red edges) are gaia-supplement rows.

## Reproduction

All processing scripts live in `../programs_star/` (see numbered file
prefixes 52–57).  Each script is self-contained and writes to this
directory; rerunning a script overwrites only the files it owns.
The Gaia DR3 cache (`gaia_dr3_cosmos_v2.parquet`) is kept here to avoid
re-querying the Gaia archive.
