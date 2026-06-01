# HANDOFF → COSMOS Star Catalog Project session: rebuild master as v04

**Context (decided 2026-06-01, in the SN-search session):**
A completeness bug was found and the first fix is done. The star-catalog
session should carry the rebuild from step-4 onward.

## The bug (already fixed at step-3)
`52_step3_catalog_match_v03.py` kept JWST catalog sources ONLY if a DAO peak
landed within `JWST_PIX_AS = 30 mas` in ≥1 band — making DAO a SURVIVAL GATE.
This silently dropped **273,308 (35.5%)** of in-footprint COSMOS-Web sources,
including on-host SNe whose DAO peak merges into the host light (e.g. confirmed
SNe 468896, 19931, 39020). Binding principle: **DAO absence ≠ SN absence**
(master CLAUDE.md §7.0, §8).

## What's already done (committed, do not redo)
- **`programs_star/52_step3_catalog_match_v04.py`** — fixed step-3. Keeps ALL
  768,840 in-footprint JWST catalog sources; DAO photometry is now an
  ANNOTATION (filled where matched within 30 mas, NaN otherwise), not a gate.
- **`csvfiles_star/cat_matched_JWST_v04.parquet`** — 768,840 rows × 85 cols
  (was 495,532). Verified: all 21 numeric-id master SNe present, incl. the 3
  that v03 dropped. Same schema as `cat_matched_JWST_v03.parquet`.
- HST / Euclid-VIS / Euclid-NISP step-3 tables UNCHANGED — reuse the v03 ones
  (`cat_matched_{HST,Euclid_VIS,Euclid_NISP}_v03.parquet`). Only JWST changed.

## What the star-catalog session needs to run (the v04 chain)
Re-run these on the v04 JWST table (others = v03), producing `_v04` outputs.
Each is currently hard-wired to `_v03` paths — copy to `_v04` and swap the
JWST input to `cat_matched_JWST_v04.parquet`:

1. `53_step4_gaia_augment_v03.py`     → `cat_matched_JWST_with_gaia_v04.parquet`
2. `54_step5_cross_match_v03.py`      → cross-mission union (HST/Euclid v03 + JWST v04)
3. `55_step6_refine_pointsource_v03.py`
4. `56_step7_proper_motion_v03.py` / `57_step7_pm_v2_v03.py` → `refined_*_with_pm_v04`
5. `59_master_or_catalog_v03.py`      → **`master_or_catalog_v04.parquet`** (the deliverable)

Key watch-items (from master CLAUDE.md §B1-B6, all should already be handled
in the v03 logic — just verify they survive the bigger input):
- B1: globally-unique primary_id (per-tile hst_id collisions).
- B3: hst_saturated_likely flag.
- B4: is_likely_star aggregation (Euclid phz 1=STAR; HST mu_class inverted).
- The 516,298 newly-kept DAO-absent JWST sources will have NaN dao_* columns —
  make sure is_likely_star / is_point_source aggregations treat NaN-DAO as
  "not a confirmed point source" (these are mostly faint galaxies / on-host —
  exactly the SN-candidate population, correctly NOT flagged as stars).

## Then hand BACK to the SN-search session
When `master_or_catalog_v04.parquet` exists, the SN-search session will:
- `make_sn_candidates_v04` (star/AGN exclusion) → `sn_candidates_v04`
- `make_fits_lookup_v04` → `fits_lookup_v04` (the new ~769K-based pool)
- re-run v07 inference on the bigger pool.

## PSFEx note (also for the star-catalog session, later)
User wants PSFEx-grade photometry on top of DAO eventually. PSFEx/SExtractor
are NOT installed (no binary). For catalog-grade work, install the Astromatic
toolchain in that session. (The SN-search session is meanwhile using
photutils EPSF + forced PSF fitting for the on-host-SN residual channel.)
