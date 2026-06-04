# programs_star — v01 results (2026-05-26)

Pipeline scripts:
```
20_footprints              Step 1  per-tile WCS bboxes, per-band footprint union, common region, per-star coverage lookup
21_dao_detect              Step 2  DAOStarFinder @ 5σ on every mosaic (6-way parallel)
22_classify_stars          Step 3  G2-G4 morphology + SNR≥5 → good_stars; bright + sharp<0.4 → saturated
23_crossmatch_pm           Step 4  brightness-aware cross-match (HST↔Euclid, JWST↔Euclid, HST↔JWST)
24_proper_motion           Step 5  Linear PM fit, Euclid as reference epoch, quiver + histogram
25_orphans                 Step 6  good DAO detections inside another mission's footprint w/o a PM pair
26_make_index              ─       v01 HTML index (htmls/star_v01/index.html)
27_unified_catalog         ─       Multi-band mag join → star_master_v01
28_pixel_coverage_refine   ─       Downsampled SCI masks per tile → drop chip-gap false orphans
29_gaia_validation         ─       Gaia DR3 cone in COSMOS box → completeness, PM zero-point check
30_clean_stars             ─       Tight cuts (sharp 0.5-0.75, ≥4 bands, σ_µ<5) → clean_stars_v01
31_orphan_thumbs           ─       Multi-band 5″ cutouts for top orphans + top high-PM stars
```

DAO wall time: 42 min (33.2 M raw detections, 6 workers).  Everything else < 30 s.

## Counts at each stage

| Stage | Output | Numbers |
|---|---|---|
| Step 1 footprints | 401 tiles, 9 bands | common 9-band region = **0.58 deg²** |
| Step 2 DAO | per-band parquet | **33.2 M** raw detections |
| Step 3 classify | good_stars + saturated_stars | **4.19 M** good, **4 k** saturated |
| Step 4 pairs | HST↔Euclid / JWST↔Euclid / HST↔JWST | 158,962 + 68,699 + 91,550 = **319,211** pairs |
| Step 5 PM | proper_motion_v01.parquet | 132,335 unique Euclid stars; **28,772** with 3-epoch PM; median |µ| = **4.96 mas/yr** |
| Step 6 orphans | orphans.parquet | 1,318,622 bbox-flagged (upper bound) |
| Step 6 high-conf | orphans_highconf.parquet | **294,787** at SNR > 20 + sharp ∈ [0.5,0.75] |
| Step 7 unified | star_master_v01.parquet | 132,335 stars; median n_bands_detected = **4** |
| Step 8 pixel refine | orphans_refined.parquet | 1,144,104 real orphans (13 % drop from bbox-only); mask build 18.3 min for 101 tiles |
| Step 9 Gaia | gaia_match_v01.parquet | **3,839** Gaia matches (22.5 % of Gaia G<21.5 in box; mostly footprint-limited) |
|     | gaia PM scatter | median Δµ_α* = **+2.66 mas/yr**, σ = **11.8** mas/yr; median Δµ_δ = +0.81, σ = 9.9 |
| Step 10 clean | clean_stars_v01.parquet | **56,670** high-confidence stars |
| Step 11 thumbs | orphans_thumbs.html | top 30 per orphan kind + top 30 high-PM |
| Ad-hoc | asteroid_candidates_refined.csv | HST orphan inside JWST **actual pixel mask** + Euclid bbox, no modern pair, SNR > 30, dedup → **1,539** entries |
| Ad-hoc | modern_transient_candidates_refined.csv | JWST/Euclid orphan inside HST **actual pixel mask**, no HST pair, SNR > 30, dedup → **53,667** (still contaminated by extended galaxies) |
| Ad-hoc | asteroid_candidates_all.csv | bbox-only HST orphans (3,667 dedupped) |
| Ad-hoc | highpm_top50.csv | clean stars with |µ| > 50 mas/yr (63 total, top 50 shown) |

## Step 1 (footprints)

| band | n tiles | area (deg²) |
|------|---:|---:|
| F814W (HST)                    | 81 | 2.26 |
| F115W/F150W/F277W/F444W (JWST) | 20 each | 0.58 each |
| VIS + NIR-YJH (Euclid)         | 60 each | 3.91 each |

Common 9-band region: 0.58 deg² (≈ JWST footprint).
A10 f115w uses `scidir/...A10_v0_8_sci.fits` (parent `.fits.gz` is genuinely gzip-corrupt).

## Step 9 (Gaia DR3)

PNGs in `htmls/star_v01/`:
- **gaia_completeness.png** — flat ~25 % across G=15-21; the floor matches the JWST-footprint fraction of the Gaia query box.
- **gaia_pm_compare.png** — scatter follows the 1:1 line; σ ≈ 10-12 mas/yr is the per-star PM error.
- **gaia_pos_compare.png** — positional residuals (after PM-propagating Gaia DR3 to 2024.5).

Median PM zero-points are small (+2.7, +0.8 mas/yr) — good agreement with Gaia. Per-star scatter (~10 mas/yr) is roughly twice the naïve floor from 19-yr baseline + 20 mas centroid (~1 mas/yr) — suggests either residual astrometric offsets between epoch frames or larger per-band centroid uncertainty.

## Step 10 (clean stars)

Tighter cuts to get a high-confidence stellar catalog:
- sharp ∈ [0.5, 0.75]
- n_bands_detected ≥ 4
- |µ| < 500 mas/yr
- σ(µ_ra), σ(µ_dec) < 5 mas/yr
- 2+ epochs

→ **56,670 stars**, median |µ| ≈ 4 mas/yr. Quiver: `clean_pm_quiver.png`. Histogram: `clean_pm_hist.png`.

## Known limitations

1. **Footprints (Step 1) are bbox-only**; refined to actual pixel coverage by Step 8 (drops 13 % chip-gap false positives). Refinement also applied to asteroid/transient candidate lists (see `*_refined.csv`).
2. **Step 3 morphology cuts pass many galaxies.** sharp ∈ [0.4, 0.85] from master CLAUDE.md is loose — many compact galaxies make it in. Step 10's tighter sharp ∈ [0.5, 0.75] cuts ~17 % of the candidates; for an even cleaner stellar sample you'd want sharp 0.55-0.7 + multi-band consistency.
3. **Saturation detector is conservative** (uses brightest good-star peak as threshold) → caught only ~4 k saturated stars total across 9 bands. Bright stars in HST F814W and Euclid VIS likely include many that landed in "good_stars" via DAO sharpness drop. v02: use absolute peak thresholds from FITS header `SATURATE` keyword.
4. **Per-mission position uses only the primary band.** A star detected in F150W but not F115W is lost from the JWST side.
5. **Modern transient candidates (98 k) are heavily contaminated.** Most are extended galaxies whose centroid drifted > 0.4″ across the HST→2024 baseline; a real transient filter needs SED + multi-band consistency.
6. **No AB-mag calibration** of DAO instrumental mags. To compare across bands, multiply by zero-points from FITS headers (HST `ABMAG_ZP`, JWST `PIXAR_SR` × MJy/sr→AB, Euclid header `ZP`).

## Deliverables you'll want to look at

- `htmls/star_v01/index.html` — top-level summary with all images linked.
- `htmls/star_v01/orphans_thumbs.html` — visual diagnostic, top 90 orphans + 30 high-PM stars × 5 bands (475 cutouts total).
- `csvfiles_star/star_master_v01.csv` — primary tabular catalog (132 k × 21 cols).
- `csvfiles_star/clean_stars_v01.csv` — clean subset (57 k stars).
- `csvfiles_star/gaia_match_v01.parquet` — Gaia DR3 cross-validation per-star.
- `csvfiles_star/highpm_top50.csv` — interesting moving-star candidates.
- `csvfiles_star/asteroid_candidates_top200.csv` — HST-only PSF detections inside modern-mission pixel coverage with no pair.
