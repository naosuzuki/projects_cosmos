# COSMOS SN identification — working criteria & conventions

**Status: working baseline as of v29 (2026-05-26).** These are the rules
in force right now. They are **subject to revision** as we test against
star / host-galaxy / random-position control samples and as we expand
beyond the 11 known SN to mass production (1000+ candidates). Always
check at the top of a session whether the user has updated these.

The user (Nao Suzuki) has reminded me repeatedly that the conversation
summary tends to compress *algorithmic* decisions into oblivion while
preserving *task progression*. This file exists so I don't re-learn the
same lessons from his reactions each session.

----

## 1. Inviolable principles (do not deviate without explicit user OK)

1. **PSF / point-source morphology is the PRIMARY discriminator.** S/N is
   a tie-breaker, never a primary picker. Sorting by S/N alone — peak,
   peak/bg_std, aperture S/N, anything — has been the source of every
   misidentification in this project. The user has had to correct me on
   this at least seven times across versions v17–v27.

2. **Host position ≠ SN position.** Never use host coords as a SN proxy.
   Never report the host as "the SN" — even if catalog `sep_host` is 0,
   that is a real overlap, not a missing-SN.

3. **Best-S/N band is picked per source, never per telescope.** The
   per-telescope fixed-band mapping (HST→F814W, JWST→F115W, EUCLID→NIR-Y)
   is only a fallback when multi-band scan is not possible. The default
   is to scan every band of the source's own telescope and pick by the
   rules below.

4. **No overwrites of v## directories.** Every iteration is a new
   version. v22 → v23 → v24 → … never edit v22 in place. The historical
   record matters; the user reviews regressions by stepping back through
   versions. `_crosshair_test/00README` and `_crosshair_test/index.html`
   are the version index.

5. **Sign conventions for image-space references.** Astronomy: N = up,
   E = LEFT. So "1.2″ left" = "1.2″ east" = **+1.2 / 3600 / cos(dec)**
   added to RA. "0.1″ above" = "0.1″ north" = +0.1/3600 added to Dec. I
   have made the sign-error mistake on RA at least twice; the offset must
   be `+offset_arcsec / 3600 / cos(dec_rad)` for east.

6. **Versioning every iteration.** When the user says "do X again" after
   a correction, increment the version. Never silently rewrite an
   existing version's artefacts.

7. **Commit & push when a version is approved.** When the user signs off
   on a version ("looks good", "all good", etc.), the canonical
   artefacts get a git commit + push.  Canonical = the CSVs in
   `csvfiles_sn/` and the top-level `_crosshair_test/index.html`; the
   per-version PNG decks may or may not be committed (large, regenerable
   from scripts + CSVs — leave the call to the user).  Analysis scripts
   (`make_v##_*.py`, `add_aper_snr_*.py`, etc.) and `CLAUDE.md` itself
   should be committed so the workflow is reproducible on a fresh clone.
   The user runs `git commit` + `git push` themselves; my job is to
   flag what's untracked when a milestone closes so they can choose
   what to include.

8. **Knowledge transfer to sibling directories.** When the user points
   me to a new working directory ("now work in /path/to/X"), I write a
   scoped `CLAUDE.md` there.  The scoped file is NOT a copy of this one;
   it is a directory-specific extension that:
     - Opens with a pointer back to THIS master:
       `/Users/suzuki/github/projects_cosmos/programs_webpage/CLAUDE.md`
       and says "read that first".
     - Carries over only the principles relevant to that directory's
       task (typically: morphology criteria §2, bug patterns §2.5,
       canonical CSV paths §3, commit-and-push §1#7).
     - Records the directory's own scope, files, and conventions.
   This is so that the algorithmic decisions we earned here (PSF-first,
   sep_sn gate, snr_aper picker, host ≠ SN, no-overwrite versioning,
   commit-on-approval) propagate to wherever the user splits the work
   next, instead of getting lost between sessions.

----

## 2. Working criteria for morphology-based SN identification

These are the cuts validated against the 11 known SN in v29
(2026-05-26). All 11 SN are correctly identified, zero
misidentifications.

### 2.1 Per-band eligibility gates (must pass ALL four)

A band is "PSF-like detection at the catalog SN position" only if:

| gate | rule | rejects                                                |
|------|------|--------------------------------------------------------|
| G1   | `sep_sn ≤ max(0.10″, 1.5 × FWHM_band)`  | bands where DAO peak is on host/blend (246188 F277W) |
| G2   | `0.40 ≤ sharp ≤ 0.85`                    | extended hosts (`>0.85`), CRs/blends (`<0.40`)        |
| G3   | `|rnd1| ≤ 0.50`                          | x/y-axis asymmetry from blending                       |
| G4   | `|rnd2| ≤ 0.50`                          | diagonal asymmetry from blending                       |

`sep_sn` = arcsec from catalog SN position to closest DAO 3σ candidate
in that band. `FWHM_band` is the empirical PSF (see §5).

### 2.2 Detection significance (secondary)

`snr_aper` = PSF-matched aperture S/N at the **catalog SN coord** (NOT
at the DAO peak; the DAO peak can be on the host).

  * aperture radius = 1.0 × FWHM_band  (SNR-optimal for a Gaussian)
  * sky ring 2.0–3.5 × FWHM_band

A band is "detected" if it passes G1-G4 **and** `snr_aper ≥ 3σ`.

### 2.3 Multi-band SN confirmation

For the SN to be classified as "real":

| telescope | rule                                                                            |
|-----------|---------------------------------------------------------------------------------|
| HST       | F814W passes G1-G4 AND `snr_aper ≥ 5σ`                                          |
| JWST      | ≥1 band with `snr_aper ≥ 5σ` AND ≥2 bands with `snr_aper ≥ 3σ`; all pass G1-G4  |
| Euclid    | same as JWST (skip VIS — different epoch)                                       |

### 2.4 Best-band selection

Among bands passing G1-G4, the band with the **highest `snr_aper`** wins.
This is the band rendered on the diagnostic PNG and used for the
"selected SN" DAO parameters.

**Important:** never pick the best band by `peak/bg_std` of the DAO
candidate. That metric is hijacked when a redder/broader-PSF band's
closest DAO is actually the host (it was the v17/v19/v20/v27 bug).
`snr_aper` is computed AT THE SN COORD and is not hijackable that way.

### 2.5 Pre-flight bug-watch

Five patterns I have re-introduced repeatedly. Every script that selects
a best band MUST check for all five:

1. Sorting only by S/N (any flavour) and skipping morphology gates.
2. Using DAO peak / bg_std (or any DAO-peak-derived S/N) as the picker
   while the DAO peak is potentially 0.2–1.0″ from the SN coord.
3. Aperture sums that include nearby host flux because the
   PSF-matched ring lands on the host. snr_aper still drops as expected
   (host is in the "background" estimate so it subtracts off the SN), but
   in a way that flags the band — DON'T fix the snr by widening the ring,
   reject the band via the morphology gates instead.
4. **Eligibility check that only enforces G1 (sep_sn).** This is the
   first thing that breaks. `eligible = (sep_sn <= tol)` ALONE will
   admit hosts in red broad-PSF bands, even though their sharp/rnd1/rnd2
   are clearly out of range. Eligibility MUST require G1 ∧ G2 ∧ G3 ∧ G4.
   (Bug found 2026-05-27 on the 6 new JWST SN in v31; they were all
   misidentified as F444W picks at sep_host≈0 with sharp 0.29-0.41.)
5. **Picker that only looks at `cands[0]`** (the DAO candidate CLOSEST
   to the catalog position). For an SN offset from the host, the
   closest DAO is the host itself; the SN is `cands[1]` or further.
   The picker must loop through all candidates in the search radius,
   filter by G2-G4, and prefer the closest PSF-clean candidate that is
   OFFSET from the catalog position (sep > ~0.10″). If only host-position
   candidates pass, that's the SN-on-host case.
6. **`Cutout2D(..., mode="partial", fill_value=np.nan)` can silently
   suppress DAO peaks** that the raw memmap slice picks up.  Observed
   on 19931 F115W (2026-05-27): raw 6″ slice → 67 peaks; same box via
   Cutout2D → 0 peaks.  Root cause not pinned down (likely NaN handling
   in DAOStarFinder when the sub-array has NaNs at the boundary), but
   the workaround is: for CATALOG-MODE DAO scanning, slice the raw
   `sci_hdu.data[y0:y1, x0:x1].astype(np.float64)` like
   `recut_3_hst.aper_photometry` and `find_v31_positions.measure_band`
   already do.  Use Cutout2D only for the RGB rendering step, never as
   the input to DAOStarFinder.

### 2.6 Validation roadmap (before mass production)

Before deploying these cuts on 1000+ candidates, the user has asked for:

  1. **Star control sample** (~50 unsaturated stars, mag 19–21, across
     same tiles): should pass G1-G4 in every band they're detected in.
     If any star fails, threshold is too tight.
  2. **Host-galaxy control sample** (~50 galaxies at same magnitudes,
     away from any SN): should mostly fail G2 with `sharp > 0.85`. Tunes
     the upper sharp bound.
  3. **Random-position control sample** (~50 catalog-blank points in
     same tiles): should fail G1. Confirms G1 isn't drifting onto noise.
  4. **Position jitter test**: perturb each SN by ±0.05″, re-run, check
     same band is still picked. Knife-edge picks indicate fragile cuts.
  5. **Threshold sensitivity**: vary each threshold by ±10% and count
     band-flips across the 11 SN. Stable thresholds flip few; fragile
     thresholds flip many.

----

## 3. The 11 known SN reference set

**Canonical (permanent) location** for the position lookup and the
photometry table:

```
/Users/suzuki/github/projects_cosmos/csvfiles_sn/
    lookup_sn11_v30.csv   ← position lookup (9 cols: id, telescope,
                            sn_ra/dec, host_ra/dec, hst/jwst/euclid tile)
    tbl_sn11_v30.csv      ← photometry table (35 cols: best_band/snr from
                            morphology criteria, per-band mag + snr, etc.)
```

Both are pinned to the v30 webpage version (the morphology-validated
release).  Working `/tmp/sn_lookup_13.csv` and `/tmp/sn_11_v30.csv` were
the scratch copies during development; `/tmp` can disappear, so always
read from `csvfiles_sn/` for any new script.

When future versions need a new SN snapshot, copy the working files into
`csvfiles_sn/lookup_sn11_v<NN>.csv` and `tbl_sn11_v<NN>.csv` to keep the
mapping `csv ↔ webpage version` 1:1.

Schema of `lookup_sn11_v30.csv`:
`id, telescope, sn_ra, sn_dec, host_ra, host_dec, hst, jwst, euclid`.

| ID     | tel    | best band | sep_host | sn_ra        | sn_dec      | hst | jwst | euclid     |
|--------|--------|-----------|----------|--------------|-------------|-----|------|------------|
| 130972 | HST    | F814W     | 1.04″    | 150.279695   | 2.041101    | 052 | A4   | 101542818  |
| 371996 | HST    | F814W     | 1.11″    | 150.282844   | 1.932259    | 040 | A10  | 101542818  |
| 471959 | HST    | F814W     | 0.52″    | 150.278904   | 2.430962    | 076 | B3   | 101545698  |
| 63924  | EUCLID | NIR-Y     | 1.87″    | 149.968573   | 2.233455    | 066 | A2   | 101544256  |
| 318858 | JWST   | F115W     | 0.75″    | 150.185337   | 1.842646    | 041 | A9   | 101541377  |
| 320233 | JWST   | F150W     | 0.00″    | 150.146358   | 1.866410    | 041 | A9   | 101541377  |
| 246188 | JWST   | F115W     | 0.34″    | 149.908420   | 1.994614    | 054 | A7   | 101542816  |
| 296673 | JWST   | F150W     | 1.07″    | 150.103129   | 2.006954    | 053 | A3   | 101542817  |
| 435613 | JWST   | F115W     | 0.26″    | 150.073048   | 2.530626    | 089 | B2   | 101545697  |
| 435769 | JWST   | F115W     | 0.26″    | 150.108823   | 2.518886    | 089 | B2   | 101545697  |
| 468896 | JWST   | F150W     | 0.42″    | 150.264529   | 2.414231    | 076 | B3   | 101545698  |

These positions are user-confirmed via iterative DAO-circle review
(v22 → v23 → v24 → v26 → v29). Updates require a fresh DAO scan at
the new hint and a user OK on the resulting PNG.

----

## 4. Architecture & key files

```
programs_webpage/
  recut_3_hst.py          base library: FITS open + WCS cache + path
                          helpers + aperture phot + cutout rendering.
                          ALL other scripts import from this.
  04_polish_labels.py     adds 4-corner labels + partial crosshair on
                          top of raw cutouts. Reads photometry CSV.
  make_v24_dao_check.py   per-source DAO-circle PNGs (single band)
  make_v26_dao_check.py   same, optimised (PatchCollection, single scan)
  make_v27_multiband.py   multi-band scan, best-band picker (BUGGY:
                          used peak/bg_std, fooled by host)
  make_v29.py             multi-band scan, BUG-FIXED picker. PRIMARY
                          script for the current criteria.
  CLAUDE.md               ←this file

htmls/_crosshair_test/
  00README                history of v01–v29
  index.html              version listing
  v##/                    one directory per iteration; NEVER overwrite
                          - DAO-check decks: sn_<id>_DAO_check.png
                          - polished decks:  polished_<id>_<panel>.png
                          - dao_params_*.csv  (multi-band tables)
```

### Caches & efficiency notes

`recut_3_hst.py` keeps two module-level caches:

  * `_FITS_CACHE`  — `path → open HDUList`, kept open for the script
    lifetime. Subsequent `_open_cached(p)` calls return the same handle.
  * `_WCS_CACHE`   — `path → WCS`, parses the header once.

Two important efficiency tricks already implemented; don't break them:

  1. **Slice from memmap before astype.** `sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)` not the reverse — otherwise astropy materialises the whole tile (500MB-30GB) instead of the 80KB box.
  2. **No `--measure` re-run if photometry already written.** The CSV
     `/tmp/sn_11_v##.csv` is reused by the polish step.

### Per-band PSF FWHM (empirical, 2026-05-26)

| band   | FWHM (arcsec) | pixel scale | notes                      |
|--------|---------------|-------------|----------------------------|
| F814W  | 0.134         | 30 mas      | HST ACS                    |
| F115W  | 0.057         | 30 mas      | JWST NIRCam, sharpest      |
| F150W  | 0.057         | 30 mas      | JWST NIRCam                |
| F277W  | 0.130         | 30 mas      | JWST NIRCam, broader       |
| F444W  | 0.160         | 30 mas      | JWST NIRCam, broadest      |
| VIS    | 0.194         | 100 mas     | Euclid; different epoch    |
| NIR-Y  | 0.524         | 100 mas     | Euclid NISP                |
| NIR-J  | 0.537         | 100 mas     | Euclid NISP                |
| NIR-H  | 0.567         | 100 mas     | Euclid NISP                |

### Display stretch settings — polished gallery (`recut_single_source.py`)

User-tuned 2026-05-28 across the v05 gallery. **These are the canonical
settings — do not change without explicit user OK.** Same `render_gray` and
`render_rgb_*` are used by every gallery (v03, v04, v05, and any future
versions), so the stretch is consistent across the project.

#### HST F814W and Euclid VIS — `render_gray()`

```python
minimum = 0.0001
bw      = sqrt( clip(data, minimum, ∞) )
# Auto-stretch vmax: 1/3 old recipe + 2/3 new recipe (user-blended)
vmax_old = max(1.0,  central_peak_sqrt * 0.85)   # original — too dim for faint sources
vmax_new = max(0.05, central_peak_sqrt * 0.85)   # full auto-scale — too bright
maximum  = (vmax_old + 2 * vmax_new) / 3
imshow(bw, vmin=minimum, vmax=maximum)
```

`central_peak_sqrt` is the max value of the central 16×16-pixel window of
`bw` (the sqrt-stretched array). Rationale: HST/VIS in e⁻/s are far smaller
in magnitude than JWST in MJy/sr, so the original `max(1.0, …)` floor pinned
vmax at 1.0 for any faint source and washed them out. The blend keeps bright
sources unchanged (where val·0.85 ≥ 1.0 the two formulas collapse to the same
value) while lifting the faint-source brightness.

#### JWST RGB — `render_rgb_jwst()` (NOT changed)

```python
minimum = 1e-4
b,g,r   = sqrt(clip(data, minimum, ∞))
maximum = max(1.0, max(central_peak_sqrt over b,g,r) * 0.85)
```

Reference recipe — works as-is because JWST pixel values in MJy/sr give
typical central-peak sqrt values ~0.3-0.7, safely under the saturation
threshold 1/0.85 = 1.18.

#### Euclid NISP RGB — `render_rgb_nisp()`

```python
minimum = 1e-4
# Per-channel background subtraction (NISP-specific, see rationale below)
b_data -= median(b_data)
g_data -= median(g_data)
r_data -= median(r_data)
b,g,r   = sqrt(clip(data, minimum, ∞))
# vmax floor = 7.0 (NISP-specific, user-tuned final value)
maximum = max(7.0, max(central_peak_sqrt over b,g,r) * 0.85)
```

Two NISP-specific adjustments vs. JWST:

1. **Per-channel background subtraction.** NISP sky background in
   sqrt-stretched space is **10-50× higher** than HST/JWST/VIS (empirical;
   from top-5 candidates, NISP bg ~0.3-1.5 sqrt vs HST/JWST/VIS bg
   ~0.01-0.12 sqrt). Cause: NISP's 300mas pixels collect ~100× more sky
   photons per pixel than 30mas instruments, plus the IR sky itself is
   brighter than the optical sky. Without bg-sub, the elevated pedestal
   filled 40-100 % of the display range and washed the image out. After
   bg-sub the residual data behaves like JWST naturally does (bg ≈ 0,
   source above).

2. **vmax floor = 7.0** (not 1.0 like JWST). Even after bg-sub, NISP
   central-source peak_sqrt is ~1.0-2.0 (2-3× JWST's ~0.3-0.7), because
   each NISP pixel still integrates more source photons. With floor=1.0
   many bright NISP sources saturated; the iteration history was
   `1.0 → 2.5 → 4.0 → 6.0 → 7.0` until the user signed off ("perfect").
   At floor=7.0, typical NISP source (peak_sqrt~1.5) displays at 21 % grey,
   and saturation only kicks in at peak_sqrt > 8.2 (very rare).

#### Pagination — `paginate_gallery.py`

A single HTML page with 200 candidates × 5 panels ≈ 155 MB of images
crushes the browser (renders top-down, appears to stop loading after a
few). `paginate_gallery.py --version <vNN> --per-page 25 --top 200`
splits into 8 pages of 25 candidates each (~19 MB/page, lazy-loaded
images), auto-detects `polished_top200/` or `polished_top100/`, adds a
page nav bar at top + bottom, and adds an honest red/green status banner
per version.

----

### v04 / v05 consolidator patches (locked in 2026-05-28 after user's top-25 review)

Both `consolidate_v05.py` (CNN-based 1-of-3 telescope rule) and
`consolidate_v04.py` (HST-template difference imaging) share the same
11-patch filter. Locked in after the user reviewed the top 25 of v05 and
found **0/25 were real SNe** — every one was a known failure mode below.
After the patches, v05 went 38,993 → 79 candidates, v04 went 424,577 → 712.

Constants in both consolidators:
```python
MAG_FAINT_HST       = 27.8   # HST F814W 3σ depth (5σ = 27.2 + 0.55 mag)
HST_VIS_SNR_VETO    = 3.0   # aperture-SNR threshold for HST/VIS visibility
NISP_SNR_VETO       = 3.0   # wide-aperture (1.0 × FWHM) NISP threshold
NISP_TIGHT_APER_MULT= 0.3   # smaller aperture for blending-corrected NISP re-measure
NISP_TIGHT_SNR_VETO = 3.0
HST_PM_SEARCH_RADIUS_AS    = 5.0   # HST DAO neighbor search radius
HST_PM_NEIGHBOR_SNR_MIN    = 5.0   # require HST DAO snr above this to flag
SHARP_HI = 0.75              # tightened from 0.85; rejects extended galaxies
```

11 patches in order of application per candidate:

| # | Patch | Rejects |
|---|---|---|
| 1 | saturation (any HST/JWST/VIS band mag < 21 with snr ≥ 3) | bright stars |
| 2 | morphology G2 sharp ∈ [0.40, 0.75] | extended galaxies |
| 3 | cross-band consistency (multi-band telescopes) | single-band flares |
| 4 | best-band aperture snr ≥ 5 | marginal |
| 5 | best-band mag ≥ MAG_FLOOR (21.0) | bright |
| 6 | best-band mag ≤ MAG_FAINT_HST (27.8) — HST 3σ cap | unverifiable faint |
| 7 | **HST visible** (snr_F814W ≥ 3) — persistent in old epoch | host galaxies, slow stars |
| 8 | **Euclid-VIS visible** (snr_VIS ≥ 3) | persistent in Euclid optical |
| 9 | **NISP wide veto** (≥ 2 of 3 NISP bands at snr ≥ 3, 1.0 × FWHM aperture) | high-z dropouts, IR-bright sources |
| 10 | **NISP tight veto** (≥ 2 of 3 NISP bands at snr ≥ 3, 0.3 × FWHM aperture) | NISP detections caught after blending correction |
| 11 | **HST DAO neighbor within 5″** (snr ≥ 5, not the candidate itself) | **high-PM stars** that decouple HST and JWST positions |

Each consolidator also writes new diagnostic columns for visual review:
`mag_Y_tight, snr_Y_tight, …, mag_H_tight, snr_H_tight, hst_pm_neighbor_sep,
hst_pm_neighbor_pid, hst_pm_neighbor_snr`.

#### Why patches 9-11 exist (failure modes the basic vetoes miss)

  * **High-z dropout galaxies** (#4, #5, #8, #15 in user's v05 review).
    Bright in NISP IR (Y/J/H mag ~20), faint/missing in HST + VIS.
    Visually identical to a SN candidate from HST/VIS-veto alone.
    Patch 9 + 10 catch them.

  * **NISP blending** (#6, #7, #2 in user's v05 review).
    NISP 300-mas pixels + 0.5″ aperture sum host + nearby flux at the
    candidate position. Wide-aperture mag reports the NEIGHBOR brightness
    (~20 mag), not the SN. Tight 0.3 × FWHM aperture re-measures at the
    candidate position only. For the user's #6 (671543): wide mag = 20.65
    snr 5.2 → tight 0.2 × FWHM mag = 23.66 snr 2.1 (no real source).

  * **High-PM stars** (#1, #16, #22 in user's v05 review).
    Stars with proper motion > 0.5″ over the HST→JWST baseline (~17 yr)
    appear as TWO separate catalog entries (HST entry at old position,
    JWST entry at new position; matcher tolerance ~ 0.3-1.0″ doesn't
    link them). The HST persistence veto (patch 7) measures HST flux AT
    the JWST position — sees nothing because the source moved. Patch 11
    searches the HST DAO catalog within 5″ of each candidate; if any
    UNMATCHED HST DAO entry is found at 0.5″ < sep < 5″, reject.
    User example #1 (270470): HST DAO at 2.5″ → PM ≈ 145 mas/yr.

The proper fix for #11 is upstream (`programs_star/52_step3_catalog_match_v04.py`):
do a second matching pass at 5″ for unmatched sources and set
`is_likely_star=True` with a new `high_pm_inferred=True` column. Until
that runs, the consolidator-side check is the workaround.

----

## 5. Common pitfalls (project-specific, in addition to §2.5)

1. **VIS != NISP.** Skip VIS in any multi-band SN logic — VIS is a
   different epoch, not contemporaneous with NISP YJH.

2. **LEPHARE z lookup misses.** Some catalog IDs return z=0.0000 or
   z=-99.0000 from the lephare table. Pre-existing data gap, not a bug.
   Just print the value; the user knows it.

3. **63924 host Y/J/H mags are nan.** The Euclid NISP host ring lands
   on a masked pixel for this source. Pre-existing data gap.

3a. **Euclid DR1 superseded files.** As of 2026-05-27 the user moved
   the May-2025 superseded MER mosaics into
   `/Volumes/exdisk1/data/Euclid/COSMOS_DR1/superseded_may2025/`.
   The top-level directory now holds exactly one file per (tile, band)
   from the August-2025 release.  `resolve_euclid_path()` globs only
   top-level (no recursion), so it picks the August version
   automatically.  Don't fall back to `superseded_may2025/` unless
   doing a deliberate comparison.

4. **JWST cutouts must be rotated North-up.** `cut(path, pos,
   north_up_jwst=True)` for any JWST FITS that goes into a polished
   image. Without this the JWST orientation is the spacecraft roll
   angle, which conflicts with the standard N=up E=left convention
   used in HST/Euclid panels.

5. **`--no-crosshair` for polished output.** The polish step (04_)
   draws its own crosshair. The recut step must NOT draw one on top.

6. **Re-using `_FITS_CACHE` across scripts.** `import recut_3_hst as R`
   gives you the SAME cache dict the imported script uses. So loading
   a tile in measurement step 1 keeps it open for step 2. Don't
   `del _FITS_CACHE` mid-script.

----

## 6. Open items, scoped for follow-up

* Validate the §2 cuts against the star / galaxy / random control
  samples (§2.6). Until that's done, the cuts are tested only on the
  11 SN.
* Mass-production architecture for 1000+ SN candidates: scaling the
  per-source DAO+aperture+morphology evaluation, batch IO, parallel
  workers — currently single-process (~2 s for 11 SN, would be ~30 min
  for 10k SN on warm cache, longer cold).
* CNN-based morphological classifier mentioned as long-term goal —
  current PSF+multi-band approach is the interim hand-engineered
  baseline.

----

## 7. v06 — generative/no-subtraction CNN + the MASTER SN list (2026-06-01)

### 7.0 What the user decided (binding)
* SN detection rule = **exactly ONE telescope detects** (HST *or* JWST *or*
  Euclid). Two+ telescopes ⇒ persistent (galaxy/star) ⇒ NOT a SN.
  Euclid = VIS **AND** NISP (simultaneous epoch). NISP-only ⇒ NISP detector
  persistence (reject). VIS-only ⇒ artifact (reject). JWST+NISP ⇒ high-z
  dropout GALAXY (reject), NOT a SN.
* High-z SN: only the discovery telescope sees it; at our depth the others
  are too shallow, so "one telescope only" holds even for high-z.
* Goal: rediscover ≥ 80% of the validated SNe before producing a v06 list.
* Prefer NO stored cutouts for the future (millions of images). For first
  trials, caching cutouts on the external disk (/Volumes/My Book) is allowed.
* The previous-gen CNN in `/Users/suzuki/github/projects_jwst/programs_cosmos/`
  (`train_supernova_cnn.py`) performed BETTER than v04/v05 — study it.

### 7.1 Guiding papers
* **TransiNet** (Sedaghat & Mahabal 2018, MNRAS 476, 5365; arXiv:1710.01422):
  generative encoder-decoder that OUTPUTS the difference image (no explicit
  subtraction); learns registration+PSF-match+sky+noise internally. Key
  tricks: L1 loss, "attention trick" (remap target [0,1]→[0,100] so the tiny
  transient isn't ignored), synthetic training with per-epoch PSF/sky/noise +
  registration jitter. 98.4% precision / 75.5% recall on CRTS.
* **Inada, Sako, Acero-Cuellar & Bianco 2026** (AJ, 10.3847/1538-3881/ae38d8):
  transient detection **WITHOUT image subtraction** — feed search+template
  pair directly; transformer w/ localized attention. 97.4% (no-diff) vs 97.8%
  (with-diff); the gap NARROWS with more data. Still uses 51×51 stamps.
* Synthesis adopted for v06: feed the co-aligned epoch pair directly (no
  subtraction), HST = template, JWST = science; both 30 mas so alignment is
  just co-centering.

### 7.2 v06 training design (`programs_webpage/train_v06.py`)
* 5-channel σ-units input [HST_F814W, F115W, F150W, F277W, F444W], later a
  10-channel **dual representation** (ch0-4 σ-units asinh = cross-epoch
  comparable; ch5-9 per-channel peak-norm = within-band compactness).
* Positives = profile-matched Gaussian injection (HST-only for HST SNe; all
  4 JWST bands consistent for JWST SNe). Negatives = SAME host w/o injection
  (paired) + asteroid (1-band) + same-color (F277+F444 only).
* Cutouts cached at `/Volumes/My Book/data/cosmos_v06/_v06_cutouts_sigma.npz`
  (σ-units). Training pre-generates samples in RAM, trains on MPS.
* Validation = **leave-one-out** on the real SNe (train on synthetic + the
  other real SNe augmented, score the held-out one). HONEST out-of-sample.

### 7.3 KEY LESSONS from the v06 iterations (do not relearn these)
1. **Pure-synthetic training does NOT transfer** (3/17). Must MIX real SNe
   (heavily augmented) with synthetic — both papers do this.
2. **Per-channel [0,1] normalization is a BUG for cross-epoch nets**: it makes
   a blank HST channel and a SN-bearing JWST channel both peak at 1.0, ERASING
   the brightness difference that IS the transient. Use **σ-units**
   (pixel − sky_median)/sky_MAD so channels are physically comparable.
3. **Few-SN models are high-variance**: per-SN scores swung 1.0↔0.0001 between
   configs. Fix = **ensemble** (train K=7 models, average sigmoid).
4. **σ-units (cross-epoch) and compactness (per-band) are COMPLEMENTARY**:
   σ-units recovers HST + off-host; compactness recovers on-host JWST. The
   10-channel dual-rep + ensemble was best: **8/17 @P≥0.5, 12/17 @P≥0.3**.
5. The remaining wall = faint **on-host, cross-band JWST** SNe + too few
   positives (17). It is DATA-LIMITED, not method-limited.
6. The previous-gen 34-SN list is **NOT all real** — user said do not use it.
   Mosaics live on EXTERNAL disk (/Volumes/exdisk1); ≤3 parallel FITS readers
   or it stalls (0% CPU). MPS GPU work doesn't show in Activity Monitor CPU%.

### 7.4 THE MASTER SUPERNOVA LIST (the unblock — keep growing this)
The fix for the data wall was to expand the validated positive set.
* Source catalog: `projects_jwst/programs_cosmos/data/sn_ori.txt` = 65-SN
  DeCoursey COSMOS-Web catalog (name, RA/Dec, z 0.01-3.55, discovery filter,
  Ia/cc class). 55/65 fall in our full-coverage footprint.
* Inspection gallery built at `htmls/sn_search/catalog65_inspect/` → user
  visually validated **35 of 65** as real in COSMOS-Web.
* The other 30 are `primer_or_offfield` — **NOT rejected**; likely real but on
  PRIMER / other data, not visible in our COSMOS-Web FITS. Re-check later.
* **MASTER LIST = 52 validated real SNe** = 17 user-known + 35 validated.
  Breakdown: 48 JWST / 3 HST / 1 EUCLID, all FITS-covered.

Files (canonical):
```
csvfiles_sn/master_sn_verdicts.csv  — every candidate + status
      (real | primer_or_offfield | pending); EDIT status here
csvfiles_sn/master_sn_list.csv      — compiled status==real rows (52)
csvfiles_sn/master_sn_gallery.csv   — render-ready (known17 first)
programs_webpage/build_master_sn.py — seed (no args) / --compile
programs_webpage/make_master_sn_web.py — render master gallery
htmls/sn_search/master_sn/index.html   — MASTER SN page (52, static)
```
Workflow to GROW the list: add/mark `status=real` in master_sn_verdicts.csv →
`python build_master_sn.py --compile` → re-run make_master_sn_web.py →
retrain v06 on the larger master_sn_list.csv.

### 7.5 Next step (pending)
Retrain v06 on the 52-SN master list (48 JWST positives vs the old 13) — this
directly attacks the data-starvation wall that capped LOO recovery at 8/17.
