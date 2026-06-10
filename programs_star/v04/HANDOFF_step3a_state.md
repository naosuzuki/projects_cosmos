# HANDOFF — Step 3a state & next-session plan (2026-06-10)

## ⚡ STATUS 2026-06-10 16:30 — Step 3a-① PSF models 100% COMPLETE

**The LS rerun described below is DONE** (368/368, all `global_pooled`,
onsets g 17.12 / r 16.75 / i 17.65 / z 15.94, tilt on 368/368, commit
8a869b37).  The complete Step 3a-① inventory now stands at:
JWST×4 (80) + HST (20) + Euclid VIS (60) + NISP×3 (60) + HSC×5 (405) +
LS DR10×4 (368) — every tile with a .psf model, meta JSON, QA plots on the
web (`html/psf_qa/`, server `cd html && python -m http.server 8000`).

**A NEW SESSION should start directly at the "Next session worklist" section
below: 3a-② χ²₊ hot+cold detection (51_) → 3a-③ dual SEx+DAO photometry
(53_) → Step 3b.**  Division of labour: the OLD session continues data
collection (HSC catalog 41_, GTC, etc.) — it owns download scripts/logs;
the NEW session owns the v04 step-3 code commits (git pull before commit).

## Historical resume point (completed; kept for the record)

**LS DR10 saturation was redesigned after the user caught bad g-band onsets
(recorded 19.7–24 where the plots show ~17).**  Final method (user-validated
on brick 1493p017 g): GLOBAL per-band onset from `64_step3a_lsdr10_global_
saturation.py` — pooled Gaia-anchored peak turnover on the **Gaia G axis**
(saturated stars slide along MAG_AUTO-based relations, hiding the break;
G is an external truth axis) + band−G color + p95 MAG_AUTO slide margin.
Values in `programs_star/csv_saturation/lsdr10_global_onsets.json`:
**g 17.12, r 16.75, i 17.65, z 15.94** (g matches the user's eyeball ~17).
The builder (54_lsdr10) reads the JSON automatically (`global_pooled`).

**On-disk LS models are a PATCHWORK of debugging generations** (145 old
percentile / 140 bleed-contaminated maskbits / 58 none / 24 intermediates /
1 final).  **FIRST ACTION: full force rerun:**
```
python 60_run_mass_production.py --missions lsdr10 --force
# + live web refresher loop (55_make_psf_qa_site.py every 120 s)
```
then the QA sweep (tilt on?, MADt<0.15?, n_model_stars, onsets uniform),
user spot-check of LS-g saturation plots on the web, commit timing CSV.
A corrupt half-written pass1 (killed run) was the `Empty or corrupt FITS`
crash — 64_/sweeps now skip/remove unreadable pass1 files.

Background context for the restart: the gh-CLI banner glitch = stale app
process state (token itself verified working); restart fixes it.

For the next session: read this + ms.tex §B.1 (locked Step 3a/3b/4/5 plan,
`~/github/papers/26_jwsteuclidhst_note/ms.tex` lines ~1320–1670).

## Where Step 3a stands

### 3a-① Empirical PSF models — COMPLETE for all imaging missions
| mission | tiles × bands | models | recipe notes |
|---|---|---|---|
| JWST NIRCam | 20 × (F115W,F150W,F277W,F444W) | 80 | Tanaka+2023, spikes kept |
| HST ACS | 20 × F814W | 20 | peak-plateau saturation |
| Euclid VIS | 60 × VIS | 60 | FWHM-floor fix |
| Euclid NISP | 20 × (Y,J,H) | 60 | two-pass bright-anchor |
| HSC s23b | 81 × grizy | 405 | TILTED locus (Gaia-fit), MASK-bit sat |
| LS DR10 | 92 × griz | 368 | see LS lessons below |

All under `/Volumes/exdisk1/data/photometry_v04/<inst>/<tile>/psf/`
(`stars_<tile>.psf` + `psf_<tile>.meta.json` + 6 QA PNGs).
QA site: `html/psf_qa/` (server: `cd html && python -m http.server 8000`).
Driver: `60_run_mass_production.py --missions jwst,hst,euclid,euclid_nisp,hsc,lsdr10`
(resumable; per-tile timing → `programs_star/csv_timing/mass_timing.csv`).

### Ground-recipe extensions beyond the locked plan (validated HSC + LS)
1. **Tilted stellar locus** — FR = slope·mag + icpt fit (HSC: point sources;
   LS: Gaia-only), parallel artifact/upper rails at ∓3·MAD.
2. **Measured faint purity limit** (LS) — magnitude where in-band point-source
   median FR departs the Gaia line by >2·MAD (compact-galaxy takeover).
3. **Saturation**: HSC = MASK SAT bit; LS = maskbits SATUR bit ONLY
   (ALLMASK blanket-flags low-nexp edge bricks) + onset sanity clamp
   (onset > point-source median ⇒ ignore) + locus-departure bright fallback.
4. **Gaia anchor threshold ≥8** (a tight locus from few pure Gaia stars beats
   hundreds of contaminated CLASS_STAR objects) + MAD floor 0.04 px.
5. **WIDE Gaia CSV** `catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv`
   (37,235 rows, RA 148.5–151.65, Dec 0.75–3.8, no VIS-polygon trim) — the
   original VIS-trimmed CSV starved LS rim bricks of anchors (THE root cause
   of the first-pass quality flags). 54_lsdr10 + 56_single prefer it.

### Known open items
- HSC ZP=27.0 is PROVISIONAL — pin empirically (DAO growth-curve vs Gaia/PS1).
- A handful of LS rim band-bricks are genuinely shallow (n_model_stars<40,
  recorded in meta) — usable-with-care, flagged via meta, not deleted.
- gh CLI banner: stale app process state; user restarts app after this session.

## Next session worklist (in order)

### 1. Finish 3a-② hot+cold detection (`51_`, task #82)
χ²₊ stack per mission (eq. in ms.tex §B.1), cold pass + hot pass outside cold
Kron ellipses, DETECT_MODE column. Space missions first (JWST/HST/VIS).

### 2. 3a-③ DUAL photometry (`53_`, task #83) — design agreed with user
- **SExtractor+PSFEx PSF photometry on ALL sources** (per band, native
  imaging, `-PSF_NAME stars_<tile>.psf`) → FLUX_PSF/MAG_PSF/CHI2_PSF/
  SPREAD_MODEL. Param file needs those output columns.
- **DAO morphology on ALL sources**: DAOStarFinder sharp/rnd1/rnd2 (cheap,
  detection-stage; required by 3b classifier).
- **DAO PSF-fitting photometry on the POINT-SOURCE POOL only** (photutils
  `PSFPhotometry` with grouping = ALLSTAR-style simultaneous fits; feed the
  SAME PSFEx model via GriddedPSFModel so SEx−DAO differences isolate the
  fitting method) → `psf_phot_consistent` flag.
- **Pool = union (high recall, precision is 3b's job):**
  (a) Gaia wide-CSV matches; (b) tilted-locus band ±5·MAD + SPREAD_MODEL;
  (c) DAOStarFinder sharp/round vetoes. Audit loop: 3b stars not in pool ⇒
  widen + refit (resumable).

### 3. Step 3b — Gaia-PM-anchored classifier (ms.tex §B.1 verbatim)
Training: Gaia PM ≥5σ AND ≥5 mas/yr, 14<G<18. Per-band per-mag 95th-percentile
envelopes over 4 stats (CHI2_PSF, sharp, rnd1, rnd2) →
`is_star_psf_<filter>`; aggregate = Gaia-PM OR ≥3 unsaturated-band votes.
No QSO logic (deferred to Step 6+).
