# HANDOFF — Step 3a state & next-session plan (2026-06-10)

## ⚡⚡ STATUS 2026-06-10 evening — Step 3a-② detection REWRITTEN, A4-validated, mass run PENDING

**51_step3a_chi2_detect.py is now mission-generic** (jwst / hst / euclid;
empirical FWHMs from the 3a-① metas; real KRON_RADIUS in the merge via new
`configs/chi2_detect.param`; configs renamed `chi2_cold.sex`/`chi2_hot.sex`).
Driver: `65_run_chi2_detection.py` (resumable, timing CSV
`csv_timing/chi2_timing.csv`).

**Two validated recipe changes (evidence in 51_ docstring + this session):**
1. **Detection image is χ₊ = sqrt(Σ/N), NOT the squared χ²₊** of ms.tex
   eq. (chi2plus).  On the squared image SExtractor's background-RMS mesh
   is unstable: a 15% kernel change (pilot guessed FWHMs → empirical)
   swung A4 cold counts 10,913 → 2,163 on pixel-identical bright sources
   (RMS-map p99 107 → 242).  The sqrt form = what SWarp CHI2 (Szalay+99)
   actually produces, i.e. what Shuntov/Galametz really ran on; same
   kernel change then moves counts only 2.5%.  **ms.tex §B.1 eq. needs
   the sqrt added** (not yet edited).
2. **Hot pass at the locked 3.0σ/MINAREA 8 is spurious-dominated** on our
   homogenization-correlated noise: negative-image test on A4 found MORE
   detections on pure noise (2,291) than on sky (1,649); cold is pure (0).
   Threshold sweep → JWST hot_thresh=5.0 (5.6% spurious).  Per-mission
   `hot_thresh` lives in 51_'s MISSIONS dict; HST/Euclid = None (3.0)
   UNTIL calibrated from their first tile's negqa.  Negative-image QA is
   built into 51_ (recipe_version=2): per tile, central 8192² window,
   pos/neg counts + spurious_frac in the chi2 meta.

**A4 state on disk:** ran with sqrt form recipe v1 (pre-negqa, hot 3.0σ):
cold 42,446 / hot kept 236,910 / merged 279,356.  Will auto-rebuild under
recipe v2 (meta mismatch) on the next run.  Old pilot outputs moved to
`jwst_chi2/pilot_legacy/`; `jwst_chi2/A4full/` kept for the squared-form
comparison record.

**Resume worklist (in order):**
1. `python -m py_compile 51_… 65_…` (compile check not yet run after the
   negqa edits), then `51_ --mission jwst --tile A4 --force` → confirm
   negqa numbers ≈ the manual cutout test (cold pure; hot ~5% spurious).
2. One HST tile + one Euclid tile → read negqa from chi2 meta → set
   hot_thresh for hst/euclid in MISSIONS → rerun those tiles --force.
3. Mass run: `65_run_chi2_detection.py` (jwst 2-way, hst 3-way, euclid
   4-way on the HDD).  ~100 tiles total.
4. Then 3a-③ (53_ rewrite per the agreed dual SEx+DAO design below) —
   ORDER agreed with user: detection → saturation flags (flag, never
   drop) → SEx+PSFEx PSF photometry ALL sources (dual-image, 3a-① .psf
   models) → DAO sharp/rnd1/rnd2 ALL sources → DAO PSFPhotometry on the
   point-source pool (same PSFEx model via GriddedPSFModel) → 3b.

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
