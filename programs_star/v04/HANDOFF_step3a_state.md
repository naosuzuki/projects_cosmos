# HANDOFF — Step 3a state & next-session plan (2026-06-10)

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
