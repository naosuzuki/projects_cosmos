# HANDOFF — Step 3a state & next-session plan (2026-06-10)

## ✅ 2026-07-06 18:35 — PSF v2 CAMPAIGN COMPLETE (all 12 surveys)

Every survey now carries the hostgalxy-consistent v2 record:
- HST F814W 20/20 + JWST 4 bands x 20 tiles 80/80 REBUILT with the
  10xFWHM space isolation + 7-bit stamp gate (psf_gate.py; JWST adds
  nbr_dmag=6, blend core-exclusion, bright/spike gate-2 exemption).
  JWST model stars: SW ~200-266/tile, LW ~75-83/tile, spike stars kept.
  ⚠ stars_<suffix>.psf + metas CHANGED for ALL HST+JWST tiles →
  3a-③ photometry for HST and JWST must be RERUN.
- JWST pass1 is now VIGNET-less (55 MB vs 30 GB; the giant row-major
  catalogs wedged 6 builders for 3 h at 0.1% CPU + filled the disk on
  the first overnight attempt).  stars LDAC VIGNETs are built in-memory
  (54_ write_stars_ldac) — raw stamps, spikes can never be deblend-
  blanked; companion contamination handled by the gate.
- Euclid VIS/NISP + HSC psf_qa pages now IMPORT the hostgalxy record
  (qa_psf_cosmos site / ~/data/psf_v01; VIS 30, NISP-Y 30, J/H 28,
  HSC 10 COSMOS tracts x grizy).  v04 Euclid/HSC model files on exdisk1
  are UNTOUCHED (photometry chain unaffected) — but note the newer
  hostgalxy Euclid/HSC models exist if 3a-③ wants them.
- unWISE/PS1/LS/SDSS fully rebuilt (60_ --force) with the raw-cutout QA.
- Site: all rows new-style (gate mosaics w/ red reject frames, toggle
  viewers, neighbour Δmag-vs-sep).  Deploy repo re-exported.


## ⚠️ 2026-07-05 — HST F814W PSF MODELS BEING REBUILT (v2 gate) — 3a-③ HST photometry should wait/rerun

User-approved v2 recipe (commit e55fe2d2): hostgalxy-consistent star
selection — SPACE isolation = 10 x PSF_FWHM (~1"), 7-bit stamp-quality
gate (edge/nbr/bad-pix/blend/sat/asym/outer), rejects persisted to
stars_rejected_<tile>.fits.  Model star counts drop ~290-360 → ~120-150
per tile; stars_<tile>.psf and metas CHANGE for all 20 HST tiles.
A1-A3+B5 done; remaining 16 running detached (log /tmp/hst_v2_rebuild.log,
ETA ~4 h from 23:50).  Any 3a-③ photometry already run with v1 HST PSFs
should be redone after this lands.  JWST + other space telescopes get the
same 10xFWHM treatment next (user directive), then ground surveys'
mosaics regenerate with the new raw-cutout QA style.


## ⚡⚡⚡⚡⚡ STATUS 2026-06-11 evening shutdown — mass production PAUSED mid-Euclid; plot-revision round pending

The 69_ pass-1 driver and its auto-followup were STOPPED (user) before
completion.  State on disk (all resumable):
- HST 19+1/20 COMPLETE (53+67+66); JWST: only A4 complete; Euclid:
  ~18 tiles complete, 23 pre-borrow-fix failures (will auto-retry),
  rest not started.  Diag plots (70_) exist only for jwst/A4.
- TO RESUME EVERYTHING: `cd programs_star/v04 && /opt/miniconda3/bin/python
  69_run_photometry.py --missions hst,euclid,jwst` then
  `71_make_diag_site.py`.  Resumable per stage; failed tiles retried
  automatically (53_ now borrows nearest-tile PSF/ZP for sparse NISP).
- NEXT (user-driven): a REVISION ROUND on the 70_ plots + 71_ site,
  piloting on jwst/A4, BEFORE relaunching mass production (so the diag
  stage runs with final plot code).  User asked how stars are selected
  for plots (answered: 3b is_star for plots 1-2; + unsaturated, n₁=0,
  clean stamp for composites 3-4; noted purity-over-completeness bias —
  a looser sample is a one-line change if requested).

## ⚡⚡⚡⚡ STATUS 2026-06-11 afternoon — FULL 3a-③+3b+DAO+diag MASS PRODUCTION SELF-DRIVING (superseded — driver stopped)

**A4 patch complete end-to-end** (jwst/A4 + hst/A4 + euclid
101541375/101542818/101542817): detection → 53_ PSF photometry →
67_ 3b stars (358/81/285/293/320 — stable) → 66_ DAO pool photometry
(SEx−DAO scatter 0.06–0.11 mag, per-band wing-truncation offsets in
meta, psf_phot_consistent ~70%) → 70_ per-filter diag plots.

**Chain 53→67→66→70 via 69_run_photometry.py over all 96 tiles is
RUNNING with an auto-followup** (background job): when pass 1 ends, a
completion sweep retries the 23 pre-fix sparse-NISP Euclid failures
(53_ band_meta now BORROWS nearest tile's PSF/ZP) + fills diag plots
everywhere, then 71_make_diag_site.py refreshes html/diag_qa/ and a
final per-mission tally prints.  HST 19/19 already complete.

**New scripts this session:** 66_ (photutils PSFPhotometry, GriddedPSFModel
from PSFEx poly, stamps normalized to over²; pool = Gaia ∪ tilted-locus
∪ DAO vetoes), 69_ (4-stage driver + phot_timing.csv), 70_ (per-filter:
RA/Dec star map, APER−PSF/DAO compare, ×9-drizzle composite PSF sqrt
stretch ±1.2″, profile cuts), 71_ (diag QA site in psf_qa layout).
67_ vote-rule fix: band vote needs ≥2 finite stats (≥1 DAO) + mag within
training range +1 (jwst/A4 18,846 → 358).

**Edge objects:** tiles overlap (Euclid ~2′, COSMOS-Web less); plan =
dedup by edge distance at the cross-tile union using the Step-1
products (csvfiles_star/v04/tile_lookup.parquet + coverage masks +
common_area_*WKT/geojson) — NO pixel stitching unless footprints show
true seams.  User question answered 2026-06-11.

**Next after mass production:** verify final tally (expect ~96 tiles ×
4 stages, ~25–30k stars total), spot-check diag site, commit timing
CSVs, then Step 4 (per-filter empirical saturation calibration) and
Step 5 (PM four pairs) per ms.tex; ms.tex §B.1 amendments still owed
(chi+ sqrt eq., per-pixel N, signed single-band, hot-thresh negqa,
3b G window 14–21).

## ⚡ STATUS 2026-06-11 — 3a-① unWISE COMPLETE (10/10): all ground+space surveys now have PSF models

**unWISE neo7 W1/W2 (data-collection session): DONE.**  5 COSMOS tiles ×
w1/w2 = 10 models, all `per_tile_ceiling` saturation, χ²=0.36–0.42.
- New: `54_step3a_build_psf_model_unwise.py` (SDSS-derived; 2.75″/px,
  ZP_AB = 22.5 Vega + VEGA2AB {w1 2.699, w2 3.339}, no mask plane),
  `64_step3a_unwise_global_saturation.py` → csv_saturation/
  `unwise_global_onsets.json` (W1 onset 10.80 AB ceiling 7.66e4 counts;
  W2 10.24 AB ceiling 2.26e5; matches WISE physics ~8.0/7.0 Vega).
  Per-tile turnover missed W2's pinned mag~9.5 stars on 2/5 tiles —
  global-JSON route fixed it (4–6 saturated now excluded per tile).
- 56_: unwise_w1/w2 INSTRUMENTS entries + saturation-plot bright edge
  is now onset-aware (was hard mag>13 / xlim 13.5 — unWISE's whole
  saturated population fell off-frame).  60_: `unwise` mission added
  (--reuse-pass1).  55_: unWISE-W1/W2 rows live on the QA site.
- Caveat (intrinsic to WISE 6″ resolution): unresolved galaxies sit ON
  the stellar locus; the SNR-weighted locus fit is star-dominated and
  the purity probe cannot detect departure.  W1 ~1700–1900 model stars
  per tile, W2 ~670–850.

## ⚡⚡⚡ STATUS 2026-06-11 night — 3a-② COMPLETE (96/96 tiles); 3a-③ + 3b coded & HST-validated

**Step 3a-② detection: DONE.**  `65_run_chi2_detection.py`: 96 tiles OK,
0 fail (20 JWST + 20 HST + 60 Euclid), 2.8 h.  Final recipe v3 in 51_:
- χ₊ = sqrt(Σ/N(x,y)) per-pixel band count + binary coverage MAP_WEIGHT
  (chi2_<tile>.wht.fits); HST = SIGNED single-band image (truncation
  collapsed the background estimator: A4 went 86 → 11,938 sources).
- hot DETECT_THRESH per mission via negative-image QA (in every chi2
  meta): jwst 5.0σ (4.3% spurious), euclid 4.0σ (4.5%), hst 3.0σ (0.0%).
  Cold pass is pure everywhere (≤0.8%).  Euclid rim tiles legitimately
  show high negqa (BGSUB edge residuals) — flagged by meta, not deleted.
- A4 sanity: 53,703 merged ≈ COSMOS2025 density.  ms.tex §B.1 eq.
  (chi2plus) needs amending (sqrt + per-pixel N + signed single-band).

**Step 3a-③ (53_step3a_dual_photometry.py): coded, HST A4 validated.**
Dual-image (det=chi₊+weight, meas=native band) forced PSF photometry
with the 3a-① stars_<tile>.psf, ZP/sat levels from 3a-① metas,
NUMBER-aligned join (no positional matching), saturation = flag never
drop (core-masked | peak-plateau | onset-mag), DAO sharp/rnd1/rnd2
KDTree-matched to the spine.  HST A4: 11,938 rows × 51 cols; PSF mags
all finite; Gaia check G−F814W = −0.18 ± 0.54; 65 saturated flagged;
DAO matched 84%.  CAVEAT: CHI2_PSF absolute scale is arbitrary (weight
scale + GAIN 0 → no Poisson term) — fine for 3b envelopes (same scale
in training), absolute interpretation deferred.  DAO morphology is the
slow stage (HST A4 ~2 h under disk contention) — profile before the
photometry mass run.  DAO PSF-FITTING photometry on the point-source
pool (psf_phot_consistent, GriddedPSFModel) = 66_, NOT yet written
(not needed by 3b).

**Step 3b (67_step3b_classifier.py): coded, machinery validated on HST A4.**
Gaia wide-CSV match (1″), PM ≥5σ & ≥5 mas/yr training, per-band
per-mag percentile envelopes (CHI2_PSF one-sided; DAO sharp/rnd1/rnd2
two-sided), is_star_psf_<band>, aggregate = Gaia-PM OR ≥3 unsaturated
votes.  DEVIATION (documented in 67_ docstring): training G window
14–21 not 14–18 — the locked window is itself saturated in deep space
imaging (HST onset 19.0 → 4 usable stars).  HST A4: 30 training stars,
4 envelopes, 617 band votes, is_star=81 (Gaia-PM; 1-band tiles can't
reach 3 votes until the cross-mission union).

**2026-06-11 08:25 wrap-up (computer shutdown).  ALL PILOTS DONE:**
- 53_ speed SOLVED: SPREAD_MODEL was 170× the whole pass cost
  (benchmark 404.6 s vs 2.4 s on 4.2 Mpx) → new configs/pass2_psf.param
  without it; 68_make_phot_psf.py wrote 6×FWHM _phot.psf crops (1958).
  Euclid 4-band tile now 655 s; JWST A4 (280k rows × 8 passes) ~35 min.
- phot pilots complete: hst/A4 (11,938×51, Gaia G−F814W −0.18±0.54),
  euclid/101541375 (46,747×103), jwst/A4 (53,703 rows, 4 bands).
- 67_ 3b pilots: euclid/101541375 → is_star 421 (320 by ≥3 votes,
  187 Gaia-PM, ~5.2k/deg² PLAUSIBLE); hst/A4 → 81 (Gaia-PM only,
  1 band).  ⚠ KNOWN ISSUE: jwst/A4 → 18,846 "stars" (35%!) — faint
  sources with NaN DAO stats pass the flat-extrapolated envelopes
  trivially (my rule: non-finite stat doesn't fail the band; JWST DAO
  match is shallow vs the spine).  FIX NEXT SESSION in 67_: require
  ≥2 finite stats (incl. ≥1 DAO) for a band vote AND don't extrapolate
  envelopes fainter than the training range + ~1 mag (set vote=False
  beyond).  Euclid was less affected (deeper DAO match fraction).

**Next session worklist:**
1. Fix 67_ vote rule (above), re-check jwst/A4 (expect few hundred to
   ~2k stars incl. faint-compact, NOT 18k) and euclid/101541375.
2. Write 69_run_photometry.py mass driver (mirror 65_; jwst 2-way,
   hst 3-way, euclid 4-way; ~3-4 h for all 96 tiles) → run 53_ + 67_
   everywhere.
3. 66_ DAO pool PSF-fitting photometry (GriddedPSFModel from the
   PSFEx .psf, ALLSTAR-style grouping) → psf_phot_consistent.
4. ms.tex §B.1 amendments: chi+ sqrt + per-pixel N + signed single-band
   eq.; hot-thresh calibration by negative-image QA; 3b G window 14–21.
5. QA pages for detection+phot (counts, negqa, envelope PNGs are in
   <mission>_chi2/<tile>/phot/star_envelopes_<tile>.png).

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
