# PSF QA pipeline — session handoff (Step 3a)

**Last updated:** 2026-06-08. **STATUS: JWST 4-band A4 COMPLETE & VALIDATED.**
Per-filter DEBLEND_MINCONT + SAMPLE_FWHMRANGE tuned; all 4 bands' PSF
models reproduce the diffraction spikes; QA site rebuilt.  Full record
now in `programs_star/CLAUDE.md` §"Step 3a".  Reproduce with
`./57_run_jwst_psf_qa.sh A4`.  Remaining: HST F814W A4 run (adapter ready),
scale to all tiles.  (History below kept for reference.)

## Where we are

Building per-band empirical PSF models (Step 3a) for the COSMOS cross-mission
stellar catalog v04, with a static QA website to inspect them. Pilot tile = **A4**.

### Scripts (all in `programs_star/v04/`)
- `54_step3a_build_psf_model.py` — JWST PSF builder (SExtractor pass-1 → star
  selection → PSFEx). Per-band configs `configs/jwst_nircam_<band>.sex`.
- `54_step3a_build_psf_model_euclid.py` — Euclid VIS variant (single-HDU, ZP from
  MAGZERO, 51px VIGNET).
- `54_step3a_build_psf_model_hst.py` — HST ACS F814W variant (COSMOS_ACS2005,
  ZP from PHOTFLAM/PHOTPLAM). **Built but NOT yet run.**
- `56_make_psf_qa_plots.py` — makes the 6 QA plots (JWST). `_euclid.py` variant too.
- `55_make_psf_qa_site.py` — builds the static site at `html/psf_qa/`.

### "Good star" selection (locked, in 54_)
- CLASS_STAR>0.8, SNR_WIN>100, ELONGATION<1.5, ELLIPTICITY<0.20, FWHM>fwhm_min
- fwhm_min = 0.5 × PSF_FWHM (from high-SNR median) — kills CR/hot-pixel artifacts
- locus−3·MAD < FLUX_RADIUS < locus + 2.5·std  (iterative MAD)
- masked_core==0 (saturation: i2d masks saturated cores to 0)
- gap_masked: >5 zero pixels in VIGNET footprint (±vig_half) → excluded
- NOT (n_1≥2 AND FLAGS<2)  (neighbour contamination; FLAGS<2 keeps spike stars)
- PSFEx: SAMPLE_FLAGMASK 0x00fc, SAMPLE_MAXELLIP 0.18

## CURRENT TASK (in progress — RESUME HERE)

**Per-filter DEBLEND_MINCONT optimization** so diffraction spikes stay ATTACHED
to the parent star (user requirement: "diffraction spikes are part of PSF
structure"). Aggressive deblending (old MINCONT=0.0005) split spikes into
separate objects → spike pixels blanked (-1e30) in the VIGNET → PSFEx rejected
bright stars (F444W: only 87/376 accepted, mostly bright ones rejected with
FLAGS_PSF=32 TOO_HIGH_SATU).

### Sweep results (saved: `deblend_sweep_A4.log`)
Metric: % of bright (mag<20) stars with FLAGS<2 (spikes merged) + mean n_1.

| Band | OLD FLAGS<2% | **CHOSEN MINCONT** | new FLAGS<2% | mean n_1 |
|------|---|---|---|---|
| F115W | 9%  | **0.02** | 88% | 0.21 |
| F150W | 3%  | **0.05** | 85% | 0.29 |
| F277W | 5%  | **0.05** | 66% | 0.12 |
| F444W | 2%  | **0.05** | 74% | 0.01 |

### DONE so far
- ✅ Sweep complete, knee identified, chosen values decided
- ✅ **Configs ALREADY UPDATED**: `configs/jwst_nircam_{f115w,f150w,f277w,f444w}.sex`
  now have the new DEBLEND_MINCONT (0.02 for F115W, 0.05 for the other 3).
  Verify with: `grep '^DEBLEND_MINCONT' configs/jwst_nircam_*.sex`
- ✅ Knee plot was being made: `/tmp/deblend_knee.png` (regenerate; /tmp clears
  on reboot — the plotting code is in the transcript).

### TODO on resume (the re-run was interrupted by shutdown)
1. Re-run **full pass-1** (NOT --reuse-pass1, since deblending changed) for all 4:
   ```
   cd ~/github/projects_cosmos
   for b in f115w f150w f277w f444w; do
     python programs_star/v04/54_step3a_build_psf_model.py \
       --instrument jwst_nircam_$b --tile A4 --filter $b
     python programs_star/v04/56_make_psf_qa_plots.py \
       --instrument jwst_nircam_$b --tile A4 --filter $b
   done
   python programs_star/v04/55_make_psf_qa_site.py --clean
   ```
   (each pass-1 ~100s; full loop ~15-20 min)
2. **VERIFY**: bright F444W stars now PSFEx-accepted (was 87/376; expect many
   more), and the PSF model VISIBLY shows the diffraction-spike pattern.
   Check meta JSON: `psf_<band>.meta.json` → psf_accepted, psf_chi2.
3. If F444W bright stars STILL rejected at 0.05, raise LW MINCONT to 0.1 and redo.
4. Regenerate `/tmp/deblend_knee.png` (knee diagnostic) — user asked to see it.

## Other pending items (lower priority)
- Run HST ACS F814W A4 (script ready: `54_..._hst.py --tile A4`, uses
  `/Volumes/exdisk1/data/HST/COSMOS_ACS2005/mosaic_cosmos_web_2023apr_30mas_tile_A4_hst_acs_wfc_f814w_drz.fits`)
- Commit all changes (54_/56_/55_ + configs + ms.tex) — NOT yet committed
- ms.tex §B.1: document the FWHM-artifact floor, gap-masked cut, and per-filter
  DEBLEND_MINCONT optimization

## Website
- `html/psf_qa/index.html` — 6 bands, click → per-tile rows → click panel → viewer
- Currently shows F115W/F150W/F277W/F444W/VIS (pre-deblend-fix versions); needs
  refresh after the re-run above.
- psf_samples/psf_residuals cell-centering bug FIXED (auto-detect cell size
  151/101/51; figsize scales with nrow; v19-equivalent).

## Data locations
- JWST i2d: `/Volumes/exdisk1/data/JWST/COSMOS_v0.8/mosaic_nircam_<band>_COSMOS-Web_30mas_A4_v1.0_i2d.fits`
- Euclid VIS: `/Volumes/exdisk1/data/Euclid/COSMOS_DR1/EUC_MER_BGSUB-MOSAIC-VIS_TILE<id>*.fits`
- HST F814W: `/Volumes/exdisk1/data/HST/COSMOS_ACS2005/` (A4 exists)
- Outputs: `/Volumes/exdisk1/data/photometry_v04/<instrument>/<tile>/psf/`
- ⚠ exdisk1 had transient I/O errors during this session — if FITS reads fail,
  remount the drive.
