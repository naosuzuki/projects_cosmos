#!/usr/bin/env python
"""
54_step3a_build_psf_model.py — build the per-band empirical PSF model
for v04 Step 3a, using the validated COSMOS-Web / Tanaka+2023 recipe
and our star selection (keep diffraction-spike stars, exclude
saturated and artifact detections).

Validated on JWST F115W tile A4 (see the pilot diagnostics in
/Volumes/exdisk1/data/photometry_v04/pilot/):
  - PSF model: PSF_SIZE 201 (SW) / 301 (LW) oversampled 2×, PIXEL_AUTO,
    linear spatial model, SAMPLE_MINSN 100 (Tanaka+2023, arXiv:2309.03266)
  - PSF model now captures the NIRCam diffraction spikes (χ²≈1.66 on
    the high-SNR sample; a spike-LESS FLAGS<2 model gave a misleadingly
    lower 1.10 because it omitted the bright stars where spikes matter)

Star selection (per band):
  CLASS_STAR  > 0.8
  SNR_WIN     > 100
  ELONGATION  < 1.5
  ELLIPTICITY < 0.20      ("good star" ellipticity cap; F115W A4 study)
  FWHM_IMAGE  > 0                             (reject fit-failures)
  locus−3·MAD < FLUX_RADIUS < locus+2.5σ      (artifact-cut floor +
                                               stellar-locus ceiling)
  NOT (n_1 ≥ 2 AND FLAGS < 2)                 (neighbour-contamination cut:
                                               drop stars with ≥2 detections
                                               within 1″ that are NOT their
                                               own spike-deblends.  Bright
                                               diffraction-spike stars have
                                               FLAGS ≥ 2 because SExtractor
                                               deblends their spikes; those
                                               n_1 counts are self-spikes,
                                               not external companions.)
   where:
     locus = median FLUX_RADIUS of the CLASS_STAR>0.8 / SNR>100 /
             FLAGS<2 / ELON<1.5 base sample.
     MAD   = 1.4826 × median|FR−locus| on iterative 3·MAD-clipped base.
     Floor = locus − 3·MAD ≈ 31 mas for F115W (locus 37 mas, MAD 2 mas).
     n_1   = count of detections (any type) within 1″ of the candidate
             centroid (KDTree neighbor query on the pass-1 catalog;
             self excluded).
  masked_core == 0                            (NOT saturated — JWST i2d
                                               masks the flat-topped
                                               saturated cores to 0;
                                               this is the saturation
                                               signature, not FLAGS&4)
  NO FLAGS<2 cut                              (keep bright stars whose
                                               own diffraction spikes
                                               were deblended)
PSFEx then runs with SAMPLE_FLAGMASK 0x00fc (allow neighbor+deblend,
reject saturated+edge bits) + SAMPLE_AUTOSELECT for final clipping.

Procedure:
  1. SExtractor pass-1 on the band SCI (big VIGNET: 101 SW / 151 LW)
  2. compute stellar locus + apply the selection above (the masked-core
     test reads the SCI pixels at each centroid)
  3. write a filtered FITS_LDAC star catalog (in-place row filter to
     preserve the LDAC structure incl. VIGNET TDIM)
  4. run PSFEx (psfex_jwst_sw.psfex / _lw.psfex) → <band>.psf model
  5. write a star-selection diagnostic JSON

Outputs to /Volumes/exdisk1/data/photometry_v04/<instrument>/<tile>/psf/.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

# SW = short-wavelength (F115W, F150W); LW = long (F277W, F444W)
SW_BANDS = {'f115w', 'f150w'}
LW_BANDS = {'f277w', 'f444w'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', default='jwst_nircam_f115w',
                   help='instrument key matching configs/<instrument>.sex')
    p.add_argument('--tile', default='A4')
    p.add_argument('--filter', default='f115w',
                   choices=['f115w', 'f150w', 'f277w', 'f444w'])
    p.add_argument('--core-box', type=int, default=5,
                   help='central NxN box to test for masked (=0) pixels '
                        '(saturation signature). Default 5.')
    p.add_argument('--reuse-pass1', action='store_true',
                   help='reuse existing pass-1 catalog if present')
    return p.parse_args()


def band_image(tile, band):
    p = JWST_DIR / f'mosaic_nircam_{band}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'
    if not p.exists():
        sys.exit(f'Missing tile image: {p}')
    return p


def zp_from_pixar(image):
    with fits.open(image) as h:
        for hdu in h:
            if 'PIXAR_SR' in hdu.header:
                return -2.5*np.log10(hdu.header['PIXAR_SR']*1e6) + 8.9
    sys.exit('No PIXAR_SR for ZP')


def main():
    args = parse_args()
    band = args.filter
    chan = 'sw' if band in SW_BANDS else 'lw'
    out  = WORK / args.instrument / args.tile / 'psf'
    out.mkdir(parents=True, exist_ok=True)
    img  = band_image(args.tile, band)
    zp   = zp_from_pixar(img)

    print(f'Instrument : {args.instrument}   ({chan.upper()} channel)')
    print(f'Tile/band  : {args.tile} / {band.upper()}   ZP_AB={zp:.4f}')
    print(f'PSF config : psfex_jwst_{chan}.psfex   param: pass1_jwst_{chan}.param')

    # ── 1. extract SCI+WHT single-HDU, run SExtractor pass 1 ──
    with fits.open(img) as h:
        sci = h['SCI'].data.astype(np.float32)
        sci_hdr = h['SCI'].header
        wht = h['WHT'].data.astype(np.float32)
        wht_hdr = h['WHT'].header
    sci_path = out / f'sci_{band}.fits'
    wht_path = out / f'wht_{band}.fits'
    fits.PrimaryHDU(sci, sci_hdr).writeto(sci_path, overwrite=True)
    fits.PrimaryHDU(wht, wht_hdr).writeto(wht_path, overwrite=True)

    cat1 = out / f'pass1_{band}.fits'
    if not (args.reuse_pass1 and cat1.exists()):
        print('\n── SExtractor pass 1 (detection + big VIGNET) ──', flush=True)
        cmd = ['sex', str(sci_path),
               '-c', str(CONFIGS / f'{args.instrument}.sex'),
               '-CATALOG_NAME', str(cat1),
               '-PARAMETERS_NAME', str(CONFIGS / f'pass1_jwst_{chan}.param'),
               '-FILTER_NAME', str(CONFIGS / 'default.conv'),
               '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
               '-WEIGHT_IMAGE', str(wht_path),
               '-MAG_ZEROPOINT', f'{zp:.4f}',
               '-VERBOSE_TYPE', 'QUIET']
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr); sys.exit('SExtractor pass1 failed')
        print(f'  wall {time.time()-t0:.1f}s')

    # ── 2. star selection ──
    print('\n── star selection ──', flush=True)
    hcat = fits.open(cat1)
    obj  = hcat[2].data
    mag  = np.asarray(obj['MAG_AUTO'], float)
    fr   = np.asarray(obj['FLUX_RADIUS'], float)
    fwhm = np.asarray(obj['FWHM_IMAGE'], float)
    cs   = np.asarray(obj['CLASS_STAR'], float)
    snr  = np.asarray(obj['SNR_WIN'], float)
    flg  = np.asarray(obj['FLAGS'], int)
    elon = np.asarray(obj['ELONGATION'], float)
    xx   = np.asarray(obj['X_IMAGE'], float)
    yy   = np.asarray(obj['Y_IMAGE'], float)

    # stellar locus from a clean base.  Ellipticity ceiling is
    # ELONGATION<1.5 (= ELLIPTICITY<0.33), plus PSFEx's MAXELLIP=0.20.
    # Add a FWHM lower bound to exclude single-pixel CR/hot-pixel
    # artifacts — these contaminate the LW base sample (F277W has ~6000
    # such sources at FR≈0.8 px) and inflate MAD so the artifact cut
    # goes negative.  Threshold = 0.5 × PSF FWHM, estimated from the
    # high-SNR subset where real stars dominate.
    base0 = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
    hi = base0 & (snr > 1000)
    if hi.sum() < 5:
        hi = base0 & (snr > 500)
    psf_fwhm_est = float(np.median(fwhm[hi])) if hi.sum() >= 3 else float(np.median(fwhm[base0]))
    fwhm_min = 0.5 * psf_fwhm_est
    base = base0 & (fwhm > fwhm_min)
    print(f'  PSF FWHM estimate (high-SNR median): {psf_fwhm_est:.2f} px '
          f'→ base FWHM-min = {fwhm_min:.2f} px '
          f'(removes {(base0 & ~base).sum()} CR/hot-pixel artifacts)')
    std0 = np.std(fr[base])
    # Iterative 3·MAD sigma-clipping on the base FR distribution.  This
    # self-converges to the actual PSF locus without any hand-picked SNR
    # threshold, and gives a robust scatter estimate (MAD) immune to
    # outliers.  Validated on F115W A4: converges in 3 iterations to
    # median=1.22 px, MAD=0.073 px, cut=1.01 px — sits in the gap
    # between the artifact cluster (FR < 0.85 px) and the PSF locus.
    med = np.median(fr[base])
    mad = 1.4826 * np.median(np.abs(fr[base] - med))
    for _ in range(20):
        sel = np.abs(fr[base] - med) < 3.0 * mad
        new_med = np.median(fr[base][sel])
        new_mad = 1.4826 * np.median(np.abs(fr[base][sel] - new_med))
        if abs(new_med - med) < 1e-4 and abs(new_mad - mad) < 1e-4:
            med, mad = new_med, new_mad; break
        med, mad = new_med, new_mad
    # Artifact cut: 3 robust σ (MAD-σ) below the converged stellar
    # locus.  Anything below this is sharper-than-PSF → cosmic ray / hot
    # pixel, NOT a faint star (faints scatter UP from the locus due to
    # photon noise, not down).
    cut = med - 3.0 * mad
    # Keep std handy for the upper ceiling (uses base std, not MAD —
    # the upper end is dominated by extended objects we want a soft cut on)
    std = std0

    # masked-core (saturation) test on the SCI pixels
    half = args.core_box // 2
    ny, nx = sci.shape
    xi = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi = np.clip(np.round(yy).astype(int), half, ny-half-1)
    masked_core = np.zeros(len(obj), int)
    for k in range(len(obj)):
        box = sci[yi[k]-half:yi[k]+half+1, xi[k]-half:xi[k]+half+1]
        masked_core[k] = np.sum(box == 0)
    saturated = masked_core > 0
    edge = (flg & 8) > 0

    # VIGNET-gap test: PSFEx rejects any sample whose VIGNET contains
    # masked (zero) pixels as TOO_HIGH_SATU.  This is NOT saturation —
    # the masked pixels come from chip gaps / coverage holes in the
    # mosaic that fall inside the (large) VIGNET footprint.  Bright LW
    # stars near a gap have clean cores but a masked VIGNET, so they
    # were mislabeled as PSF candidates and then surprise-rejected by
    # PSFEx.  Pre-flag them here so they don't appear as "PSF stars".
    #
    # Use the VIGNET half-size for this band (SW=50, LW=75 native px).
    vig_half = 75 if chan == 'lw' else 50
    xiv = np.clip(np.round(xx).astype(int), vig_half, nx-vig_half-1)
    yiv = np.clip(np.round(yy).astype(int), vig_half, ny-vig_half-1)
    # Only worth computing for the sources that pass the cheap cuts —
    # restrict to a pre-mask so we don't scan 365k boxes.
    cheap = (cs > 0.8) & (snr > 100) & (elon < 1.5) & (fr > 0)
    vignet_masked = np.zeros(len(obj), int)
    for k in np.where(cheap)[0]:
        box = sci[yiv[k]-vig_half:yiv[k]+vig_half+1, xiv[k]-vig_half:xiv[k]+vig_half+1]
        vignet_masked[k] = np.sum(box == 0)
    # tolerate a few isolated bad pixels (PSFEx interpolates over <~5);
    # flag stars whose VIGNET overlaps a real gap (many masked pixels).
    gap_masked = vignet_masked > 5

    # Neighbour count within 1" (n_1).  External neighbours within 1"
    # contaminate the PSF.  Spike-deblended bright stars (FLAGS≥2) get
    # their own spike fragments counted as neighbours, so we only cut on
    # n_1 ≥ 2 AND FLAGS < 2 (no self-deblending).  Verified on F115W A4:
    # cuts 9 visibly-contaminated stars (vs cutting 44 if FLAGS<2 not
    # required, 35 of which were real spike stars we want to keep).
    from scipy.spatial import cKDTree
    R_ARCSEC = 1.0
    pix_scale = 0.030               # arcsec/px on COSMOS-Web 30 mas mosaics
    r_pix = R_ARCSEC / pix_scale
    tree = cKDTree(np.column_stack([xx, yy]))
    n1 = np.array([len(tree.query_ball_point([xx[k], yy[k]], r_pix)) - 1
                   for k in range(len(obj))])
    contaminated = (n1 >= 2) & (flg < 2)

    # "Good star" ellipticity ceiling at the candidate stage:
    # ELLIPTICITY < 0.20 (with PSFEx SAMPLE_MAXELLIP=0.18 downstream).
    # F115W A4 study: 153/167 (91.6%) of PSFEx-accepted "good stars"
    # have SExtractor ELLIPTICITY<0.20.
    ell_arr = np.asarray(obj['ELLIPTICITY'], float)
    e_round = ell_arr < 0.20
    star = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
            & (fr > cut) & (fr < med + 2.5*std)
            & e_round
            & (~saturated) & (~edge) & (~contaminated) & (~gap_masked))
    print(f'  stellar locus = {med:.3f} px  (MAD-σ {mad:.3f} px, '
          f'iterative 3·MAD clipping)')
    print(f'  artifact cut  = locus − 3·MAD = {cut:.3f} px')
    # base of candidates that pass everything EXCEPT the contam cut —
    # gives the meaningful "candidates dropped by this cut" count.
    pre_contam = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
                  & (fr > cut) & (fr < med + 2.5*std)
                  & e_round
                  & (~saturated) & (~edge) & (~gap_masked))
    dropped_by_contam = pre_contam & contaminated
    # gap-masked candidates: pass everything except the gap test
    pre_gap = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
               & (fr > cut) & (fr < med + 2.5*std)
               & e_round & (~saturated) & (~edge))
    dropped_by_gap = pre_gap & gap_masked
    print(f'  PSF stars selected : {star.sum()}')
    print(f'    of which deblend-flagged (spike) : {(star & (flg >= 2)).sum()}')
    print(f'  excluded saturated (masked core)   : {saturated.sum()}')
    print(f'  excluded gap-masked VIGNET         : {dropped_by_gap.sum()}'
          f'  [of {pre_gap.sum()} candidates pre-gap-cut]')
    print(f'  excluded contam (n_1≥2 AND FLAGS<2): {dropped_by_contam.sum()}'
          f'  [of {pre_contam.sum()} candidates pre-cut]')
    print(f'  min FLUX_RADIUS among stars        : {fr[star].min():.3f} px '
          f'({"OK" if fr[star].min() > cut else "FAIL"} > cut {cut:.3f})')

    # ── 3. write filtered LDAC (in-place row filter preserves TDIM) ──
    star_cat = out / f'stars_{band}.fits'
    hcat[2].data = obj[star]
    hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({star.sum()} stars)')

    # ── 4. PSFEx ──
    # Per-band SAMPLE_FWHMRANGE scaled to the measured PSF FWHM.  The fixed
    # config range (SW 1.5-4.0, LW 2.0-6.0) cut THROUGH the stellar locus
    # for the wide LW PSFs: F444W PSF FWHM≈5.9 px but bright stars measure
    # up to 8.7 px (resolved wings), so the 6.0 ceiling rejected ALL bright
    # stars (the ones carrying the diffraction-spike signal).  Scale the
    # window to [0.6, 2.5]×PSF_FWHM so bright stars are kept across bands.
    fwhm_lo = max(1.2, 0.6 * psf_fwhm_est)
    fwhm_hi = 2.5 * psf_fwhm_est
    print('\n── PSFEx ──', flush=True)
    print(f'  SAMPLE_FWHMRANGE = {fwhm_lo:.2f},{fwhm_hi:.2f} px '
          f'(scaled to PSF FWHM {psf_fwhm_est:.2f})')
    cmd = ['psfex', str(star_cat),
           '-c', str(CONFIGS / f'psfex_jwst_{chan}.psfex'),
           '-SAMPLE_FWHMRANGE', f'{fwhm_lo:.2f},{fwhm_hi:.2f}',
           '-CHECKIMAGE_TYPE', 'RESIDUALS,PROTOTYPES,SNAPSHOTS,SAMPLES',
           '-CHECKIMAGE_NAME',
           f'{out}/resi.fits,{out}/proto.fits,{out}/snap.fits,{out}/samp.fits']
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr); sys.exit('PSFEx failed')
    psf = star_cat.with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists(): shutil.move(alt, psf)
    ph = fits.open(psf)[1].header
    print(f'  wall {time.time()-t0:.1f}s')
    print(f'  PSF model: chi2={ph.get("CHI2",-1):.3f}  '
          f'FWHM={ph.get("PSF_FWHM",-1):.2f} px  accepted={ph.get("ACCEPTED","?")}')

    # ── 5. diagnostic JSON ──
    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': args.instrument, 'tile': args.tile, 'filter': band,
        'channel': chan, 'zp_ab': float(zp),
        'psf_fwhm_est_px': float(psf_fwhm_est),
        'sample_fwhmrange': [float(fwhm_lo), float(fwhm_hi)],
        'stellar_locus_px': float(med), 'locus_std_px': float(std),
        'locus_mad_sigma_px': float(mad),
        'artifact_cut_px': float(cut),
        'artifact_cut_recipe': 'locus - 3 * MAD(tight base)',
        'n_psf_stars': int(star.sum()),
        'n_spike_stars_kept': int((star & (flg >= 2)).sum()),
        'n_saturated_excluded': int(saturated.sum()),
        'n_gap_masked_excluded': int(dropped_by_gap.sum()),
        'n_contaminated_excluded': int(dropped_by_contam.sum()),
        'contam_cut_recipe': 'n_1 >= 2 AND FLAGS < 2  (n_1 = neighbours within 1″)',
        'psf_chi2': float(ph.get('CHI2', -1)),
        'psf_fwhm_px': float(ph.get('PSF_FWHM', -1)),
        'psf_accepted': int(ph.get('ACCEPTED', -1)),
        'psf_model': str(psf),
        'recipe': 'Tanaka+2023 COSMOS-Web; keep diffraction-spike stars; '
                  'exclude masked-core saturated; artifact-cut FR floor',
    }
    (out / f'psf_{band}.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n[save] {out / f"psf_{band}.meta.json"}')
    print('=== PSF model build complete ===')


if __name__ == '__main__':
    main()
