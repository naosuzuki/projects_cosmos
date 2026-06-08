#!/usr/bin/env python
"""
54_step3a_build_psf_model_euclid.py — Euclid VIS (DR1 MER) PSF model
builder.  Single-HDU adaptation of the JWST builder, carrying the SAME
"good star" technology that was validated on the JWST bands:

  - FWHM artifact floor (kills the single-pixel CR/hot-pixel "second
    locus" that otherwise inflates the MAD and drives the artifact cut
    to negative half-light radius)
  - dynamic PSFEx SAMPLE_FWHMRANGE = [0.6,2.5]×PSF_FWHM so the
    acceptance window never clips the stellar locus / bright stars
  - VIGNET gap-mask test (a star whose VIGNET overlaps a coverage hole
    is rejected by PSFEx; pre-flag it so it never poses as a PSF star)
  - ELLIPTICITY<0.20 candidate ceiling (PSFEx SAMPLE_MAXELLIP=0.18)

Format differences from JWST:
  - single primary HDU (no SCI/WHT split); WEIGHT_TYPE NONE
  - ZP_AB from the MAGZERO header keyword
  - pixel scale 0.10"/px; VIGNET 51 px (VIS PSF ≈ 1.6 px FWHM)
  - tile id = the 9-digit number in EUC_MER_BGSUB-MOSAIC-VIS_TILE<id>

Star selection (identical philosophy to JWST):
  CLASS_STAR>0.8, SNR_WIN>100, ELONGATION<1.5, ELLIPTICITY<0.20,
  FWHM>0.5·PSF_FWHM,  locus-3·MAD < FLUX_RADIUS < locus+2.5σ,
  masked_core==0,  not gap-masked,  NOT (n_1>=2 AND FLAGS<2).
  Diffraction-spike (FLAGS>=2) stars are KEPT.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from psf_saturation import core_peak, detect_saturation

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
EUC_ROOT = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

PIX_VIS  = 0.10   # arcsec / pix for Euclid VIS
VIG_HALF = 25     # VIGNET(51,51) half-size, for the gap-mask test


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instrument', default='euclid_vis')
    p.add_argument('--tile', required=True,
                   help='Euclid tile id (the 9-digit number from TILE<id>)')
    p.add_argument('--core-box', type=int, default=3,
                   help='central NxN box to test for masked (=0) pixels '
                        '(Euclid VIS PSF is smaller than JWST → use 3×3)')
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def find_tile_image(tile_id: str) -> Path:
    matches = list(EUC_ROOT.glob(f'EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile_id}*.fits'))
    if not matches:
        sys.exit(f'No Euclid VIS image matches TILE{tile_id} in {EUC_ROOT}')
    return matches[0]


def main():
    args = parse_args()
    img = find_tile_image(args.tile)
    out = WORK / args.instrument / args.tile / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    # read ZP and SCI from the single HDU
    with fits.open(img) as h:
        sci = h[0].data.astype(np.float32)
        hdr = h[0].header
    zp = float(hdr['MAGZERO'])
    print(f'Instrument : {args.instrument}')
    print(f'Tile       : {args.tile}   ZP_AB={zp:.4f}   PIX={PIX_VIS}″')
    print(f'Image      : {img.name}  shape={sci.shape}')

    # write a single-HDU SCI file for SExtractor input
    sci_path = out / f'sci_{args.tile}.fits'
    fits.PrimaryHDU(sci, hdr).writeto(sci_path, overwrite=True)

    cat1 = out / f'pass1_{args.tile}.fits'
    if not (args.reuse_pass1 and cat1.exists()):
        print('\n── SExtractor pass 1 ──', flush=True)
        cmd = ['sex', str(sci_path),
               '-c', str(CONFIGS / 'euclid_vis.sex'),
               '-CATALOG_NAME', str(cat1),
               '-PARAMETERS_NAME', str(CONFIGS / 'pass1_euclid_vis.param'),
               '-FILTER_NAME', str(CONFIGS / 'gauss_2.0_5x5.conv'),
               '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
               '-MAG_ZEROPOINT', f'{zp:.4f}',
               '-WEIGHT_TYPE', 'NONE',
               '-VERBOSE_TYPE', 'QUIET']
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr); sys.exit('SExtractor failed')
        print(f'  wall {time.time()-t0:.1f}s')

    # ── star selection ──
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
    ell_arr = np.asarray(obj['ELLIPTICITY'], float)
    xx   = np.asarray(obj['X_IMAGE'], float)
    yy   = np.asarray(obj['Y_IMAGE'], float)

    # FWHM artifact floor + iterative MAD locus (see module docstring).
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
    med = np.median(fr[base])
    mad = 1.4826 * np.median(np.abs(fr[base] - med))
    for _ in range(20):
        sel = np.abs(fr[base] - med) < 3.0 * mad
        nmed = np.median(fr[base][sel])
        nmad = 1.4826 * np.median(np.abs(fr[base][sel] - nmed))
        if abs(nmed - med) < 1e-4 and abs(nmad - mad) < 1e-4:
            med, mad = nmed, nmad; break
        med, mad = nmed, nmad
    cut = med - 3.0 * mad

    # masked-core saturation test on SCI
    half = args.core_box // 2
    ny, nx = sci.shape
    xi = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi = np.clip(np.round(yy).astype(int), half, ny-half-1)
    masked_core = np.zeros(len(obj), int)
    for k in range(len(obj)):
        box = sci[yi[k]-half:yi[k]+half+1, xi[k]-half:xi[k]+half+1]
        masked_core[k] = np.sum(box == 0)
    edge = (flg & 8) > 0

    # Saturation = masked core (Euclid VIS MER zeroes saturated cores) OR a
    # core peak sitting on the bright-end PEAK PLATEAU (non-masking case).
    # detect_saturation finds the plateau from the point-source sample; for
    # Euclid the cores are masked so usually no plateau is found and the
    # masked-core test does the work (the onset magnitude is still reported
    # for the diagnostic line).
    is_point = (cs > 0.8) & (snr > 100)
    peak = core_peak(sci, xx, yy, half=2, idx=np.where(is_point)[0])
    sat_peak_level, onset_mag = detect_saturation(mag, peak, masked_core, is_point)
    peak_saturated = np.isfinite(peak) & (peak >= sat_peak_level)
    saturated = (masked_core > 0) | peak_saturated
    print(f'  saturation: masked-core={int((masked_core>0).sum())}  '
          f'peak-plateau={int(peak_saturated.sum())}  '
          f'(sat_peak_level={"∞" if not np.isfinite(sat_peak_level) else f"{sat_peak_level:.3g}"}, '
          f'onset_mag={onset_mag if onset_mag is None else round(onset_mag,2)})')

    # VIGNET-gap test: a star whose VIGNET footprint overlaps a coverage
    # hole (zero pixels) is surprise-rejected by PSFEx; pre-flag it.
    xiv = np.clip(np.round(xx).astype(int), VIG_HALF, nx-VIG_HALF-1)
    yiv = np.clip(np.round(yy).astype(int), VIG_HALF, ny-VIG_HALF-1)
    cheap = (cs > 0.8) & (snr > 100) & (elon < 1.5) & (fr > 0)
    vignet_masked = np.zeros(len(obj), int)
    for k in np.where(cheap)[0]:
        box = sci[yiv[k]-VIG_HALF:yiv[k]+VIG_HALF+1, xiv[k]-VIG_HALF:xiv[k]+VIG_HALF+1]
        vignet_masked[k] = np.sum(box == 0)
    gap_masked = vignet_masked > 5

    # n_1 (1″ at 0.1″/px → 10 px)
    tree = cKDTree(np.column_stack([xx, yy]))
    r_pix = 1.0 / PIX_VIS
    n1 = np.array([len(tree.query_ball_point([xx[k], yy[k]], r_pix)) - 1
                   for k in range(len(obj))])
    contaminated = (n1 >= 2) & (flg < 2)

    # "Good star" ellipticity ceiling (F115W A4 study): ELLIPTICITY < 0.20
    # at candidate stage, PSFEx SAMPLE_MAXELLIP=0.18 downstream.
    e_round = ell_arr < 0.20
    star = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
            & (fr > cut) & (fr < med + 2.5*std0)
            & e_round
            & (~saturated) & (~edge) & (~contaminated) & (~gap_masked))

    pre_contam = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
                  & (fr > cut) & (fr < med + 2.5*std0)
                  & e_round
                  & (~saturated) & (~edge) & (~gap_masked))
    pre_gap = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
               & (fr > cut) & (fr < med + 2.5*std0)
               & e_round & (~saturated) & (~edge))
    dropped_by_gap = pre_gap & gap_masked
    print(f'  stellar locus = {med:.3f} px  (MAD-σ {mad:.3f} px)')
    print(f'  artifact cut  = locus − 3·MAD = {cut:.3f} px')
    print(f'  PSF stars selected : {star.sum()}')
    print(f'    deblend-flagged spike stars     : {(star & (flg >= 2)).sum()}')
    print(f'  excluded saturated (masked core)  : {saturated.sum()}')
    print(f'  excluded gap-masked VIGNET        : {dropped_by_gap.sum()}'
          f'  [of {pre_gap.sum()} candidates pre-gap-cut]')
    print(f'  excluded contam (n_1≥2 & FLAGS<2) : {(pre_contam & contaminated).sum()}'
          f'  [of {pre_contam.sum()} pre-cut]')
    if star.sum():
        print(f'  min FR among PSF stars            : {fr[star].min():.3f} px'
              f'  ({"OK" if fr[star].min() > cut else "FAIL"} > cut {cut:.3f})')

    # write filtered LDAC
    star_cat = out / f'stars_{args.tile}.fits'
    hcat[2].data = obj[star]
    hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({star.sum()} stars)')

    # ── PSFEx — dynamic SAMPLE_FWHMRANGE scaled to the measured PSF FWHM ──
    fwhm_lo = max(0.8, 0.6 * psf_fwhm_est)
    fwhm_hi = 2.5 * psf_fwhm_est
    print('\n── PSFEx ──', flush=True)
    print(f'  SAMPLE_FWHMRANGE = {fwhm_lo:.2f},{fwhm_hi:.2f} px '
          f'(scaled to PSF FWHM {psf_fwhm_est:.2f})')
    cmd = ['psfex', str(star_cat),
           '-c', str(CONFIGS / 'psfex_euclid_vis.psfex'),
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

    # diagnostic JSON
    (out / f'psf_{args.tile}.meta.json').write_text(json.dumps({
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': args.instrument, 'tile': args.tile,
        'zp_ab': zp, 'pixel_scale_arcsec': PIX_VIS,
        'psf_fwhm_est_px': float(psf_fwhm_est),
        'sample_fwhmrange': [float(fwhm_lo), float(fwhm_hi)],
        'stellar_locus_px': float(med), 'locus_std_px': float(std0),
        'locus_mad_sigma_px': float(mad),
        'artifact_cut_px': float(cut),
        'n_psf_stars': int(star.sum()),
        'n_spike_stars_kept': int((star & (flg >= 2)).sum()),
        'n_saturated_excluded': int(saturated.sum()),
        'sat_peak_level': (None if not np.isfinite(sat_peak_level) else float(sat_peak_level)),
        'sat_onset_mag': onset_mag,
        'n_peak_saturated': int(peak_saturated.sum()),
        'n_gap_masked_excluded': int(dropped_by_gap.sum()),
        'n_contaminated_excluded': int((pre_contam & contaminated).sum()),
        'psf_chi2': float(ph.get('CHI2', -1)),
        'psf_fwhm_px': float(ph.get('PSF_FWHM', -1)),
        'psf_accepted': int(ph.get('ACCEPTED', -1)),
        'psf_model': str(psf),
    }, indent=2))
    print(f'\n[save] {out / f"psf_{args.tile}.meta.json"}')
    print('=== Euclid VIS PSF build complete ===')


if __name__ == '__main__':
    main()
