#!/usr/bin/env python
"""
54_step3a_build_psf_model_euclid.py — Euclid VIS (DR1 MER) PSF model
builder following the same star-selection philosophy as the JWST script,
adapted to the different image format.

Differences from the JWST version:
  - Image is a single primary HDU (no SCI/WHT split)
  - ZP_AB from MAGZERO header keyword (Euclid pipeline product)
  - No PIXAR_SR; pixel scale 0.10"/px (CD matrix)
  - Smaller VIGNET (51 px) since VIS PSF is ~1.6 px FWHM
  - Different SExtractor / PSFEx configs (euclid_vis.sex,
    psfex_euclid_vis.psfex, pass1_euclid_vis.param)
  - Tile id = the long EUC_MER_BGSUB-MOSAIC-VIS_TILE<NNNNN> identifier;
    we just pass it as `--tile <id>` and look up the file by glob.

Star selection (identical philosophy to JWST):
  CLASS_STAR>0.8, SNR_WIN>100, ELONGATION<1.5, FWHM>0
  locus-3*MAD < FLUX_RADIUS < locus+2.5σ  (iterative MAD)
  masked_core == 0   (saturation = central pixels zero in source pixels)
  NOT (n_1>=2 AND FLAGS<2)
  NO FLAGS<2 cut
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
EUC_ROOT = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

PIX_VIS  = 0.10   # arcsec / pix for Euclid VIS


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
    xx   = np.asarray(obj['X_IMAGE'], float)
    yy   = np.asarray(obj['Y_IMAGE'], float)

    # iterative MAD on the base sample for locus + cut
    base = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
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
    saturated = masked_core > 0
    edge = (flg & 8) > 0

    # n_1 (1″ at 0.1″/px → 10 px)
    tree = cKDTree(np.column_stack([xx, yy]))
    r_pix = 1.0 / PIX_VIS
    n1 = np.array([len(tree.query_ball_point([xx[k], yy[k]], r_pix)) - 1
                   for k in range(len(obj))])
    contaminated = (n1 >= 2) & (flg < 2)

    # "Good star" ellipticity ceiling (F115W A4 study): ELLIPTICITY < 0.20
    # at candidate stage, PSFEx SAMPLE_MAXELLIP=0.18 downstream.
    ell_arr = np.asarray(obj['ELLIPTICITY'], float)
    e_round = ell_arr < 0.20
    star = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > 0)
            & (fr > cut) & (fr < med + 2.5*std0)
            & e_round
            & (~saturated) & (~edge) & (~contaminated))

    pre_contam = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > 0)
                  & (fr > cut) & (fr < med + 2.5*std0)
                  & e_round
                  & (~saturated) & (~edge))
    print(f'  stellar locus = {med:.3f} px  (MAD-σ {mad:.3f} px)')
    print(f'  artifact cut  = locus − 3·MAD = {cut:.3f} px')
    print(f'  PSF stars selected : {star.sum()}')
    print(f'    deblend-flagged spike stars     : {(star & (flg >= 2)).sum()}')
    print(f'  excluded saturated (masked core)  : {saturated.sum()}')
    print(f'  excluded contam (n_1≥2 & FLAGS<2) : {(pre_contam & contaminated).sum()}'
          f'  [of {pre_contam.sum()} pre-cut]')
    print(f'  min FR among PSF stars            : {fr[star].min():.3f} px'
          f'  ({"OK" if fr[star].min() > cut else "FAIL"} > cut {cut:.3f})')

    # write filtered LDAC
    star_cat = out / f'stars_{args.tile}.fits'
    hcat[2].data = obj[star]
    hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({star.sum()} stars)')

    # ── PSFEx ──
    print('\n── PSFEx ──', flush=True)
    cmd = ['psfex', str(star_cat),
           '-c', str(CONFIGS / 'psfex_euclid_vis.psfex'),
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
        'instrument': args.instrument, 'tile': args.tile,
        'zp_ab': zp, 'pixel_scale_arcsec': PIX_VIS,
        'stellar_locus_px': float(med), 'locus_mad_sigma_px': float(mad),
        'artifact_cut_px': float(cut),
        'n_psf_stars': int(star.sum()),
        'n_spike_stars_kept': int((star & (flg >= 2)).sum()),
        'n_saturated_excluded': int(saturated.sum()),
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
