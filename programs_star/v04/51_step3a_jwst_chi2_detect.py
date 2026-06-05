#!/usr/bin/env python
"""
51_step3a_jwst_chi2_detect.py — Build the χ²+ detection image from
4 NIRCam bands (COSMOS-Web 30 mas mosaics) and run the SExtractor
cold + hot detection passes, then merge.

Follows the Shuntov+ 2025 (COSMOS2025) recipe verified against
Galametz+ 2013 (CANDELS).  See ms.tex §B.1 Step 3a for the locked
plan.

Procedure (per tile):
  1. Load SCI + WHT extensions for each of F115W, F150W, F277W, F444W
     from the COSMOS-Web v0.8 mosaics.
  2. PSF-homogenize each non-F444W band to F444W by convolving with
     a Gaussian kernel of σ such that σ²_target = σ²_source + σ²_kernel.
     (Empirical PSF homogenization deferred to production; Gaussian
     approximation is sufficient for the pilot.)
  3. NSCI = SCI_homog × √WHT  (noise-equalized image; unit variance
     per pixel at the noise level).
  4. Truncate NSCI → max(NSCI, 0) per pixel.
  5. χ²+ = Σ_b NSCI_b² (sum over 4 bands of squared positive
     contributions).
  6. Save χ²+ image FITS (WCS inherited from F444W).
  7. Run SExtractor cold pass → cold_cat + seg_cold
  8. Run SExtractor hot pass  → hot_cat  + seg_hot
  9. Merge: keep all cold; add hot whose centroid lies outside every
     cold source's Kron ellipse (A_IMAGE × KRON_RADIUS, etc.).
 10. Save merged_cat.fits (the source list for downstream per-band
     PSFEx + PSF photometry).

Outputs go under /Volumes/exdisk1/data/photometry_v04/jwst_chi2/<TILE>/.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# Project paths
PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04/jwst_chi2')

# Empirical effective FWHM of COSMOS-Web 30 mas mosaics (arcsec, then pix).
# F115W FWHM measured by PSFEx on the full A4 tile: 0.073" (2.44 pix at 30 mas).
# Other bands estimated by scaling the diffraction limit and applying the
# same drizzle broadening factor (~1.7x).  Replace with empirical values
# (PSFEx per band) for production.
PIX_SCALE = 0.030
FWHM_ARCSEC = {
    'f115w': 0.073,    # empirical (A4 tile)
    'f150w': 0.085,    # estimate
    'f277w': 0.120,    # estimate
    'f444w': 0.165,    # estimate (target of homogenization)
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tile', default='A4',
                   help='COSMOS-Web tile letter (default A4).')
    p.add_argument('--cutout-size', type=int, default=0,
                   help='Optional cutout side in pixels (0 = full tile).')
    p.add_argument('--target-band', default='f444w',
                   choices=['f444w', 'f277w', 'f150w', 'f115w'],
                   help='PSF homogenization target (default f444w broadest).')
    p.add_argument('--skip-stack', action='store_true',
                   help='Reuse existing chi2 image (debug).')
    p.add_argument('--skip-cold', action='store_true')
    p.add_argument('--skip-hot', action='store_true')
    return p.parse_args()


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


def kernel_sigma_to(arcsec_src: float, arcsec_tgt: float) -> float:
    """Return σ (pix) of the Gaussian kernel needed to bring a Gaussian
    PSF of FWHM arcsec_src up to FWHM arcsec_tgt.  σ in pixels.
    Returns 0 if src >= tgt (no homogenization needed)."""
    if arcsec_src >= arcsec_tgt:
        return 0.0
    s_src = arcsec_src / 2.355 / PIX_SCALE
    s_tgt = arcsec_tgt / 2.355 / PIX_SCALE
    return float(np.sqrt(s_tgt**2 - s_src**2))


def load_band(tile: str, band: str, cutout: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return SCI, WHT arrays (and SCI header dict) for a tile/band."""
    src = JWST_DIR / (f'mosaic_nircam_{band}_COSMOS-Web_30mas_'
                      f'{tile}_v1.0_i2d.fits')
    if not src.exists():
        sys.exit(f'Missing band: {src}')
    print(f'  loading {band:5s}  ({src.stat().st_size/1e9:.1f} GB) …', flush=True)
    t0 = time.time()
    with fits.open(src) as h:
        sci_hdr = h['SCI'].header
        nx, ny = sci_hdr['NAXIS1'], sci_hdr['NAXIS2']
        if cutout > 0:
            cx, cy = nx//2, ny//2
            x0, x1 = cx - cutout//2, cx + cutout//2
            y0, y1 = cy - cutout//2, cy + cutout//2
            sci = h['SCI'].data[y0:y1, x0:x1].astype(np.float32, copy=True)
            wht = h['WHT'].data[y0:y1, x0:x1].astype(np.float32, copy=True)
            sci_hdr = sci_hdr.copy()
            if 'CRPIX1' in sci_hdr: sci_hdr['CRPIX1'] -= x0
            if 'CRPIX2' in sci_hdr: sci_hdr['CRPIX2'] -= y0
            sci_hdr['NAXIS1'] = x1 - x0
            sci_hdr['NAXIS2'] = y1 - y0
        else:
            sci = h['SCI'].data.astype(np.float32, copy=True)
            wht = h['WHT'].data.astype(np.float32, copy=True)
            sci_hdr = sci_hdr.copy()
    print(f'    {band} shape {sci.shape}  ({time.time()-t0:.1f}s)')
    return sci, wht, dict(sci_hdr)


def build_chi2(tile: str, cutout: int, target_band: str,
               work: Path) -> Path:
    """Build the χ²+ detection image and save it."""
    chi2_path = work / f'chi2_{tile}.fits'
    if chi2_path.exists():
        print(f'  using existing {chi2_path.name}')
        return chi2_path

    fwhm_tgt = FWHM_ARCSEC[target_band]
    chi2 = None
    ref_hdr = None
    for band in ('f115w', 'f150w', 'f277w', 'f444w'):
        sci, wht, hdr = load_band(tile, band, cutout)
        sigma_pix = kernel_sigma_to(FWHM_ARCSEC[band], fwhm_tgt)
        if sigma_pix > 0.1:
            print(f'    homogenize {band} → {target_band}: '
                  f'Gaussian σ = {sigma_pix:.2f} pix')
            sci = gaussian_filter(sci, sigma_pix, mode='constant', cval=0)
        # noise-equalize
        nsci = sci * np.sqrt(np.maximum(wht, 0))
        # truncate negatives
        np.maximum(nsci, 0, out=nsci)
        # accumulate χ²+
        if chi2 is None:
            chi2 = nsci**2
            ref_hdr = hdr
        else:
            chi2 += nsci**2
        del sci, wht, nsci

    # Save χ²+ image
    print(f'\n  χ²+ stats: min {chi2.min():.2e}  '
          f'med {np.median(chi2):.2e}  max {chi2.max():.2e}')
    h = fits.PrimaryHDU(data=chi2.astype(np.float32))
    for k, v in ref_hdr.items():
        if k not in ('SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND',
                     'XTENSION','PCOUNT','GCOUNT','EXTNAME'):
            try:
                h.header[k] = v
            except Exception:
                pass
    h.header['BUNIT']    = ('chi2+',  'noise-equalized sum-of-squares')
    h.header['HISTORY']  = f'chi2+ stack of f115w+f150w+f277w+f444w'
    h.header['HISTORY']  = f'homogenized to {target_band} FWHM={fwhm_tgt}\"'
    h.writeto(chi2_path, overwrite=True)
    print(f'  [save] {chi2_path}  ({chi2_path.stat().st_size/1e9:.2f} GB)')
    return chi2_path


def run_sex_mode(image: Path, mode: str, work: Path,
                 tile: str) -> tuple[Path, Path]:
    """Run SExtractor cold or hot mode on the χ²+ image."""
    cfg = CONFIGS / f'jwst_nircam_chi2_{mode}.sex'
    out_cat = work / f'cat_{mode}_{tile}.fits'
    out_seg = work / f'seg_{mode}_{tile}.fits'
    cmd = ['sex', str(image),
           '-c',                str(cfg),
           '-CATALOG_NAME',     str(out_cat),
           '-PARAMETERS_NAME',  str(CONFIGS / 'default_pass1.param'),
           '-STARNNW_NAME',     str(CONFIGS / 'default.nnw'),
           '-FILTER_NAME',      str(CONFIGS / (
               'tophat_9.0_9x9.conv' if mode == 'cold'
               else 'gauss_3.0_5x5.conv')),
           '-CHECKIMAGE_NAME',  str(out_seg),
           ]
    print(f'  cmd: sex {image.name} -c {cfg.name} → {out_cat.name}')
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit(f'SExtractor {mode} pass failed (rc={r.returncode})')
    print(f'  wall: {time.time()-t0:.1f}s')
    return out_cat, out_seg


def merge_catalogs(cold_cat: Path, hot_cat: Path,
                   out_path: Path) -> tuple[int, int, int]:
    """Merge: keep all cold; add hot whose centroid lies outside every
    cold Kron ellipse."""
    cold = fits.open(cold_cat)[2].data
    hot  = fits.open(hot_cat)[2].data
    n_cold = len(cold)
    n_hot  = len(hot)

    if n_hot == 0:
        print('  no hot sources — merged = cold only')
        fits.HDUList([fits.PrimaryHDU(), fits.open(cold_cat)[1],
                      fits.open(cold_cat)[2]]).writeto(out_path, overwrite=True)
        return n_cold, 0, n_cold

    # Cold ellipse parameters
    xc, yc   = cold['X_IMAGE'].astype(np.float64), cold['Y_IMAGE'].astype(np.float64)
    a, b, th = (cold['A_IMAGE'].astype(np.float64),
                cold['B_IMAGE'].astype(np.float64),
                np.deg2rad(cold['THETA_IMAGE'].astype(np.float64)))
    # KRON_RADIUS not in default_pass1.param; use FLUX_RADIUS scaling as proxy
    # — typical Kron scaling factor ~2.5
    if 'KRON_RADIUS' in cold.dtype.names:
        kron = cold['KRON_RADIUS'].astype(np.float64)
    else:
        kron = np.full(n_cold, 2.5)        # fall-back

    # For each hot source, test inside any cold ellipse
    hx = hot['X_IMAGE'].astype(np.float64)
    hy = hot['Y_IMAGE'].astype(np.float64)
    inside_any = np.zeros(n_hot, dtype=bool)

    # Vectorised batch over hot sources: for each cold ellipse, check all
    # hot sources at once (vectorised), accumulate inside_any.
    for i in range(n_cold):
        dx = hx - xc[i]
        dy = hy - yc[i]
        # rotate into ellipse frame
        cos_t, sin_t = np.cos(th[i]), np.sin(th[i])
        x_p =  cos_t*dx + sin_t*dy
        y_p = -sin_t*dx + cos_t*dy
        a_kron = a[i] * kron[i]
        b_kron = b[i] * kron[i]
        if a_kron <= 0 or b_kron <= 0:
            continue
        inside = (x_p/a_kron)**2 + (y_p/b_kron)**2 < 1.0
        inside_any |= inside

    keep_hot = ~inside_any
    n_kept = keep_hot.sum()
    print(f'  cold = {n_cold:,};  hot = {n_hot:,};  hot kept = {n_kept:,}')
    # Merge into a single FITS_LDAC catalog
    from astropy.table import Table, vstack
    cold_t = Table(cold)
    cold_t['DETECT_MODE'] = ['cold'] * n_cold
    hot_t = Table(hot[keep_hot])
    hot_t['DETECT_MODE'] = ['hot'] * n_kept
    merged = vstack([cold_t, hot_t])
    merged['SOURCE_ID'] = np.arange(1, len(merged) + 1)
    merged.write(out_path, format='fits', overwrite=True)
    print(f'  [save] {out_path}  ({len(merged):,} sources)')
    return n_cold, n_kept, n_cold + n_kept


def main():
    args = parse_args()
    work = WORK / args.tile
    work.mkdir(parents=True, exist_ok=True)

    print(f'Project   : {PROJECT}')
    print(f'Configs   : {CONFIGS}')
    print(f'Workdir   : {work}')
    print(f'Tile      : {args.tile}    target band: {args.target_band.upper()}')
    print(f'PSF FWHM lookup: ' + ', '.join(
        f'{b}={FWHM_ARCSEC[b]:.3f}\"' for b in FWHM_ARCSEC))

    # 1. χ²+ detection image
    step('1. Build χ²+ detection image (4-band stack)')
    chi2_path = build_chi2(args.tile, args.cutout_size, args.target_band, work)

    # 2 + 3.  SExtractor cold + hot
    if not args.skip_cold:
        step('2. SExtractor cold pass')
        cold_cat, _ = run_sex_mode(chi2_path, 'cold', work, args.tile)
    else:
        cold_cat = work / f'cat_cold_{args.tile}.fits'
    if not args.skip_hot:
        step('3. SExtractor hot pass')
        hot_cat, _ = run_sex_mode(chi2_path, 'hot',  work, args.tile)
    else:
        hot_cat  = work / f'cat_hot_{args.tile}.fits'

    # 4. Merge
    step('4. Merge cold + hot via Kron-ellipse criterion')
    merged_path = work / f'merged_{args.tile}.fits'
    n_cold, n_hot_kept, n_total = merge_catalogs(cold_cat, hot_cat, merged_path)

    print(f'\n=== detect complete ===')
    print(f'  cold sources              : {n_cold:>7,}')
    print(f'  hot kept (outside cold)   : {n_hot_kept:>7,}')
    print(f'  total merged              : {n_total:>7,}')
    print(f'  merged catalog            : {merged_path}')


if __name__ == '__main__':
    main()
