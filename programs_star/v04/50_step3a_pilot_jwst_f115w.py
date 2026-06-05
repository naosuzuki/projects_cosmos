#!/usr/bin/env python
"""
50_step3a_pilot_jwst_f115w.py — Step 3a pilot.

Validate the SExtractor → PSFEx → SExtractor PSF-photometry chain on
a 2K×2K cutout of one JWST/NIRCam F115W tile.  Confirms:
  - Configs in /Users/suzuki/github/projects_cosmos/configs/ work
  - PIXAR_SR → AB ZP conversion is right for COSMOS-Web 30mas
  - PSFEx builds a sensible PSF model
  - SExtractor PSF photometry returns CHI2_PSF, SPREAD_MODEL
  - DAOPHOT-equivalent shape metrics emerge (we use SExtractor's
    A/B/THETA + ELLIPTICITY + FWHM_IMAGE; DAOPHOT integration is
    a later substep).

Outputs go to /Volumes/exdisk1/data/photometry_v04/pilot/ — kept out
of the source tree because they're regenerable.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
WORK     = Path('/Volumes/exdisk1/data/photometry_v04/pilot')
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tile', default='A4',
                   help='COSMOS-Web tile letter (e.g. A1, A4, B3). Default A4.')
    p.add_argument('--filter', default='f115w', choices=['f115w','f150w','f277w','f444w'])
    p.add_argument('--cutout-size', type=int, default=2000,
                   help='Cutout size in pixels (default 2000 = 60\" at 30 mas).')
    p.add_argument('--skip-extract', action='store_true',
                   help='Reuse existing cutout/catalog if present (debug).')
    return p.parse_args()


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


def make_cutout(args, work: Path) -> tuple[Path, Path]:
    """Slice a 2K cutout of SCI + WHT from the COSMOS-Web tile and write
    a 2-HDU FITS file (PRIMARY+SCI / WHT) the SExtractor pipeline can
    read."""
    src = JWST_DIR / (f'mosaic_nircam_{args.filter}_COSMOS-Web_30mas_'
                      f'{args.tile}_v1.0_i2d.fits')
    if not src.exists():
        sys.exit(f'Source tile not found: {src}')
    print(f'Source : {src.name}  ({src.stat().st_size/1e9:.1f} GB)')
    cut_sci  = work / f'cutout_{args.filter}_{args.tile}_sci.fits'
    cut_wht  = work / f'cutout_{args.filter}_{args.tile}_wht.fits'
    if args.skip_extract and cut_sci.exists() and cut_wht.exists():
        print('  → skipping extract (cutouts present)')
        return cut_sci, cut_wht
    t0 = time.time()
    with fits.open(src) as h:
        primary = h[0].header.copy()
        sci_hdr = h['SCI'].header
        wht_hdr = h['WHT'].header
        nx, ny  = sci_hdr['NAXIS1'], sci_hdr['NAXIS2']
        # Center cutout
        size = args.cutout_size
        cx, cy = nx//2, ny//2
        x0, x1 = cx - size//2, cx + size//2
        y0, y1 = cy - size//2, cy + size//2
        print(f'Cutout : [{x0}:{x1}, {y0}:{y1}]  '
              f'= {x1-x0}×{y1-y0} px at 30 mas → '
              f'{(x1-x0)*0.030:.0f}″ × {(y1-y0)*0.030:.0f}″')
        sci_data = h['SCI'].data[y0:y1, x0:x1].copy()
        wht_data = h['WHT'].data[y0:y1, x0:x1].copy()
    # Patch the WCS reference pixel so the cutout's WCS stays valid.
    # WHT typically inherits WCS keys from SCI rather than carrying its
    # own copy — patch SCI unconditionally, WHT only when present.
    for hdr in (sci_hdr, wht_hdr):
        if 'CRPIX1' in hdr:
            hdr['CRPIX1'] = hdr['CRPIX1'] - x0
        if 'CRPIX2' in hdr:
            hdr['CRPIX2'] = hdr['CRPIX2'] - y0
        hdr['NAXIS1'] = x1 - x0
        hdr['NAXIS2'] = y1 - y0
    # If WHT lacks WCS, copy from SCI so SExtractor's WEIGHT_IMAGE
    # match stays consistent (it doesn't need WCS but it shouldn't hurt).
    if 'CRPIX1' not in wht_hdr:
        for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2',
                  'CD1_1','CD1_2','CD2_1','CD2_2',
                  'CTYPE1','CTYPE2','CUNIT1','CUNIT2'):
            if k in sci_hdr:
                wht_hdr[k] = sci_hdr[k]
    # Compute AB zeropoint from PIXAR_SR (primary header keyword)
    pixar_sr = primary.get('PIXAR_SR') or sci_hdr.get('PIXAR_SR')
    if pixar_sr is None:
        sys.exit('PIXAR_SR not found in header — cannot compute AB ZP.')
    # AB mag = -2.5 log10(flux_Jy) + 8.9; flux_Jy = MJy/sr × sr × 1e6
    zp_ab = -2.5 * np.log10(pixar_sr * 1e6) + 8.9
    primary['ZP_AB'] = (zp_ab, 'AB zeropoint derived from PIXAR_SR')
    print(f'PIXAR_SR = {pixar_sr:.4e}  →  ZP_AB = {zp_ab:.3f}')
    # Write SCI cutout (primary kept; SCI as imageHDU for SExtractor)
    hdul = fits.HDUList([
        fits.PrimaryHDU(data=sci_data, header=sci_hdr),
    ])
    hdul[0].header.update({k: primary[k] for k in ('ZP_AB',) if k in primary})
    hdul.writeto(cut_sci, overwrite=True)
    # Write WHT in its own file (SExtractor takes WEIGHT_IMAGE as a path)
    fits.PrimaryHDU(data=wht_data, header=wht_hdr).writeto(
        cut_wht, overwrite=True)
    print(f'  SCI cutout : {cut_sci}  ({cut_sci.stat().st_size/1e6:.1f} MB)')
    print(f'  WHT cutout : {cut_wht}  ({cut_wht.stat().st_size/1e6:.1f} MB)')
    print(f'  cutout wall: {time.time()-t0:.1f}s')
    return cut_sci, cut_wht


def run_sex(sci: Path, wht: Path, out_cat: Path,
            psf: Path | None = None) -> None:
    cfg = CONFIGS / 'jwst_nircam_f115w.sex'
    zp  = float(fits.getheader(sci)['ZP_AB'])
    # Pass 1 uses default_pass1.param (no PSF columns, has VIGNET for PSFEx).
    # Pass 2 uses default_pass2.param (PSF columns, no VIGNET).
    param = CONFIGS / ('default_pass2.param' if psf is not None
                       else 'default_pass1.param')
    cmd = [
        'sex', str(sci),
        '-c',                str(cfg),
        '-CATALOG_NAME',     str(out_cat),
        '-PARAMETERS_NAME',  str(param),
        '-FILTER_NAME',      str(CONFIGS / 'default.conv'),
        '-STARNNW_NAME',     str(CONFIGS / 'default.nnw'),
        '-WEIGHT_IMAGE',     str(wht),
        '-MAG_ZEROPOINT',    f'{zp:.4f}',
    ]
    if psf is not None:
        cmd += ['-PSF_NAME', str(psf)]
    print(f'  cmd: sex … -c {cfg.name} -PARAMETERS_NAME {param.name} '
          f'→ {out_cat.name}'
          + (f' (PSF={psf.name})' if psf else ''))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit(f'SExtractor failed (rc={r.returncode})')
    print(f'  wall : {time.time()-t0:.1f}s')


def run_psfex(in_cat: Path) -> Path:
    cfg = CONFIGS / 'default.psfex'
    cmd = ['psfex', str(in_cat), '-c', str(cfg)]
    print(f'  cmd: psfex {in_cat.name} -c {cfg.name}')
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit(f'PSFEx failed (rc={r.returncode})')
    print(f'  wall : {time.time()-t0:.1f}s')
    psf_path = in_cat.with_suffix('.psf')
    if not psf_path.exists():
        # psfex sometimes writes to cwd; check
        alt = Path.cwd() / psf_path.name
        if alt.exists():
            shutil.move(alt, psf_path)
    return psf_path


def summarize(cat_path: Path) -> None:
    with fits.open(cat_path) as h:
        tbl = h[2].data    # LDAC: HDU0 primary, HDU1 header table, HDU2 sources
    n = len(tbl)
    print(f'  rows   : {n:,}')
    if n == 0:
        return
    cols = tbl.columns.names
    print(f'  cols   : {len(cols)}  e.g. {cols[:8]} ...')
    if 'MAG_AUTO' in cols:
        m = tbl['MAG_AUTO']
        ok = np.isfinite(m) & (m > 0) & (m < 99)
        if ok.any():
            mq = np.quantile(m[ok], [0.05, 0.5, 0.95])
            print(f'  MAG_AUTO 5/50/95% : {mq[0]:.2f} / {mq[1]:.2f} / {mq[2]:.2f}')
    if 'CHI2_PSF' in cols:
        c = tbl['CHI2_PSF']
        ok = np.isfinite(c) & (c > 0) & (c < 1e6)
        if ok.any():
            cq = np.quantile(c[ok], [0.05, 0.5, 0.95])
            print(f'  CHI2_PSF 5/50/95% : {cq[0]:.2f} / {cq[1]:.2f} / {cq[2]:.2f}')
    if 'CLASS_STAR' in cols:
        cs = tbl['CLASS_STAR']
        n_star = (cs > 0.9).sum()
        n_gal  = (cs < 0.1).sum()
        print(f'  CLASS_STAR > 0.9 : {n_star:,}   < 0.1 : {n_gal:,}')


def main():
    args = parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    print(f'Project  : {PROJECT}')
    print(f'Configs  : {CONFIGS}')
    print(f'Workdir  : {WORK}')
    print(f'Tile     : {args.tile}  filter {args.filter.upper()}')

    # 1. Cutout
    step('1. Build cutout')
    sci, wht = make_cutout(args, WORK)

    # 2. SExtractor pass 1 (detection + raw photometry)
    step('2. SExtractor pass 1 (detect, no PSF model)')
    cat1 = WORK / f'cat_pass1_{args.filter}_{args.tile}.fits'
    run_sex(sci, wht, cat1)
    summarize(cat1)

    # 3. PSFEx (PSF model from pass-1 catalog)
    step('3. PSFEx (build PSF model)')
    psf = run_psfex(cat1)
    if psf.exists():
        with fits.open(psf) as h:
            ph = h[1].header
            fwhm = ph.get('PSF_FWHM')
            samp = ph.get('PSF_SAMP')
            naxis= ph.get('PSF_NAXIS') or ph.get('NAXIS')
            print(f'  PSF model : '
                  + (f'FWHM {fwhm:.3f} pix' if fwhm else 'FWHM n/a')
                  + (f', sampling {samp:.3f}' if samp else '')
                  + (f', NAXIS={naxis}' if naxis else ''))
            print(f'  PSF file  : {psf}')
    else:
        print(f'  WARNING: PSF file not found at {psf}')

    # 4. SExtractor pass 2 (PSF photometry)
    step('4. SExtractor pass 2 (with PSF model)')
    cat2 = WORK / f'cat_pass2_{args.filter}_{args.tile}.fits'
    run_sex(sci, wht, cat2, psf=psf)
    summarize(cat2)

    print('\n=== pilot complete ===')


if __name__ == '__main__':
    main()
