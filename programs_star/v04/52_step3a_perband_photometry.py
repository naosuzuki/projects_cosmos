#!/usr/bin/env python
"""
52_step3a_perband_photometry.py — generalised per-(telescope, filter)
SExtractor + PSFEx + SExtractor PSF-photometry driver for v04 Step 3a.

This is the production successor to the JWST F115W pilot
(50_step3a_pilot_jwst_f115w.py).  It takes any SCI image (and optional
WHT) plus an instrument key like \`jwst_nircam_f115w\` and runs:
  1.  SExtractor pass 1 (detection + VIGNET for PSFEx) using
      \`configs/<instrument>.sex\` + \`default_pass1.param\`.
  2.  PSFEx with band-tuned sample selection (FWHM range,
      ellipticity, S/N) to build the per-band PSF model.
  3.  SExtractor pass 2 (PSF photometry) reusing the same instrument
      config but switching to \`default_pass2.param\` and loading the
      PSF model.

Wired separately from detection: the χ²+-stack hot+cold detection
catalog from 51_step3a_jwst_chi2_detect.py is the authoritative
source list across bands; per-band photometry catalogs from 52_ get
cross-matched to it downstream (53_).

Outputs go under
  /Volumes/exdisk1/data/photometry_v04/<instrument>/<tile_or_image_name>/
with files:
  cat_pass1.fits          detection catalog
  cat_pass1.psf           PSFEx model
  cat_pass2.fits          PSF-photometry catalog (the deliverable)
  meta.json               run summary
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

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')

# Per-(telescope, filter family) PSFEx sample-selection overrides.
# Tightens the stellar locus depending on PSF FWHM and image type.
# Format:  prefix → (FWHM_LO, FWHM_HI, MAXELLIP, MINSN)
# Units of FWHM are pixels at the band's native pixel scale.
PSFEX_TUNING = {
    'jwst_nircam':  (1.5, 3.0, 0.10, 30),
    'hst_acs':      (2.0, 4.0, 0.15, 30),
    'euclid_vis':   (1.2, 2.5, 0.15, 20),
    'euclid_nisp':  (1.0, 2.5, 0.20, 15),
    'hsc':          (2.5, 5.0, 0.15, 30),
    'sdss':         (2.5, 4.5, 0.20, 15),
    'ps1':          (3.0, 5.5, 0.15, 20),
    'lsdr10':       (3.0, 5.5, 0.15, 20),
    'galex':        (2.0, 5.0, 0.30, 10),
    'unwise':       (1.5, 4.0, 0.30, 10),
}


def lookup_psfex_tuning(instrument: str) -> tuple[float, float, float, float]:
    """Find the closest matching prefix in PSFEX_TUNING."""
    for k in sorted(PSFEX_TUNING, key=len, reverse=True):
        if instrument.startswith(k):
            return PSFEX_TUNING[k]
    return (1.5, 5.0, 0.20, 20)            # default fallback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', required=True,
                   help='Instrument key, e.g. "jwst_nircam_f115w" '
                        '(must match configs/<instrument>.sex)')
    p.add_argument('--image', required=True, type=Path,
                   help='Path to SCI image FITS.')
    p.add_argument('--weight', type=Path, default=None,
                   help='Path to WHT image FITS (optional; some configs need it).')
    p.add_argument('--zp-ab', type=float, default=None,
                   help='AB zeropoint; if absent, read from header keyword '
                        'ZP_AB or fall back to MAG_ZEROPOINT in the .sex config.')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='Custom output directory (default: '
                        '/Volumes/exdisk1/data/photometry_v04/<instrument>/<image-stem>).')
    p.add_argument('--skip-pass1', action='store_true')
    p.add_argument('--skip-psfex', action='store_true')
    p.add_argument('--skip-pass2', action='store_true')
    return p.parse_args()


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


def lookup_zp(image: Path, override: float | None) -> float:
    """Resolve the AB zeropoint to pass to SExtractor."""
    if override is not None:
        return override
    hdr = fits.getheader(image)
    for key in ('ZP_AB', 'ABMAG_ZP', 'MAGZP', 'MAG_ZP'):
        if key in hdr:
            return float(hdr[key])
    # JWST: derive from PIXAR_SR if present
    if 'PIXAR_SR' in hdr:
        zp = -2.5 * np.log10(hdr['PIXAR_SR'] * 1e6) + 8.9
        return float(zp)
    # Fall back to whatever the .sex config sets (SExtractor handles this)
    return float('nan')


def run_sex(sci: Path, weight: Path | None, out_cat: Path,
            instrument: str, param: str, zp: float,
            psf: Path | None = None) -> None:
    cfg = CONFIGS / f'{instrument}.sex'
    if not cfg.exists():
        sys.exit(f'No config at {cfg}.  Available: '
                 f'{[p.stem for p in CONFIGS.glob("*.sex")]}')
    cmd = [
        'sex', str(sci),
        '-c',                str(cfg),
        '-CATALOG_NAME',     str(out_cat),
        '-PARAMETERS_NAME',  str(CONFIGS / param),
        '-FILTER_NAME',      str(CONFIGS / 'default.conv'),
        '-STARNNW_NAME',     str(CONFIGS / 'default.nnw'),
    ]
    if weight is not None:
        cmd += ['-WEIGHT_IMAGE', str(weight)]
    if not np.isnan(zp):
        cmd += ['-MAG_ZEROPOINT', f'{zp:.4f}']
    if psf is not None:
        cmd += ['-PSF_NAME', str(psf)]
    print(f'  cmd: sex {sci.name} -c {cfg.name} -PARAMETERS_NAME {param} '
          f'→ {out_cat.name}' + (f' (PSF={psf.name})' if psf else ''))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit(f'SExtractor failed (rc={r.returncode})')
    print(f'  wall: {time.time()-t0:.1f}s')


def run_psfex(in_cat: Path, instrument: str) -> Path:
    cfg = CONFIGS / 'default.psfex'
    fwhm_lo, fwhm_hi, maxellip, minsn = lookup_psfex_tuning(instrument)
    cmd = ['psfex', str(in_cat), '-c', str(cfg),
           '-SAMPLE_FWHMRANGE',   f'{fwhm_lo},{fwhm_hi}',
           '-SAMPLE_MAXELLIP',    f'{maxellip}',
           '-SAMPLE_MINSN',       f'{minsn}',
           '-SAMPLE_VARIABILITY', '0.15',
           '-SAMPLE_AUTOSELECT',  'Y',
           ]
    print(f'  cmd: psfex {in_cat.name} ... '
          f'FWHM {fwhm_lo}-{fwhm_hi} ellip<{maxellip} SN>{minsn}')
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        sys.exit(f'PSFEx failed (rc={r.returncode})')
    print(f'  wall: {time.time()-t0:.1f}s')
    psf = in_cat.with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists():
            shutil.move(alt, psf)
    return psf


def summarize(cat: Path):
    with fits.open(cat) as h:
        tbl = h[2].data
    n = len(tbl)
    print(f'  rows  : {n:,}')
    if n == 0:
        return
    cols = tbl.columns.names
    for k in ('MAG_AUTO','MAG_PSF','CHI2_PSF','CLASS_STAR','SPREAD_MODEL'):
        if k in cols:
            v = tbl[k]
            ok = np.isfinite(v) & (v > -1e5) & (v < 1e5)
            if ok.any():
                q = np.quantile(v[ok], [0.05, 0.5, 0.95])
                print(f'  {k:13s} 5/50/95 : {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}')


def main():
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.exists():
        sys.exit(f'Image not found: {image}')
    weight = args.weight.expanduser().resolve() if args.weight else None
    if args.out_dir is None:
        out_dir = WORK / args.instrument / image.stem
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    zp = lookup_zp(image, args.zp_ab)
    print(f'Instrument : {args.instrument}')
    print(f'Config     : {CONFIGS / f"{args.instrument}.sex"}')
    print(f'Image      : {image}')
    print(f'Weight     : {weight}')
    print(f'Out dir    : {out_dir}')
    print(f'ZP (AB)    : {zp:.4f}')

    cat1 = out_dir / 'cat_pass1.fits'
    cat2 = out_dir / 'cat_pass2.fits'

    # 1. pass 1
    if not args.skip_pass1:
        step('1. SExtractor pass 1 (detection, no PSF)')
        run_sex(image, weight, cat1, args.instrument,
                'default_pass1.param', zp, psf=None)
        summarize(cat1)

    # 2. PSFEx
    if not args.skip_psfex:
        step('2. PSFEx (build PSF model)')
        psf = run_psfex(cat1, args.instrument)
        if psf.exists():
            with fits.open(psf) as h:
                ph = h[1].header
                fwhm = ph.get('PSF_FWHM')
                print(f'  PSF FWHM : {fwhm:.3f} pix' if fwhm else
                      '  PSF FWHM : n/a')
    else:
        psf = cat1.with_suffix('.psf')

    # 3. pass 2
    if not args.skip_pass2:
        step('3. SExtractor pass 2 (PSF photometry)')
        run_sex(image, weight, cat2, args.instrument,
                'default_pass2.param', zp, psf=psf)
        summarize(cat2)

    # meta
    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument':      args.instrument,
        'image_path':      str(image),
        'weight_path':     str(weight) if weight else None,
        'zp_ab':           float(zp),
        'cat_pass1':       str(cat1),
        'cat_pass2':       str(cat2),
        'psf_model':       str(cat1.with_suffix('.psf')),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n[save] {out_dir / "meta.json"}')
    print('\n=== per-band photometry complete ===')


if __name__ == '__main__':
    main()
