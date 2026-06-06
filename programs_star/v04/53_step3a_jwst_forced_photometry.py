#!/usr/bin/env python
"""
53_step3a_jwst_forced_photometry.py — forced dual-image PSF photometry
for the 4 JWST/NIRCam bands at the positions detected on the χ²+ stack
(51_step3a_jwst_chi2_detect.py).

Approach (option B, COSMOS-Web style):
  For each band b in {f115w, f150w, f277w, f444w}:
    For mode m in {cold, hot}:
      1. SExtractor DUAL-IMAGE pass 1:
           detection image = chi2_<tile>.fits
           measurement image = native band image
           → catalog with VIGNET sliced from the BAND image (for PSFEx)
      2. PSFEx on that catalog → band PSF model (per mode)
      3. SExtractor DUAL-IMAGE pass 2:
           same detection (chi2) + measurement (band) + PSF model
           → FLUX_PSF, MAG_PSF, CHI2_PSF, SPREAD_MODEL per source
    The detection on chi2 with the cold (or hot) config reproduces
    exactly the same source list (and NUMBER ordering) as 51_'s cold
    (or hot) pass, because it is the same image + same config.

  Then JOIN the per-band pass-2 catalogs to 51_'s merged catalog by
  (DETECT_MODE, NUMBER), keeping only the sources that survived 51_'s
  Kron-ellipse merge.  Output is one row per merged source with PSF
  photometry in all 4 bands — the deliverable consumed by Step 3b's
  Gaia-PM-anchored star/galaxy classifier.

Because detection is identical across all 4 bands (driven by the same
χ² image), source lists are perfectly aligned — no positional
cross-matching is needed; the join is exact.

Outputs to
  /Volumes/exdisk1/data/photometry_v04/jwst_chi2/<tile>/forced/
    cat_<band>_<mode>_p2.fits     per band/mode forced PSF photometry
    forced_<tile>.fits            wide table: 1 row/merged source × 4 bands
    forced_<tile>.meta.json
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
from astropy.table import Table, join

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04/jwst_chi2')

BANDS = ['f115w', 'f150w', 'f277w', 'f444w']
MODES = ['cold', 'hot']

# PSFEx sample-selection per band (pixels at 30 mas).  Stars sit at
# the narrow end of the FWHM distribution; tight ranges isolate them.
PSFEX_FWHM = {
    'f115w': (1.5, 2.5),
    'f150w': (1.6, 2.8),
    'f277w': (2.0, 3.5),
    'f444w': (2.5, 4.5),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tile', default='A4')
    p.add_argument('--bands', nargs='+', default=BANDS, choices=BANDS)
    p.add_argument('--skip-existing', action='store_true',
                   help='Reuse per-band/mode pass-2 catalogs if present.')
    return p.parse_args()


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


def band_image(tile: str, band: str) -> Path:
    p = JWST_DIR / (f'mosaic_nircam_{band}_COSMOS-Web_30mas_'
                    f'{tile}_v1.0_i2d.fits')
    if not p.exists():
        sys.exit(f'Missing band image: {p}')
    return p


def band_zp(image: Path) -> float:
    """AB zeropoint from PIXAR_SR in the SCI header."""
    with fits.open(image) as h:
        for hdu in h:
            if 'PIXAR_SR' in hdu.header:
                return -2.5*np.log10(hdu.header['PIXAR_SR']*1e6) + 8.9
    sys.exit(f'No PIXAR_SR in {image}')


def extract_sci_wht(image: Path, tile: str, band: str, work: Path
                    ) -> tuple[Path, Path]:
    """COSMOS-Web i2d files are multi-extension; SExtractor dual-image
    mode wants plain single-HDU FITS.  Write SCI and WHT to scratch
    single-HDU files (full tile)."""
    sci_out = work / f'sci_{band}_{tile}.fits'
    wht_out = work / f'wht_{band}_{tile}.fits'
    if sci_out.exists() and wht_out.exists():
        return sci_out, wht_out
    t0 = time.time()
    with fits.open(image) as h:
        sci = h['SCI'].data
        wht = h['WHT'].data
        sci_hdr = h['SCI'].header
        wht_hdr = h['WHT'].header
    fits.PrimaryHDU(data=sci, header=sci_hdr).writeto(sci_out, overwrite=True)
    fits.PrimaryHDU(data=wht, header=wht_hdr).writeto(wht_out, overwrite=True)
    print(f'    extracted SCI/WHT for {band} ({time.time()-t0:.1f}s)')
    return sci_out, wht_out


def run_sex_dual(chi2: Path, band_sci: Path, band_wht: Path,
                 mode: str, out_cat: Path, param: str, zp: float,
                 psf: Path | None = None) -> None:
    """Dual-image SExtractor: detection on chi2, measurement on band."""
    cfg = CONFIGS / f'jwst_nircam_chi2_{mode}.sex'
    conv = ('tophat_9.0_9x9.conv' if mode == 'cold' else 'gauss_3.0_5x5.conv')
    cmd = [
        'sex', f'{chi2},{band_sci}',        # detection,measurement
        '-c',                str(cfg),
        '-CATALOG_NAME',     str(out_cat),
        '-PARAMETERS_NAME',  str(CONFIGS / param),
        '-STARNNW_NAME',     str(CONFIGS / 'default.nnw'),
        '-FILTER_NAME',      str(CONFIGS / conv),
        '-WEIGHT_TYPE',      'NONE,MAP_WEIGHT',     # det none, meas weight
        '-WEIGHT_IMAGE',     str(band_wht),
        '-MAG_ZEROPOINT',    f'{zp:.4f}',
        '-CHECKIMAGE_TYPE',  'NONE',
    ]
    if psf is not None:
        cmd += ['-PSF_NAME', str(psf)]
    print(f'    dual {mode}: det=chi2 meas={band_sci.name} '
          f'param={param}' + (f' PSF' if psf else ''), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(f'SExtractor dual {mode} failed (rc={r.returncode})')
    print(f'      wall {time.time()-t0:.1f}s', flush=True)


def run_psfex(in_cat: Path, band: str) -> Path:
    cfg = CONFIGS / 'default.psfex'
    lo, hi = PSFEX_FWHM[band]
    cmd = ['psfex', str(in_cat), '-c', str(cfg),
           '-SAMPLE_FWHMRANGE',  f'{lo},{hi}',
           '-SAMPLE_MAXELLIP',   '0.10',
           '-SAMPLE_MINSN',      '30',
           '-SAMPLE_AUTOSELECT', 'Y']
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(f'PSFEx failed for {band} (rc={r.returncode})')
    psf = in_cat.with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists():
            shutil.move(alt, psf)
    print(f'    PSFEx {band} wall {time.time()-t0:.1f}s', flush=True)
    return psf


def main():
    args = parse_args()
    tile = args.tile
    work = WORK / tile
    forced = work / 'forced'
    forced.mkdir(parents=True, exist_ok=True)

    chi2 = work / f'chi2_{tile}.fits'
    merged_path = work / f'merged_{tile}.fits'
    if not chi2.exists() or not merged_path.exists():
        sys.exit(f'Need {chi2.name} and {merged_path.name} from 51_ first.')

    merged = Table.read(merged_path)
    print(f'Tile        : {tile}')
    print(f'χ² image    : {chi2}')
    print(f'merged cat  : {len(merged):,} sources '
          f'({(merged["DETECT_MODE"]=="cold").sum():,} cold, '
          f'{(merged["DETECT_MODE"]=="hot").sum():,} hot)')
    print(f'bands       : {" ".join(args.bands)}')

    # Per-band photometry, joined to merged by (DETECT_MODE, NUMBER)
    band_tables = {}
    for band in args.bands:
        step(f'Band {band.upper()}')
        img = band_image(tile, band)
        zp  = band_zp(img)
        print(f'  ZP_AB = {zp:.4f}')
        sci, wht = extract_sci_wht(img, tile, band, work)

        per_mode = []
        for mode in MODES:
            p2 = forced / f'cat_{band}_{mode}_p2.fits'
            if args.skip_existing and p2.exists():
                print(f'  [{mode}] reuse {p2.name}')
            else:
                # pass 1 (VIGNET for PSFEx)
                p1 = forced / f'cat_{band}_{mode}_p1.fits'
                run_sex_dual(chi2, sci, wht, mode, p1,
                             'default_pass1.param', zp, psf=None)
                # PSFEx
                psf = run_psfex(p1, band)
                # pass 2 (PSF photometry)
                run_sex_dual(chi2, sci, wht, mode, p2,
                             'default_pass2.param', zp, psf=psf)
            t = Table.read(p2)
            t['DETECT_MODE'] = mode
            per_mode.append(t)

        from astropy.table import vstack
        bcat = vstack(per_mode, metadata_conflicts='silent')
        # keep only columns we need, prefix by band
        keep = ['NUMBER', 'DETECT_MODE',
                'FLUX_PSF', 'FLUXERR_PSF', 'MAG_PSF', 'MAGERR_PSF',
                'CHI2_PSF', 'SPREAD_MODEL', 'SPREADERR_MODEL',
                'FLUX_AUTO', 'MAG_AUTO', 'FLUX_RADIUS', 'ELLIPTICITY',
                'FWHM_IMAGE', 'CLASS_STAR', 'FLAGS']
        keep = [c for c in keep if c in bcat.colnames]
        bcat = bcat[keep]
        ren = {c: f'{band}_{c}' for c in keep
               if c not in ('NUMBER', 'DETECT_MODE')}
        for old, new in ren.items():
            bcat.rename_column(old, new)
        band_tables[band] = bcat
        print(f'  band catalog rows: {len(bcat):,}')

    # ── Join all bands onto the merged spine by (DETECT_MODE, NUMBER) ──
    step('Join 4 bands onto merged catalog')
    out = merged.copy()
    for band in args.bands:
        out = join(out, band_tables[band],
                   keys=['DETECT_MODE', 'NUMBER'], join_type='left')
        print(f'  + {band}: {len(out):,} rows')

    out_path = forced / f'forced_{tile}.fits'
    out.write(out_path, overwrite=True)
    print(f'\n[save] {out_path}  ({len(out):,} rows, {len(out.colnames)} cols)')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tile':            tile,
        'bands':           args.bands,
        'n_sources':       int(len(out)),
        'n_cold':          int((out['DETECT_MODE'] == 'cold').sum()),
        'n_hot':           int((out['DETECT_MODE'] == 'hot').sum()),
        'chi2_image':      str(chi2),
        'merged_catalog':  str(merged_path),
        'method':          'dual-image forced PSF photometry '
                           '(det=chi2 stack, meas=native band)',
    }
    (forced / f'forced_{tile}.meta.json').write_text(json.dumps(meta, indent=2))

    # Quick QA: CHI2_PSF per band for the brightest sources
    print('\n=== forced photometry complete ===')
    for band in args.bands:
        col = f'{band}_CHI2_PSF'
        mcol = f'{band}_MAG_PSF'
        if col in out.colnames and mcol in out.colnames:
            m = out[mcol]
            bright = (m > 0) & (m < 24) & np.isfinite(m)
            c = out[col][bright]
            c = c[np.isfinite(c) & (c > 0)]
            if len(c):
                q = np.quantile(c, [0.05, 0.5, 0.95])
                print(f'  {band} CHI2_PSF (MAG<24) 5/50/95 : '
                      f'{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}  '
                      f'(n={len(c):,})')


if __name__ == '__main__':
    main()
