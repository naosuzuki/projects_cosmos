#!/usr/bin/env python
"""
21_dao_detect.py  —  Step 2: DAOStarFinder on every band-tile mosaic.

For every mosaic enumerated in csvfiles_star/footprints/tile_polygons.csv:
  * Read SCI HDU (header WCS already cached in step 1, re-read here per-file).
  * Sigma-clipped background median/std on a 5× downsampled subset.
  * DAOStarFinder(threshold=5σ, fwhm=PSF_FWHM_pix, sharp [0.2, 1.0],
                  round [-1, 1])  ← permissive; tight gates land in step 3.
  * Convert x/y → ICRS ra/dec via WCS.
  * Write per-(band, tile) parquet to csvfiles_star/dao/<band>/<band>_<tile>.parquet.

Per-band aggregated catalogs are concatenated at the end:
  csvfiles_star/dao/dao_<band>.parquet
"""
from __future__ import annotations
import warnings
import time
import re
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
DAO  = ROOT / 'csvfiles_star' / 'dao'
FOOT = ROOT / 'csvfiles_star' / 'footprints' / 'tile_polygons.csv'
DAO.mkdir(parents=True, exist_ok=True)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')


def enumerate_files():
    """Identical to 20_footprints.enumerate_files() — inlined to avoid
    importing a numeric-prefixed module."""
    files = []
    pat = re.compile(r'acs_I_030mas_(\d+)_sci\.fits$')
    for p in sorted(HST_DIR.glob('acs_I_030mas_*_sci.fits')):
        m = pat.search(p.name)
        if m:
            files.append(dict(mission='HST',  band='F814W', tile=m.group(1), path=str(p)))
    jwst_pat = re.compile(r'mosaic_nircam_(f\d+w)_COSMOS-Web_30mas_([AB]\d+)_v1\.0_i2d\.fits$')
    for p in sorted(JWST_DIR.glob('mosaic_nircam_*_i2d.fits')):
        m = jwst_pat.search(p.name)
        if m:
            files.append(dict(mission='JWST', band=m.group(1).upper(), tile=m.group(2), path=str(p)))
    a10 = JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    if a10.exists():
        files.append(dict(mission='JWST', band='F115W', tile='A10', path=str(a10)))
    euc_pat = re.compile(r'EUC_MER_BGSUB-MOSAIC-([A-Z\-]+)_TILE(\d+)-([0-9A-F]+)_(\d{8}T\d{6}\.\d+Z)_00\.00\.fits$')
    latest = {}
    for p in sorted(EUCLID_DIR.glob('*.fits')):
        m = euc_pat.search(p.name)
        if not m: continue
        band, tile, hsh, ts = m.group(1), m.group(2), m.group(3), m.group(4)
        key = (band, tile)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, p)
    for (band, tile), (ts, p) in latest.items():
        files.append(dict(mission='Euclid', band=band, tile=tile, path=str(p)))
    return files

# Master §5 empirical PSF FWHM (arcsec)
FWHM_AS = dict(
    F814W=0.134, F115W=0.057, F150W=0.057, F277W=0.130, F444W=0.160,
    VIS=0.194, NIR_Y=0.524, NIR_J=0.537, NIR_H=0.567,
)

# Map filesystem band names → output keys
BAND_KEY = {
    'F814W': 'F814W',
    'F115W': 'F115W', 'F150W': 'F150W', 'F277W': 'F277W', 'F444W': 'F444W',
    'VIS': 'VIS', 'NIR-Y': 'NIR_Y', 'NIR-J': 'NIR_J', 'NIR-H': 'NIR_H',
}

THRESHOLD_SIGMA = 5.0
N_WORKERS = 6


def detect_one(args):
    """Worker: DAO on one mosaic. Returns dict with summary; writes parquet."""
    mission, band, tile, path = args
    band_key = BAND_KEY[band]
    out_dir = DAO / band_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{band_key}_{tile}.parquet'
    if out_path.exists():
        try:
            return dict(mission=mission, band=band_key, tile=tile,
                        n_src=len(pd.read_parquet(out_path)),
                        t_sec=0.0, status='cached', path=str(out_path))
        except Exception:
            pass  # corrupt; re-do

    t0 = time.time()
    try:
        with fits.open(path, memmap=False) as h:
            # find first 2D ImageHDU with WCS
            data = None
            hdr  = None
            for hdu in h:
                hh = hdu.header
                if hh.get('NAXIS', 0) == 2 and hdu.data is not None:
                    data = hdu.data
                    hdr  = hh
                    break
            if data is None:
                return dict(mission=mission, band=band_key, tile=tile,
                            n_src=0, t_sec=0, status='no-data', path=path)
            w = WCS(hdr)
        cdelt = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0  # arcsec
        fwhm_pix = FWHM_AS[band_key] / cdelt

        mask = ~np.isfinite(data) | (data == 0)
        sub  = data[::5, ::5]
        smask = np.isfinite(sub) & (sub != 0)
        if smask.sum() < 1000:
            return dict(mission=mission, band=band_key, tile=tile,
                        n_src=0, t_sec=time.time()-t0, status='empty', path=path)
        _, med, std = sigma_clipped_stats(sub[smask], sigma=3.0, maxiters=3)

        finder = DAOStarFinder(
            threshold=THRESHOLD_SIGMA * std,
            fwhm=fwhm_pix,
            sharplo=0.2, sharphi=1.0,
            roundlo=-1.0, roundhi=1.0,
        )
        src = finder(data - med, mask=mask)
        if src is None or len(src) == 0:
            df = pd.DataFrame(columns=['x','y','ra','dec','sharpness','roundness1','roundness2',
                                       'npix','peak','flux','mag','sky_med','sky_std','fwhm_pix','tile'])
            df.to_parquet(out_path)
            return dict(mission=mission, band=band_key, tile=tile,
                        n_src=0, t_sec=time.time()-t0, status='no-src', path=str(out_path))

        x = src['xcentroid'].data.astype(np.float32)
        y = src['ycentroid'].data.astype(np.float32)
        ra, dec = w.all_pix2world(x, y, 0)
        df = pd.DataFrame({
            'x': x, 'y': y,
            'ra':  ra.astype(np.float64), 'dec': dec.astype(np.float64),
            'sharpness':  src['sharpness'].data.astype(np.float32),
            'roundness1': src['roundness1'].data.astype(np.float32),
            'roundness2': src['roundness2'].data.astype(np.float32),
            'npix':       src['npix'].data.astype(np.int32),
            'peak':       src['peak'].data.astype(np.float32),
            'flux':       src['flux'].data.astype(np.float32),
            'mag':        src['mag'].data.astype(np.float32),
        })
        df['sky_med']  = np.float32(med)
        df['sky_std']  = np.float32(std)
        df['fwhm_pix'] = np.float32(fwhm_pix)
        df['tile']     = tile
        df.to_parquet(out_path, index=False)

        return dict(mission=mission, band=band_key, tile=tile,
                    n_src=len(df), t_sec=time.time()-t0, status='ok', path=str(out_path))
    except Exception as e:
        return dict(mission=mission, band=band_key, tile=tile, n_src=0,
                    t_sec=time.time()-t0, status=f'ERR:{type(e).__name__}:{e}', path=path)


def main():
    files = enumerate_files()
    files.sort(key=lambda f: (f['mission'], f['band'], f['tile']))
    jobs = [(f['mission'], f['band'], f['tile'], f['path']) for f in files]
    print(f'Queued {len(jobs)} (band,tile) jobs across {N_WORKERS} workers')

    t_start = time.time()
    results = []
    ctx = get_context('spawn')
    with ctx.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(detect_one, jobs, chunksize=1)):
            results.append(r)
            done = i + 1
            elapsed = time.time() - t_start
            eta = elapsed * (len(jobs) - done) / max(done, 1)
            print(f'  [{done:>3d}/{len(jobs)}]  {r["mission"]:6s} {r["band"]:6s} {r["tile"]:>5s}  '
                  f'n={r["n_src"]:>6d}  {r["t_sec"]:>5.1f}s  {r["status"]:<10s}  '
                  f'(elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min)', flush=True)

    rdf = pd.DataFrame(results)
    rpath = DAO / 'detection_summary.csv'
    rdf.to_csv(rpath, index=False)
    print(f'\nWrote summary {rpath}')

    # Concatenate per band
    print('\nAggregating per-band parquets:')
    for band in ['F814W','F115W','F150W','F277W','F444W','VIS','NIR_Y','NIR_J','NIR_H']:
        bdir = DAO / band
        if not bdir.exists():
            continue
        parts = sorted(bdir.glob(f'{band}_*.parquet'))
        if not parts:
            continue
        big = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        outp = DAO / f'dao_{band}.parquet'
        big.to_parquet(outp, index=False)
        print(f'  {band:<8} {len(parts):3d} tiles → {len(big):>10,d} sources  → {outp}')


if __name__ == '__main__':
    main()
