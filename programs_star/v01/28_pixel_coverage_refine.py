#!/usr/bin/env python
"""
28_pixel_coverage_refine.py  —  Deferred I/O step (NOT run as part of
the v01 autonomous pass).

Build a downsampled binary footprint mask per (band, tile) that records
where the SCI is finite & non-zero — i.e., actual pixel coverage as
opposed to the bbox-only union written by 20_footprints.py.  This
captures chip gaps and rotated-frame corners.

The mask is at e.g. 1/64 resolution → small enough to read in one shot
yet preserves chip-gap geometry (a 100-pixel gap at 30 mas pixel scale
is 3″, which spans ~6 mask pixels at 1/64 resolution).

Then re-evaluate the "real_orphan" column on csvfiles_star/orphans.parquet
using the downsampled mask.

Run with:
    python 28_pixel_coverage_refine.py

Expected wall time: ~10–30 min (reads every JWST + HST SCI tile once).
"""
from __future__ import annotations
import warnings
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

N_WORKERS = 4

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
MASKS_DIR = OUT / 'footprints' / 'masks'
MASKS_DIR.mkdir(parents=True, exist_ok=True)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')

DOWNSAMPLE = 64  # mask resolution = 1/DOWNSAMPLE of native pixel grid


def _path_for(mission: str, band: str, tile: str) -> Path:
    if mission == 'JWST':
        if tile == 'A10' and band == 'F115W':
            return JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
        return JWST_DIR / f'mosaic_nircam_{band.lower()}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'
    if mission == 'HST':
        return HST_DIR / f'acs_I_030mas_{tile}_sci.fits'
    return None  # Euclid not refined in v01


def _read_2d_full(path: Path):
    """Sequential read (not memmap) → fast even for strided downsample."""
    with fits.open(path, memmap=False) as h:
        for hdu in h:
            if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                return hdu.data, hdu.header.copy()
    return None, None


def build_mask(mission: str, band: str, tile: str):
    p = _path_for(mission, band, tile)
    if p is None or not p.exists():
        return None
    data, hdr = _read_2d_full(p)
    if data is None:
        return None
    sub = data[::DOWNSAMPLE, ::DOWNSAMPLE]
    mask = (np.isfinite(sub) & (sub != 0)).astype(np.uint8)
    return mask, hdr


def _build_one(args):
    mission, band, tile, out_npz = args
    if Path(out_npz).exists():
        return dict(mission=mission, band=band, tile=tile, status='cached')
    t0 = time.time()
    try:
        res = build_mask(mission, band, tile)
        if res is None:
            return dict(mission=mission, band=band, tile=tile, status='missing', t=time.time()-t0)
        mask, hdr = res
        np.savez_compressed(
            out_npz,
            mask=mask,
            downsample=DOWNSAMPLE,
            header=np.array(hdr.tostring(), dtype='S'),
            n1=hdr.get('NAXIS1') or hdr.get('ZNAXIS1'),
            n2=hdr.get('NAXIS2') or hdr.get('ZNAXIS2'),
        )
        return dict(mission=mission, band=band, tile=tile, status='ok',
                    shape=mask.shape, cov=float(mask.mean()), t=time.time()-t0)
    except Exception as e:
        return dict(mission=mission, band=band, tile=tile, status=f'ERR:{e}', t=time.time()-t0)


def main():
    tdf = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    # Only HST F814W + JWST F115W in v01
    sel = ((tdf['mission'] == 'HST')  & (tdf['band'] == 'F814W')) | \
          ((tdf['mission'] == 'JWST') & (tdf['band'] == 'F115W'))
    work = tdf[sel].reset_index(drop=True)
    print(f'Building masks for {len(work)} tiles at 1/{DOWNSAMPLE} resolution')

    jobs = []
    for r in work.itertuples():
        out_npz = MASKS_DIR / f'mask_{r.mission}_{r.band}_{r.tile}.npz'
        jobs.append((r.mission, r.band, r.tile, str(out_npz)))

    t_start = time.time()
    ctx = get_context('spawn')
    results = []
    with ctx.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_build_one, jobs, chunksize=1)):
            results.append(r)
            done = i + 1
            elapsed = time.time() - t_start
            eta = elapsed * (len(jobs) - done) / max(done, 1)
            cov = r.get('cov', None)
            cov_s = f'  cov={cov:.2%}' if cov is not None else ''
            print(f'  [{done:>3d}/{len(jobs)}] {r["mission"]:5s} {r["band"]:6s} {r["tile"]:>5s} '
                  f'{r["status"]:<10s}{cov_s}   (elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min)',
                  flush=True)

    # Now re-evaluate orphans
    print('\nRefining orphans.parquet using masks...')
    orph = pd.read_parquet(OUT / 'orphans.parquet')
    ra  = orph['ra'].values
    dec = orph['dec'].values

    def check_band(target_mission, target_band):
        hits = np.zeros(len(orph), dtype=bool)
        # Group orphans by which tile bbox they fall into
        ts = work[(work['mission'] == target_mission) & (work['band'] == target_band)]
        if len(ts) == 0:
            return hits
        cand_tile = np.full(len(ra), -1, dtype=int)
        polys = []
        for i, r in enumerate(ts.itertuples()):
            ra_min, ra_max = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4), max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
            dec_min, dec_max = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4), max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
            in_box = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            cand_tile[in_box & (cand_tile < 0)] = i
            polys.append(r)

        for ti in np.unique(cand_tile):
            if ti < 0:
                continue
            tr = polys[ti]
            out_npz = MASKS_DIR / f'mask_{tr.mission}_{tr.band}_{tr.tile}.npz'
            if not out_npz.exists():
                continue
            arc = np.load(out_npz)
            mask = arc['mask']
            hdr_str = arc['header'].item()
            from astropy.io.fits import Header
            hdr = Header.fromstring(hdr_str.decode('ascii') if isinstance(hdr_str, bytes) else hdr_str)
            w = WCS(hdr)
            sel = cand_tile == ti
            x, y = w.all_world2pix(ra[sel], dec[sel], 0)
            xi = (x / DOWNSAMPLE).astype(int)
            yi = (y / DOWNSAMPLE).astype(int)
            ok = (xi >= 0) & (xi < mask.shape[1]) & (yi >= 0) & (yi < mask.shape[0])
            tile_hit = np.zeros(sel.sum(), dtype=bool)
            tile_hit[ok] = mask[yi[ok], xi[ok]] != 0
            hits[np.where(sel)[0]] = tile_hit
        return hits

    orph['in_jwst_pix'] = check_band('JWST', 'F115W')
    orph['in_hst_pix']  = check_band('HST',  'F814W')

    orph['real_orphan'] = (
        (orph['missing_jwst']   & orph['in_jwst_pix']) |
        (orph['missing_hst']    & orph['in_hst_pix'])  |
        (orph['missing_euclid'] & orph['in_euclid_fp'])
    )
    out_pq = OUT / 'orphans_refined.parquet'
    orph.to_parquet(out_pq, index=False)
    print(f'Wrote {out_pq}')
    print(f'  real_orphan after refinement: {int(orph["real_orphan"].sum()):,}')


if __name__ == '__main__':
    main()
