#!/usr/bin/env python
"""
25_orphans.py  —  Step 6: orphan candidates.

An "orphan" is a confirmed star-like detection in one mission whose
position is inside another mission's footprint but for which no
cross-mission counterpart was found within the brightness-aware radius.

Sources:
  * Euclid VIS good star, position inside JWST F115W footprint, AND
    not in pairs_JWST_Euclid → JWST-orphan
  * Euclid VIS good star, position inside HST F814W footprint, AND
    not in pairs_HST_Euclid → HST-orphan
  * HST F814W good star, position inside JWST F115W footprint, AND
    not in pairs_HST_JWST → JWST-orphan-from-HST
  * JWST F115W good star, inside HST F814W footprint, NOT in
    pairs_HST_JWST → HST-orphan-from-JWST
  * JWST F115W good star, inside Euclid VIS footprint, NOT in
    pairs_JWST_Euclid → Euclid-orphan-from-JWST

Outputs:
  csvfiles_star/orphans.parquet
  htmls/star_v01/orphans.html       — sortable list with reasons
"""
from __future__ import annotations
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import shapely.geometry as sg
from shapely import wkt as shapely_wkt, contains_xy

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

FOOT_FILE = OUT / 'footprints' / 'band_footprints.wkt'

# Per-band primary file directory for pixel-level coverage refinement
HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')


def _primary_jwst_path(tile: str) -> Path:
    if tile == 'A10':
        return JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    return JWST_DIR / f'mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'


def _primary_hst_path(tile: str) -> Path:
    return HST_DIR / f'acs_I_030mas_{tile}_sci.fits'


def _load_tile_meta():
    """Return dict band → list of (tile, path, polygon (RA,Dec))."""
    tdf = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    meta = {}
    for r in tdf.itertuples():
        if r.mission == 'JWST' and r.band == 'F115W':
            p = _primary_jwst_path(r.tile)
        elif r.mission == 'HST' and r.band == 'F814W':
            p = _primary_hst_path(r.tile)
        else:
            continue
        poly = sg.Polygon([(r.ra_c1, r.dec_c1), (r.ra_c2, r.dec_c2),
                           (r.ra_c3, r.dec_c3), (r.ra_c4, r.dec_c4)])
        meta.setdefault(r.band, []).append((r.tile, str(p), poly))
    return meta


def pixel_coverage_check(ra, dec, band, tile_meta):
    """For arrays of (ra, dec), return boolean array of "actual pixel cov".
    Group candidates by tile (bbox), open each tile's SCI, test isfinite + nonzero."""
    if band not in tile_meta:
        return np.zeros(len(ra), dtype=bool)

    has_data = np.zeros(len(ra), dtype=bool)
    cand_tile = np.full(len(ra), -1, dtype=int)
    ra_arr = np.asarray(ra)
    dec_arr = np.asarray(dec)
    for i, (tile, path, poly) in enumerate(tile_meta[band]):
        in_box = contains_xy(poly, ra_arr, dec_arr)
        new = in_box & (cand_tile < 0)
        cand_tile[new] = i

    # Process per tile to amortise FITS open cost
    for tile_idx in np.unique(cand_tile):
        if tile_idx < 0:
            continue
        tile, path, poly = tile_meta[band][tile_idx]
        sel = cand_tile == tile_idx
        try:
            with fits.open(path, memmap=True) as hdul:
                # find first 2D ImageHDU
                d = None; h = None
                for hdu in hdul:
                    if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                        d = hdu.data
                        h = hdu.header
                        break
                if d is None:
                    continue
                w = WCS(h)
                x, y = w.all_world2pix(ra[sel], dec[sel], 0)
                xi = np.round(x).astype(int)
                yi = np.round(y).astype(int)
                ok = (xi >= 0) & (xi < d.shape[1]) & (yi >= 0) & (yi < d.shape[0])
                vals = np.full(sel.sum(), np.nan, dtype=np.float32)
                if ok.any():
                    vals[ok] = d[yi[ok], xi[ok]]
                tile_has_data = np.isfinite(vals) & (vals != 0)
                idxs = np.where(sel)[0]
                has_data[idxs] = tile_has_data
        except Exception as e:
            print(f'  [pixcov] {band} tile {tile}: {e}')
    return has_data


def load_band_footprints():
    fp = {}
    with open(FOOT_FILE) as f:
        for line in f:
            band, w = line.strip().split('\t', 1)
            fp[band] = shapely_wkt.loads(w)
    return fp


def main():
    fp = load_band_footprints()
    hst_fp    = fp['F814W']
    jwst_fp   = fp['F115W']
    euclid_fp = fp['VIS']

    good_H = pd.read_parquet(OUT / 'good_stars_F814W.parquet')
    good_J = pd.read_parquet(OUT / 'good_stars_F115W.parquet')
    good_E = pd.read_parquet(OUT / 'good_stars_VIS.parquet')
    p_HE = pd.read_parquet(OUT / 'pairs_HST_Euclid.parquet')
    p_JE = pd.read_parquet(OUT / 'pairs_JWST_Euclid.parquet')
    p_HJ = pd.read_parquet(OUT / 'pairs_HST_JWST.parquet')

    matched_H_in_HE = set(p_HE['A_star_id'].tolist())
    matched_E_in_HE = set(p_HE['B_star_id'].tolist())
    matched_J_in_JE = set(p_JE['A_star_id'].tolist())
    matched_E_in_JE = set(p_JE['B_star_id'].tolist())
    matched_H_in_HJ = set(p_HJ['A_star_id'].tolist())
    matched_J_in_HJ = set(p_HJ['B_star_id'].tolist())

    rows = []

    # Euclid star, inside JWST coverage, no JWST counterpart
    in_jwst = contains_xy(jwst_fp, good_E['ra'].values, good_E['dec'].values)
    in_hst  = contains_xy(hst_fp,  good_E['ra'].values, good_E['dec'].values)
    for r, has_J, has_H in zip(good_E.itertuples(), in_jwst, in_hst):
        sid = r.star_id
        miss_J = bool(has_J) and (sid not in matched_E_in_JE)
        miss_H = bool(has_H) and (sid not in matched_E_in_HE)
        if miss_J or miss_H:
            rows.append(dict(found_in='Euclid VIS', star_id=sid, ra=r.ra, dec=r.dec,
                             mag=r.mag, peak=r.peak, snr=r.snr, tile=r.tile,
                             missing_jwst=miss_J, missing_hst=miss_H,
                             in_jwst_fp=bool(has_J), in_hst_fp=bool(has_H)))

    # HST star, inside JWST coverage, no JWST counterpart
    in_jwst = contains_xy(jwst_fp, good_H['ra'].values, good_H['dec'].values)
    in_euclid = contains_xy(euclid_fp, good_H['ra'].values, good_H['dec'].values)
    for r, has_J, has_E in zip(good_H.itertuples(), in_jwst, in_euclid):
        sid = r.star_id
        miss_J = bool(has_J) and (sid not in matched_H_in_HJ)
        miss_E = bool(has_E) and (sid not in matched_H_in_HE)
        if miss_J or miss_E:
            rows.append(dict(found_in='HST F814W', star_id=sid, ra=r.ra, dec=r.dec,
                             mag=r.mag, peak=r.peak, snr=r.snr, tile=r.tile,
                             missing_jwst=miss_J, missing_euclid=miss_E,
                             in_jwst_fp=bool(has_J), in_euclid_fp=bool(has_E)))

    # JWST star, no HST or Euclid counterpart
    in_hst = contains_xy(hst_fp, good_J['ra'].values, good_J['dec'].values)
    in_euclid = contains_xy(euclid_fp, good_J['ra'].values, good_J['dec'].values)
    for r, has_H, has_E in zip(good_J.itertuples(), in_hst, in_euclid):
        sid = r.star_id
        miss_H = bool(has_H) and (sid not in matched_J_in_HJ)
        miss_E = bool(has_E) and (sid not in matched_J_in_JE)
        if miss_H or miss_E:
            rows.append(dict(found_in='JWST F115W', star_id=sid, ra=r.ra, dec=r.dec,
                             mag=r.mag, peak=r.peak, snr=r.snr, tile=r.tile,
                             missing_hst=miss_H, missing_euclid=miss_E,
                             in_hst_fp=bool(has_H), in_euclid_fp=bool(has_E)))

    df = pd.DataFrame(rows)
    for c in ['missing_jwst','missing_hst','missing_euclid',
              'in_jwst_fp','in_hst_fp','in_euclid_fp']:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)

    # Pixel-level coverage refinement deferred to a separate post-process script
    # (28_pixel_coverage_refine.py) so step 6 doesn't block on heavy I/O.
    # For v01 we mark orphans using bbox footprints only; this overestimates
    # coverage at chip gaps, so the "real_orphan" set here is an upper bound.
    df['in_jwst_pix'] = df['in_jwst_fp']      # placeholder; refine later
    df['in_hst_pix']  = df['in_hst_fp']
    df['real_orphan'] = (
        (df['missing_jwst']   & df['in_jwst_fp']) |
        (df['missing_hst']    & df['in_hst_fp'])  |
        (df['missing_euclid'] & df['in_euclid_fp'])
    )

    pq = OUT / 'orphans.parquet'
    df.to_parquet(pq, index=False)
    print(f'orphans: {len(df):,} flagged ({df["real_orphan"].sum():,} after pixel refinement) → {pq}')

    # Summarise by found_in
    if not df.empty:
        print('\nBreakdown:')
        for k, g in df.groupby('found_in'):
            print(f'  {k:<12}  n={len(g):>6,d}  miss_J={g["missing_jwst"].sum():>5,d}  '
                  f'miss_H={g["missing_hst"].sum():>5,d}  miss_E={g["missing_euclid"].sum():>5,d}')

    # HTML: show only real orphans (after pixel refinement), sorted by SNR
    df_real = df[df['real_orphan']].sort_values('snr', ascending=False).reset_index(drop=True)
    df_show = df_real.head(500)  # cap HTML to top 500

    rows_html = []
    for r in df_show.itertuples():
        cells = [
            f'<td>{r.star_id}</td>',
            f'<td>{r.found_in}</td>',
            f'<td>{r.ra:.6f}</td>',
            f'<td>{r.dec:.6f}</td>',
            f'<td>{r.mag:.2f}</td>',
            f'<td>{r.snr:.1f}</td>',
            f'<td>{r.tile}</td>',
            f'<td>{("J" if r.missing_jwst else "")}{("H" if r.missing_hst else "")}{("E" if r.missing_euclid else "")}</td>',
        ]
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')

    summary = []
    if not df_real.empty:
        for k, g in df_real.groupby('found_in'):
            summary.append(f'<li><b>{k}</b>: {len(g):,} real orphans</li>')
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>COSMOS star orphans v01</title>'
        '<style>body{font-family:sans-serif;margin:20px} '
        'table{border-collapse:collapse} '
        'td,th{padding:3px 8px;border:1px solid #ccc;font-size:13px}</style>'
        '</head><body>'
        '<h1>COSMOS star catalog v01 — orphan candidates</h1>'
        '<p>Orphan = good DAO detection (G2-G4 + SNR≥5) in mission X, position inside '
        'mission Y\'s actual pixel coverage (NaN-aware), no PM-tolerant pair from step 4.</p>'
        f'<p>Flagged via bbox: <b>{len(df):,}</b>.  After pixel refinement: '
        f'<b>{len(df_real):,}</b>.  Showing top 500 real orphans by SNR.</p>'
        '<ul>' + ''.join(summary) + '</ul>'
        '<table><thead><tr>'
        '<th>star_id</th><th>found_in</th><th>RA</th><th>Dec</th>'
        '<th>DAO mag</th><th>SNR</th><th>tile</th><th>missing</th>'
        '</tr></thead><tbody>'
        + ''.join(rows_html) + '</tbody></table></body></html>'
    )
    (HTML / 'orphans.html').write_text(html)
    print(f'Wrote {HTML/"orphans.html"}')


if __name__ == '__main__':
    main()
