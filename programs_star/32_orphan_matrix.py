#!/usr/bin/env python
"""
32_orphan_matrix.py  —  4 × 4 orphan count matrix.

Rows = source mission (good-star detections in its primary band).
Cols = target mission to look up a cross-match in.

  primary bands:
    HST          → F814W   (2003-2007)
    JWST         → F115W   (2024)
    Euclid-VIS   → VIS     (2024.5)
    Euclid-NISP  → NIR-J   (2024.5, sharpest NISP band)

Cell [row][col] = # good-star detections in `row` whose position is inside
                  `col`'s footprint (bbox; pixel-mask refinement applied to
                  JWST + HST when masks are available) AND for which no
                  cross-match in `col` is found within the brightness-aware
                  radius.

Diagonal is the per-row catalog size (good_stars_<primary>).

Writes:
  csvfiles_star/orphan_matrix.csv
  prints the matrix to stdout.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
import shapely.geometry as sg
from shapely import wkt as shapely_wkt, contains_xy

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
MASKS = OUT / 'footprints' / 'masks'

# Source rows: (mission_label, primary_band, epoch_yr)
ROWS = [
    ('HST',         'F814W', 2005.0),
    ('JWST',        'F115W', 2024.0),
    ('Euclid-VIS',  'VIS',   2024.5),
    ('Euclid-NISP', 'NIR_J', 2024.5),
]

# Brightness-aware match radius (mas/yr × baseline-yr → arcsec)
PM_MAX_SAT  = 250.0
PM_MAX_SNR  = [(100, 100.0), (30, 50.0), (10, 25.0), (-np.inf, 15.0)]
RADIUS_FLOOR_AS = 0.30


def _pm_max_for(snr, is_sat):
    if is_sat:
        return PM_MAX_SAT
    for cut, pm in PM_MAX_SNR:
        if snr > cut:
            return pm
    return PM_MAX_SNR[-1][1]


def match_pair(a: pd.DataFrame, b: pd.DataFrame, baseline_yr: float):
    """Return set of A-source row indices that have at least one B match
    within the per-source mag-dep radius."""
    if len(a) == 0 or len(b) == 0:
        return set()
    ca = SkyCoord(a['ra'].values * u.deg, a['dec'].values * u.deg)
    cb = SkyCoord(b['ra'].values * u.deg, b['dec'].values * u.deg)
    max_lim = 6.0
    idx_a, idx_b, sep, _ = search_around_sky(ca, cb, max_lim * u.arcsec)
    sep_as = sep.arcsec

    # Per-source radius
    snr_a = a['snr'].values if 'snr' in a.columns else (a['peak'].values / a['sky_std'].values)
    sat_a = a['is_saturated'].values if 'is_saturated' in a.columns else np.zeros(len(a), bool)
    pm_max_per = np.array([_pm_max_for(s, bool(t)) for s, t in zip(snr_a, sat_a)])
    radius_per = np.maximum(pm_max_per * baseline_yr / 1000.0, RADIUS_FLOOR_AS)

    keep = sep_as <= radius_per[idx_a]
    return set(idx_a[keep].tolist())


def load_pool(primary: str):
    """Pool good_stars + saturated_stars for a given primary band.  Returns df
    with snr (already computed in step 3) and is_saturated boolean."""
    g = pd.read_parquet(OUT / f'good_stars_{primary}.parquet')
    s = pd.read_parquet(OUT / f'saturated_stars_{primary}.parquet')
    g['is_saturated'] = False
    s['is_saturated'] = True
    df = pd.concat([g, s], ignore_index=True)
    return df


def load_band_footprints():
    fp = {}
    with open(OUT / 'footprints' / 'band_footprints.wkt') as f:
        for line in f:
            band, w = line.strip().split('\t', 1)
            fp[band] = shapely_wkt.loads(w)
    return fp


def load_pixel_masks():
    """Return {(band, tile): (mask, wcs)} for HST F814W + JWST F115W."""
    masks = {}
    for p in sorted(MASKS.glob('mask_*.npz')):
        arc = np.load(p, allow_pickle=False)
        # filename: mask_<mission>_<band>_<tile>.npz
        parts = p.stem.split('_')
        if len(parts) >= 4:
            mission = parts[1]
            band    = parts[2]
            tile    = '_'.join(parts[3:])
            hdr_str = arc['header'].item() if arc['header'].ndim == 0 else arc['header'][0]
            from astropy.io.fits import Header
            hdr = Header.fromstring(hdr_str.decode('ascii') if isinstance(hdr_str, bytes) else hdr_str)
            wcs = WCS(hdr)
            masks[(band, tile)] = (arc['mask'], wcs, int(arc['downsample']))
    return masks


def in_pixel_coverage(ra_arr, dec_arr, band, tile_df, masks):
    """Boolean: do (ra, dec) fall in actual pixel coverage of `band`?
    Only HST F814W + JWST F115W have masks; otherwise None."""
    have_mask = (band == 'F814W') or (band == 'F115W')
    if not have_mask:
        return None
    sel = tile_df[(tile_df['band'] == band)]
    has = np.zeros(len(ra_arr), dtype=bool)
    cand_tile = np.full(len(ra_arr), -1, dtype=int)
    polys = []
    for i, r in enumerate(sel.itertuples()):
        ra_min  = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        ra_max  = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        in_box = (ra_arr >= ra_min) & (ra_arr <= ra_max) & (dec_arr >= dec_min) & (dec_arr <= dec_max)
        new = in_box & (cand_tile < 0)
        cand_tile[new] = i
        polys.append(r)
    for ti in np.unique(cand_tile):
        if ti < 0:
            continue
        tr = polys[ti]
        if (band, tr.tile) not in masks:
            continue
        mask, wcs, ds = masks[(band, tr.tile)]
        sel_idx = cand_tile == ti
        x, y = wcs.all_world2pix(ra_arr[sel_idx], dec_arr[sel_idx], 0)
        xi = (x / ds).astype(int)
        yi = (y / ds).astype(int)
        ok = (xi >= 0) & (xi < mask.shape[1]) & (yi >= 0) & (yi < mask.shape[0])
        tile_hit = np.zeros(sel_idx.sum(), dtype=bool)
        tile_hit[ok] = mask[yi[ok], xi[ok]] != 0
        has[np.where(sel_idx)[0]] = tile_hit
    return has


def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    band_fp = load_band_footprints()
    masks   = load_pixel_masks()
    print(f'Loaded pixel masks for {sum(1 for k in masks if k[0]=="F814W")} HST tiles + '
          f'{sum(1 for k in masks if k[0]=="F115W")} JWST tiles')

    # Preload pools
    pools = {}
    for row_label, primary, _ in ROWS:
        pools[row_label] = load_pool(primary)
        print(f'  {row_label:<12} ({primary:<5}): {len(pools[row_label]):>8,d} good+sat')

    # Build matrix
    n = len(ROWS)
    mat = pd.DataFrame(0, index=[r[0] for r in ROWS], columns=[r[0] for r in ROWS])

    for i, (row_lbl, row_band, row_epoch) in enumerate(ROWS):
        a = pools[row_lbl].copy()
        ra = a['ra'].values
        dec = a['dec'].values
        # Diagonal: catalog size
        mat.loc[row_lbl, row_lbl] = len(a)
        for j, (col_lbl, col_band, col_epoch) in enumerate(ROWS):
            if i == j:
                continue
            baseline = abs(col_epoch - row_epoch)
            if baseline < 1.0:
                baseline = 1.0
            # 1. Is the source inside col's footprint?
            if col_band in ('F814W', 'F115W'):
                in_cov = in_pixel_coverage(ra, dec, col_band, tile_df, masks)
                cov_kind = 'pix'
            else:
                # WKT file uses NIR-J convention; pools use NIR_J
                wkt_key = col_band.replace('_', '-')
                in_cov = contains_xy(band_fp[wkt_key], ra, dec)
                cov_kind = 'bbox'
            n_in_cov = int(in_cov.sum())

            # 2. Cross-match a -> b
            b = pools[col_lbl]
            matched = match_pair(a, b, baseline_yr=baseline)
            matched_mask = np.zeros(len(a), dtype=bool)
            matched_mask[list(matched)] = True

            # 3. orphan = inside col footprint AND not matched
            orph = in_cov & (~matched_mask)
            mat.loc[row_lbl, col_lbl] = int(orph.sum())
            print(f'  [{row_lbl:>11s}] inside {col_lbl:>11s} ({cov_kind}): {n_in_cov:>8,d}  '
                  f'matched: {int(matched_mask.sum()):>7,d}  orphan: {int(orph.sum()):>8,d}  '
                  f'(baseline {baseline:.1f} yr)')

    # Print + save
    print('\n4×4 ORPHAN MATRIX  (rows: detected mission, cols: missing in)')
    print('Diagonal = catalog size of row mission (good+saturated, primary band).')
    print('Off-diagonal[r][c] = sources in r inside c\'s footprint with no cross-match in c.')
    print()
    print(mat.to_string())

    mat.to_csv(OUT / 'orphan_matrix.csv')
    print(f'\nWrote {OUT/"orphan_matrix.csv"}')


if __name__ == '__main__':
    main()
