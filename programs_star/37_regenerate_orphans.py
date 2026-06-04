#!/usr/bin/env python
"""
37_regenerate_orphans.py — clean orphan regeneration using
saturated_stars_v2_<BAND>.parquet as the explicit "known bright/saturated"
veto reference (instead of re-computing the conditions inline as in step 34).

Outputs (all marked _v3):
  csvfiles_star/orphans_v3.parquet            (overwritten, same content as v34)
  csvfiles_star/orphans_highconf_v3.parquet   (overwritten)
  csvfiles_star/orphan_matrix_v3_snr5.csv
  csvfiles_star/orphan_matrix_v3_snr30.csv
  csvfiles_star/orphan_matrix_v3_snr100.csv
  csvfiles_star/sn_candidates_v3.csv          (focused SN/transient list)
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io.fits import Header
from shapely import wkt as shapely_wkt, contains_xy

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
MASKS = OUT / 'footprints' / 'masks'

BANDS = [
    ('HST',         'F814W', 2005.0),
    ('JWST',        'F115W', 2024.0),
    ('Euclid-VIS',  'VIS',   2024.5),
    ('Euclid-NISP', 'NIR_J', 2024.5),
]

FOUND_TO_BAND = {
    'HST F814W':  'F814W',
    'JWST F115W': 'F115W',
    'Euclid VIS': 'VIS',
}

PM_MAX_SAT  = 250.0
PM_MAX_SNR  = [(100, 100.0), (30, 50.0), (10, 25.0), (-np.inf, 15.0)]


def _pm_max_for(snr, is_sat):
    if is_sat: return PM_MAX_SAT
    for cut, pm in PM_MAX_SNR:
        if snr > cut: return pm
    return PM_MAX_SNR[-1][1]


def match_pair(a, b, baseline_yr):
    if len(a) == 0 or len(b) == 0:
        return set()
    ca = SkyCoord(a['ra'].values * u.deg, a['dec'].values * u.deg)
    cb = SkyCoord(b['ra'].values * u.deg, b['dec'].values * u.deg)
    idx_a, idx_b, sep, _ = search_around_sky(ca, cb, 6 * u.arcsec)
    snr_a = a['snr'].values
    sat_a = a['is_saturated'].values if 'is_saturated' in a.columns else np.zeros(len(a), bool)
    pm_max = np.array([_pm_max_for(s, bool(t)) for s, t in zip(snr_a, sat_a)])
    radius = np.maximum(pm_max * baseline_yr / 1000.0, 0.30)
    keep = sep.arcsec <= radius[idx_a]
    return set(idx_a[keep].tolist())


def load_pool(band):
    g = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
    s = pd.read_parquet(OUT / f'saturated_stars_{band}.parquet')
    g['is_saturated'] = False
    s['is_saturated'] = True
    return pd.concat([g, s], ignore_index=True)


def main():
    print('Loading saturated_stars_v2 (consolidated bright/saturated catalogs)...')
    sat_v2 = {}
    sat_v2_ids = {}
    for _, band, _ in BANDS:
        path = OUT / f'saturated_stars_v2_{band}.parquet'
        if not path.exists():
            print(f'  [missing] {path}'); continue
        df = pd.read_parquet(path)
        sat_v2[band] = df
        # Build set of star_ids (these are the v2-saturated "known stars")
        sat_v2_ids[band] = set(df['star_id'].tolist()) if 'star_id' in df.columns else set()
        print(f'  {band:<5}: {len(df):>7,d} entries  (unique star_ids: {len(sat_v2_ids[band]):,})')

    # ---- Regenerate orphans by removing saturated_stars_v2 members ----
    print('\nRegenerating orphan list (orphans_refined minus saturated_stars_v2):')
    orph = pd.read_parquet(OUT / 'orphans_refined.parquet')
    print(f'  orphans_refined: {len(orph):,}')

    # Veto rule: drop orphan if its star_id is in saturated_stars_v2_<band> for
    # its `found_in` band
    def is_sat(row):
        band = FOUND_TO_BAND.get(row.found_in)
        if band is None: return False
        return row.star_id in sat_v2_ids.get(band, set())

    sat_mask = np.array([is_sat(r) for r in orph.itertuples()])
    print(f'  vetoed (in saturated_stars_v2): {int(sat_mask.sum()):,}')
    orph['vetoed_by_sat_v2'] = sat_mask

    orph_v3 = orph[~sat_mask].copy()
    orph.to_parquet(OUT / 'orphans_with_sat_v2_flag.parquet', index=False)
    orph_v3.to_parquet(OUT / 'orphans_v3.parquet', index=False)
    print(f'  orphans_v3: {len(orph_v3):,}')

    # High-confidence v3
    sharp_map = {}
    for band in ('F814W', 'F115W', 'VIS'):
        d = pd.read_parquet(OUT / f'good_stars_{band}.parquet')[['star_id', 'sharpness']]
        sharp_map[band] = d.set_index('star_id')['sharpness'].to_dict()

    def lookup_sharp(row):
        band = FOUND_TO_BAND.get(row.found_in, '')
        return sharp_map.get(band, {}).get(row.star_id, np.nan)

    orph_v3['sharpness'] = orph_v3.apply(lookup_sharp, axis=1)
    hi = orph_v3[(orph_v3['snr'] > 20) & (orph_v3['sharpness'].between(0.5, 0.75))].copy()
    hi.to_parquet(OUT / 'orphans_highconf_v3.parquet', index=False)
    print(f'  orphans_highconf_v3: {len(hi):,}')

    # ---- 4×4 matrix using saturated_stars_v2 as veto ----
    print('\nBuilding orphan matrix v3 (saturated_stars_v2 veto):')
    fp = {}
    with open(OUT / 'footprints' / 'band_footprints.wkt') as f:
        for line in f:
            b, w = line.strip().split('\t', 1)
            fp[b] = shapely_wkt.loads(w)

    masks = {}
    for p in sorted(MASKS.glob('mask_*.npz')):
        arc = np.load(p, allow_pickle=False)
        parts = p.stem.split('_'); band = parts[2]; tile = '_'.join(parts[3:])
        hdr_str = arc['header'].item() if arc['header'].ndim == 0 else arc['header'][0]
        hdr = Header.fromstring(hdr_str.decode('ascii') if isinstance(hdr_str, bytes) else hdr_str)
        masks[(band, tile)] = (arc['mask'], WCS(hdr), int(arc['downsample']))

    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')

    def in_pix(ra, dec, band):
        if band not in ('F814W', 'F115W'): return None
        sel = tile_df[tile_df['band'] == band]
        has = np.zeros(len(ra), bool)
        cand_tile = np.full(len(ra), -1, dtype=int)
        polys = []
        for i, r in enumerate(sel.itertuples()):
            ra_min = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
            ra_max = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
            dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
            dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
            in_box = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            cand_tile[in_box & (cand_tile < 0)] = i
            polys.append(r)
        for ti in np.unique(cand_tile):
            if ti < 0: continue
            tr = polys[ti]
            if (band, tr.tile) not in masks: continue
            m, wcs, ds = masks[(band, tr.tile)]
            sel_idx = cand_tile == ti
            x, y = wcs.all_world2pix(ra[sel_idx], dec[sel_idx], 0)
            xi = (x / ds).astype(int); yi = (y / ds).astype(int)
            ok = (xi >= 0) & (xi < m.shape[1]) & (yi >= 0) & (yi < m.shape[0])
            tile_hit = np.zeros(sel_idx.sum(), bool)
            tile_hit[ok] = m[yi[ok], xi[ok]] != 0
            has[np.where(sel_idx)[0]] = tile_hit
        return has

    for SNR_CUT, suffix in [(5, 'all'), (30, 'snr30'), (100, 'snr100')]:
        mat = pd.DataFrame(0, index=[r[0] for r in BANDS], columns=[r[0] for r in BANDS])
        for i, (row_lbl, row_band, row_epoch) in enumerate(BANDS):
            full = load_pool(row_band)
            if SNR_CUT > 5:
                a = full[(full['snr'] > SNR_CUT) & (full['sharpness'].between(0.5, 0.75))].copy()
            else:
                a = full.copy()
            # Veto: drop rows in saturated_stars_v2_<row_band>
            a = a[~a['star_id'].isin(sat_v2_ids.get(row_band, set()))].reset_index(drop=True)
            if len(a) == 0:
                continue
            ra = a['ra'].values; dec = a['dec'].values
            mat.loc[row_lbl, row_lbl] = len(a)
            for j, (col_lbl, col_band, col_epoch) in enumerate(BANDS):
                if i == j: continue
                baseline = max(abs(col_epoch - row_epoch), 1.0)
                if col_band in ('F814W', 'F115W'):
                    in_cov = in_pix(ra, dec, col_band)
                else:
                    in_cov = contains_xy(fp[col_band.replace('_', '-')], ra, dec)
                b = load_pool(col_band)
                matched = match_pair(a, b, baseline)
                mm = np.zeros(len(a), bool); mm[list(matched)] = True
                mat.loc[row_lbl, col_lbl] = int((in_cov & ~mm).sum())
        out = OUT / f'orphan_matrix_v3_{suffix}.csv'
        mat.to_csv(out)
        print(f'\n=== Orphan matrix v3 (SNR>{SNR_CUT}) — sat_stars_v2 vetoed ===')
        print(mat.to_string())
        print(f'  → {out}')

    # ---- Focused SN candidate list ----
    print('\nBuilding focused SN candidate list (v3):')
    sn = orph_v3[
        (orph_v3['found_in'].isin(['JWST F115W', 'Euclid VIS'])) &
        orph_v3['missing_hst'] & orph_v3['in_hst_pix'] &
        (orph_v3['snr'] > 50) &
        (orph_v3['sharpness'].between(0.55, 0.70))
    ].copy()
    sn['ra_r']  = (sn['ra']  * 36000).round() / 36000
    sn['dec_r'] = (sn['dec'] * 36000).round() / 36000
    sn = sn.sort_values('snr', ascending=False).drop_duplicates(['ra_r', 'dec_r']).drop(columns=['ra_r', 'dec_r'])
    sn.to_csv(OUT / 'sn_candidates_v3.csv', index=False)
    print(f'  SN candidates v3 (SNR>50, sharp 0.55-0.70, in HST pixel cov, no HST pair): {len(sn):,}')


if __name__ == '__main__':
    main()
