#!/usr/bin/env python
"""
40_pm_aware_pairs_v4.py — Gaia-PM-aware pair selection + v4 orphans.

Two things, in order:

(A) Gaia-confirmed star catalog.
    For every Gaia DR3 source (any classprob), look up its propagated
    position in F814W, F115W, VIS, NIR-J DAO output and record the
    nearest DAO detection within 0.5″.  These DAO rows become
    "Gaia-confirmed stars" regardless of whether they passed the
    sharp/round/SNR cuts.  This fixes the DAO-raw-only Gaia stars that
    the standard cuts missed (4,765 across the four bands per step 38).

    Each Gaia star's matched DAO rows are added to saturated_stars_v2
    (because the user wants them flagged-as-known-stars + carried in PM
    measurement).  Output:
       csvfiles_star/gaia_confirmed_stars_<BAND>.parquet
       csvfiles_star/saturated_stars_v3_<BAND>.parquet
                                (= saturated_v2 ∪ Gaia-confirmed DAO rows)

(B) PM-aware pair selection.
    For each Gaia DR3 star with PM:
       predict positions at HST 2005, JWST 2024, Euclid 2024.5 epochs
       match nearest DAO detection in each band within 0.5″ of predicted
       position
       record a pair if found in BOTH bands
    These "Gaia-anchored" pairs are added to the existing pairs_*_v1
    (which used brightness-aware radius without PM knowledge).

    pairs_<A>_<B>_v4.parquet  contains the union (dedup by A_star_id)
    of v1 brightness-aware pairs + Gaia-PM pairs.

(C) v4 orphans = orphans_refined - saturated_stars_v3 - Gaia-pair anchors.

Output
------
  csvfiles_star/gaia_confirmed_stars_<BAND>.parquet
  csvfiles_star/saturated_stars_v3_<BAND>.parquet
  csvfiles_star/pairs_HST_Euclid_v4.parquet
  csvfiles_star/pairs_JWST_Euclid_v4.parquet
  csvfiles_star/pairs_HST_JWST_v4.parquet
  csvfiles_star/orphans_v4.parquet
  csvfiles_star/orphans_highconf_v4.parquet
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

BAND_EPOCH = [('F814W', 2005.0), ('F115W', 2024.0),
              ('VIS',   2024.5), ('NIR_J', 2024.5)]
FOUND_TO_BAND = {'HST F814W': 'F814W', 'JWST F115W': 'F115W', 'Euclid VIS': 'VIS'}

MATCH_AS_GAIA = 0.5      # tight (PM-propagated)
REF_EPOCH_GAIA = 2016.0


def propagate(gaia, target_epoch):
    out = gaia.copy()
    ref_epoch = out['ref_epoch'].fillna(REF_EPOCH_GAIA).values
    dt = target_epoch - ref_epoch
    cosd = np.cos(np.deg2rad(out['dec'].values))
    pmra_d  = out['pmra'].fillna(0.0).values  / 3.6e6
    pmdec_d = out['pmdec'].fillna(0.0).values / 3.6e6
    out['ra_prop']  = out['ra'].values  + pmra_d  * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def add_dao_star_ids(dao: pd.DataFrame, band: str):
    """Reproduce the star_id convention from step 3."""
    dao = dao.reset_index().rename(columns={'index': '_idx'})
    dao['star_id'] = (band + '_' + dao['tile'].astype(str) + '_'
                      + dao['_idx'].astype(int).map(lambda i: f'{i:07d}'))
    return dao


def main():
    print('--- (A) Gaia-confirmed star catalog ---')
    gaia = pd.read_parquet(OUT / 'gaia_dr3_cosmos.parquet')
    print(f'Gaia DR3 total: {len(gaia):,}')

    confirmed_ids_by_band: dict[str, set] = {}

    for band, epoch in BAND_EPOCH:
        print(f'\n  {band}  (epoch {epoch}):')
        g_at = propagate(gaia, epoch)
        dao  = pd.read_parquet(OUT / 'dao' / f'dao_{band}.parquet')
        dao  = add_dao_star_ids(dao, band)

        cg = SkyCoord(g_at['ra_prop'].values * u.deg, g_at['dec_prop'].values * u.deg)
        cd = SkyCoord(dao['ra'].values  * u.deg, dao['dec'].values  * u.deg)
        idx_g, idx_d, sep, _ = search_around_sky(cg, cd, MATCH_AS_GAIA * u.arcsec)
        df = pd.DataFrame({'g': idx_g, 'd': idx_d, 's': sep.arcsec})
        df = df.sort_values(['g', 's']).drop_duplicates('g', keep='first')
        # also drop dup on dao side (Gaia stars too close to each other → ambiguous)
        df = df.sort_values(['d', 's']).drop_duplicates('d', keep='first')

        dao_rows = dao.iloc[df['d'].values].reset_index(drop=True).copy()
        g_rows   = g_at.iloc[df['g'].values][['source_id', 'phot_g_mean_mag',
                                               'pmra', 'pmdec', 'parallax', 'ruwe',
                                               'classprob_dsc_combmod_star']].reset_index(drop=True)
        g_rows.columns = ['gaia_source_id', 'gaia_G', 'gaia_pmra', 'gaia_pmdec',
                          'gaia_parallax', 'gaia_ruwe', 'gaia_classprob_star']
        confirmed = pd.concat([dao_rows, g_rows], axis=1)
        confirmed['gaia_sep_arcsec'] = df['s'].values

        out_pq = OUT / f'gaia_confirmed_stars_{band}.parquet'
        confirmed.to_parquet(out_pq, index=False)
        print(f'    {len(confirmed):>6,d} Gaia↔DAO pairs (within 0.5″)  → {out_pq.name}')

        # Build saturated_stars_v3 = sat_v2 ∪ Gaia-confirmed DAO rows
        sat_v2 = pd.read_parquet(OUT / f'saturated_stars_v2_{band}.parquet')
        v2_ids = set(sat_v2['star_id'].tolist())
        new_ids = set(confirmed['star_id'].tolist()) - v2_ids
        # New rows: take from full DAO (with classification 'gaia-confirmed')
        new_rows = confirmed[confirmed['star_id'].isin(new_ids)].copy()
        for c in ['sat_dao', 'sat_gaia', 'sat_peak', 'sat_local']:
            new_rows[c] = False
        new_rows['sat_gaia_dao_raw'] = True
        new_rows['sat_gaia_G_mag'] = new_rows['gaia_G']
        sat_v2['sat_gaia_dao_raw'] = False
        sat_v3 = pd.concat([sat_v2, new_rows], ignore_index=True, sort=False)
        sat_v3.to_parquet(OUT / f'saturated_stars_v3_{band}.parquet', index=False)
        confirmed_ids_by_band[band] = set(confirmed['star_id'].tolist())
        print(f'    saturated_stars_v3 = sat_v2 + {len(new_ids):,} new Gaia-confirmed '
              f'→ {len(sat_v3):,} total')

    # ----------------------------------------------------------------
    print('\n--- (B) PM-aware pair selection ---')
    # For each Gaia source with PM, get its predicted (ra, dec) at every band
    # and the nearest DAO confirmation in that band → unique star_ids per band.
    # A "Gaia-anchored pair" = the same Gaia source has confirmed star_ids in
    # BOTH bands of the pair.
    cf = {}
    for band, _ in BAND_EPOCH:
        cf[band] = pd.read_parquet(OUT / f'gaia_confirmed_stars_{band}.parquet')[
            ['gaia_source_id', 'star_id', 'ra', 'dec', 'mag', 'peak', 'sharpness']]
        cf[band].columns = ['gaia_source_id', f'sid_{band}',
                            f'ra_{band}', f'dec_{band}',
                            f'mag_{band}', f'peak_{band}', f'sharp_{band}']

    # HST ↔ Euclid VIS  (~19.5 yr)
    he = pd.merge(cf['F814W'], cf['VIS'], on='gaia_source_id', how='inner', suffixes=('','_VIS'))
    je = pd.merge(cf['F115W'], cf['VIS'], on='gaia_source_id', how='inner', suffixes=('','_VIS'))
    hj = pd.merge(cf['F814W'], cf['F115W'], on='gaia_source_id', how='inner', suffixes=('','_F115W'))
    print(f'  Gaia-anchored HST-Euclid pairs : {len(he):,}')
    print(f'  Gaia-anchored JWST-Euclid pairs: {len(je):,}')
    print(f'  Gaia-anchored HST-JWST pairs   : {len(hj):,}')

    # Union with existing v1 pairs
    def union_pairs(gaia_pairs, existing_path: Path, a_col: str, b_col: str):
        if not existing_path.exists():
            return gaia_pairs
        v1 = pd.read_parquet(existing_path)
        # Build union by A_star_id, B_star_id pair
        v1_keys = set(zip(v1['A_star_id'].tolist(), v1['B_star_id'].tolist()))
        new = gaia_pairs[~gaia_pairs[[a_col, b_col]].apply(tuple, axis=1).isin(v1_keys)]
        return v1, new

    v1_he, new_he = union_pairs(he, OUT / 'pairs_HST_Euclid.parquet', 'sid_F814W', 'sid_VIS')
    v1_je, new_je = union_pairs(je, OUT / 'pairs_JWST_Euclid.parquet', 'sid_F115W', 'sid_VIS')
    v1_hj, new_hj = union_pairs(hj, OUT / 'pairs_HST_JWST.parquet',    'sid_F814W', 'sid_F115W')
    print(f'  v1 pairs HST-Euclid           : {len(v1_he):,}  +Gaia: {len(new_he):,}')
    print(f'  v1 pairs JWST-Euclid          : {len(v1_je):,}  +Gaia: {len(new_je):,}')
    print(f'  v1 pairs HST-JWST             : {len(v1_hj):,}  +Gaia: {len(new_hj):,}')

    # Save v4 pair tables (existing format with star_id pair).  Just track the
    # set of valid (A_star_id, B_star_id) keys after union.
    pairs_v4_paths = {}
    for v1_df, new_df, a_col, b_col, out_name in [
        (v1_he, new_he, 'sid_F814W', 'sid_VIS',   'pairs_HST_Euclid_v4'),
        (v1_je, new_je, 'sid_F115W', 'sid_VIS',   'pairs_JWST_Euclid_v4'),
        (v1_hj, new_hj, 'sid_F814W', 'sid_F115W', 'pairs_HST_JWST_v4'),
    ]:
        v1_keys = set(zip(v1_df['A_star_id'].tolist(), v1_df['B_star_id'].tolist()))
        gaia_keys = set(zip(new_df[a_col].tolist(), new_df[b_col].tolist()))
        all_keys = v1_keys | gaia_keys
        union_df = pd.DataFrame(list(all_keys), columns=['A_star_id', 'B_star_id'])
        union_df['source'] = 'v1'
        union_df.loc[union_df.apply(lambda r: (r.A_star_id, r.B_star_id) in gaia_keys, axis=1), 'source'] = 'gaia_pm'
        union_df.loc[union_df.apply(lambda r: (r.A_star_id, r.B_star_id) in v1_keys & gaia_keys, axis=1), 'source'] = 'both'
        p = OUT / f'{out_name}.parquet'
        union_df.to_parquet(p, index=False)
        pairs_v4_paths[out_name] = p
        print(f'  → {p}  ({len(union_df):,} unique pairs)')

    # ----------------------------------------------------------------
    print('\n--- (C) v4 orphans = orphans_refined minus saturated_stars_v3 ---')
    orph = pd.read_parquet(OUT / 'orphans_refined.parquet')
    v3_ids_F814W = set(pd.read_parquet(OUT / 'saturated_stars_v3_F814W.parquet')['star_id'].tolist())
    v3_ids_F115W = set(pd.read_parquet(OUT / 'saturated_stars_v3_F115W.parquet')['star_id'].tolist())
    v3_ids_VIS   = set(pd.read_parquet(OUT / 'saturated_stars_v3_VIS.parquet')['star_id'].tolist())

    def in_sat_v3(row):
        band = FOUND_TO_BAND.get(row.found_in)
        if   band == 'F814W': return row.star_id in v3_ids_F814W
        elif band == 'F115W': return row.star_id in v3_ids_F115W
        elif band == 'VIS':   return row.star_id in v3_ids_VIS
        return False

    veto_mask = np.array([in_sat_v3(r) for r in orph.itertuples()])
    print(f'  vetoed via saturated_stars_v3: {int(veto_mask.sum()):,}')
    orph_v4 = orph[~veto_mask].copy()
    print(f'  orphans_v4: {len(orph_v4):,}')
    orph_v4.to_parquet(OUT / 'orphans_v4.parquet', index=False)

    # Also subtract Gaia-anchored pair matches (those are now confirmed cross-mission)
    # i.e., remove orphans where the source has a PM-aware pair in another mission.
    # gaia_pair_anchors: for each band's confirmed source, if it pairs to ANOTHER
    # mission via Gaia, it's not an orphan
    anchor_F814W_to_anything = set(he['sid_F814W']).union(set(hj['sid_F814W']))
    anchor_F115W_to_anything = set(je['sid_F115W']).union(set(hj['sid_F115W']))
    anchor_VIS_to_anything   = set(he['sid_VIS']).union(set(je['sid_VIS']))

    def has_anchor(row):
        band = FOUND_TO_BAND.get(row.found_in)
        if   band == 'F814W': return row.star_id in anchor_F814W_to_anything
        elif band == 'F115W': return row.star_id in anchor_F115W_to_anything
        elif band == 'VIS':   return row.star_id in anchor_VIS_to_anything
        return False

    anchor_mask = np.array([has_anchor(r) for r in orph_v4.itertuples()])
    print(f'  Gaia-PM-anchored cross-mission pairs (subtract): {int(anchor_mask.sum()):,}')
    orph_v4_clean = orph_v4[~anchor_mask].copy()
    print(f'  orphans_v4_clean: {len(orph_v4_clean):,}')
    orph_v4_clean.to_parquet(OUT / 'orphans_v4_clean.parquet', index=False)

    # High-conf v4
    sharp_map = {}
    for band in ('F814W', 'F115W', 'VIS'):
        d = pd.read_parquet(OUT / f'good_stars_{band}.parquet')[['star_id', 'sharpness']]
        sharp_map[band] = d.set_index('star_id')['sharpness'].to_dict()
    def lookup_sharp(row):
        band = FOUND_TO_BAND.get(row.found_in, '')
        return sharp_map.get(band, {}).get(row.star_id, np.nan)
    orph_v4_clean['sharpness'] = orph_v4_clean.apply(lookup_sharp, axis=1)
    hi = orph_v4_clean[(orph_v4_clean['snr'] > 20) &
                        (orph_v4_clean['sharpness'].between(0.5, 0.75))].copy()
    hi.to_parquet(OUT / 'orphans_highconf_v4.parquet', index=False)
    print(f'  orphans_highconf_v4: {len(hi):,}')

    print('\nDone.')


if __name__ == '__main__':
    main()
