#!/usr/bin/env python
"""
27_unified_catalog.py — assemble a unified per-star catalog by joining
the PM table (step 5) with multi-band DAO magnitudes from every band.

For each row in proper_motion_v01:
  * Locate the JWST F115W detection (closest to ra_ref) and pull
    snr_F115W, peak_F115W, mag_F115W.
  * Same for F150W, F277W, F444W within 0.10″ of the F115W position.
  * Same for Euclid VIS, NIR-Y, NIR-J, NIR-H within 0.20″ of ra_ref.
  * HST F814W within 0.4″ of ra_ref (already matched by step 4 but
    re-attach the per-band column).

Output:
  csvfiles_star/star_master_v01.parquet
  csvfiles_star/star_master_v01.csv     (lite view)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

BANDS_JWST   = ['F115W','F150W','F277W','F444W']
BANDS_EUCLID = ['VIS','NIR_Y','NIR_J','NIR_H']


def attach_band(df, ref_ra, ref_dec, band, radius_arcsec=0.10):
    """For each row in df, find closest good-star detection in band within radius."""
    good = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
    if good.empty:
        for c in ['mag','peak','snr','sharpness']:
            df[f'{c}_{band}'] = np.nan
        return df
    cref  = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg)
    cband = SkyCoord(ra=good['ra'].values*u.deg, dec=good['dec'].values*u.deg)
    idx_ref, idx_band, sep, _ = search_around_sky(cref, cband, radius_arcsec*u.arcsec)
    # Per ref row: closest band detection
    pairs = pd.DataFrame({'idx_ref': idx_ref, 'idx_band': idx_band, 'sep_as': sep.arcsec})
    pairs = pairs.sort_values(['idx_ref','sep_as']).drop_duplicates('idx_ref', keep='first')
    # Map onto df
    for c in ['mag','peak','snr','sharpness']:
        col = np.full(len(df), np.nan, dtype=np.float32)
        col[pairs['idx_ref'].values] = good[c].values[pairs['idx_band'].values]
        df[f'{c}_{band}'] = col
    return df


def main():
    pm = pd.read_parquet(OUT / 'proper_motion_v01.parquet')
    print(f'PM table: {len(pm):,} rows')

    ref_ra  = pm['ra_ref'].values
    ref_dec = pm['dec_ref'].values

    # HST F814W (already in pm via pairs, but consolidate to single column)
    pm = attach_band(pm, ref_ra, ref_dec, 'F814W', radius_arcsec=0.4)
    print('attached F814W')
    for b in BANDS_JWST:
        pm = attach_band(pm, ref_ra, ref_dec, b, radius_arcsec=0.10)
        print(f'attached {b}')
    for b in BANDS_EUCLID:
        pm = attach_band(pm, ref_ra, ref_dec, b, radius_arcsec=0.20)
        print(f'attached {b}')

    # how many detected bands per star
    band_cols = [f'mag_{b}' for b in (['F814W'] + BANDS_JWST + BANDS_EUCLID)]
    pm['n_bands_detected'] = pm[band_cols].notna().sum(axis=1)

    out_pq = OUT / 'star_master_v01.parquet'
    pm.to_parquet(out_pq, index=False)
    print(f'\nWrote {out_pq}')

    # Lite CSV view: core columns only
    lite_cols = ['euclid_id','hst_id','jwst_id','ra_ref','dec_ref',
                 'n_epochs','n_bands_detected',
                 'pm_ra_mas_yr','pm_dec_mas_yr','pm_ra_err','pm_dec_err','pm_tot_mas_yr',
                 ] + [f'mag_{b}' for b in (['F814W'] + BANDS_JWST + BANDS_EUCLID)]
    lite_cols = [c for c in lite_cols if c in pm.columns]
    out_csv = OUT / 'star_master_v01.csv'
    pm[lite_cols].to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}  ({len(lite_cols)} columns)')

    # Summary
    print('\nSummary:')
    print(f'  total stars                   : {len(pm):,}')
    print(f'  with 3-epoch PM (HST+JWST+E)  : {(pm["n_epochs"]==3).sum():,}')
    print(f'  with 2-epoch PM (HST+E or J+E): {(pm["n_epochs"]==2).sum():,}')
    print(f'  median |µ| (3-epoch, < 200)   : '
          f'{pm.loc[(pm["n_epochs"]==3) & (pm["pm_tot_mas_yr"]<200), "pm_tot_mas_yr"].median():.2f} mas/yr')
    print(f'  median n_bands_detected       : {pm["n_bands_detected"].median():.1f}')

if __name__ == '__main__':
    main()
