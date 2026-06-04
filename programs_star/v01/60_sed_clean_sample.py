#!/usr/bin/env python
"""
60_sed_clean_sample.py — build a clean SED-fitting subset from the master
OR catalog.

Selection (per the user's request, for SED fitting we want every band
detected, no saturation, no faint-limit noise):
  - is_point_source == True
  - In EVERY one of the 9 mission bands:
      * detected (mag finite)
      * source_type ∈ {catalog, catalog+gaia}   (not gaia_only_bright)
      * NOT saturated:  mag > SAT_LIMIT[band]
      * good SNR (faint cut): SNR > SNR_MIN
          (Euclid/HST give magerr → SNR = 1.0857/magerr;
           JWST gives snr directly)

Bands (9):
  Euclid VIS, HST F814W, Euclid NIR Y/J/H, JWST F115W/F150W/F277W/F444W

Output:
  csvfiles_star/sed_clean_sample_v01.parquet  (+ .csv)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

SNR_MIN = 10.0

# (band, magcol, errcol, errkind, source_type_col, saturation_limit_mag)
BANDS = [
    ('VIS',   'cat_mag_VIS_vis',    'cat_magerr_VIS_vis',   'magerr', 'source_type_vis',  20.0),
    ('F814W', 'cat_mag_F814W_hst',  'cat_magerr_F814W_hst', 'magerr', 'source_type_hst',  20.0),
    ('NIR_Y', 'cat_mag_NIR_Y_nisp', 'cat_magerr_NIR_Y_nisp','magerr', 'source_type_nisp', 18.0),
    ('F115W', 'cat_mag_F115W_jwst', 'cat_snr_F115W_jwst',   'snr',    'source_type_jwst', 20.0),
    ('NIR_J', 'cat_mag_NIR_J_nisp', 'cat_magerr_NIR_J_nisp','magerr', 'source_type_nisp', 18.0),
    ('F150W', 'cat_mag_F150W_jwst', 'cat_snr_F150W_jwst',   'snr',    'source_type_jwst', 20.0),
    ('NIR_H', 'cat_mag_NIR_H_nisp', 'cat_magerr_NIR_H_nisp','magerr', 'source_type_nisp', 18.0),
    ('F277W', 'cat_mag_F277W_jwst', 'cat_snr_F277W_jwst',   'snr',    'source_type_jwst', 20.0),
    ('F444W', 'cat_mag_F444W_jwst', 'cat_snr_F444W_jwst',   'snr',    'source_type_jwst', 20.0),
]


def main():
    print('Loading master OR catalog ...')
    df = pd.read_parquet(OUT / 'master_or_catalog_v01.parquet')
    ps = df[df['is_point_source']].copy()
    print(f'  point sources: {len(ps):,}')

    mask = np.ones(len(ps), dtype=bool)
    for name, mc, ec, kind, st, sat in BANDS:
        mag = ps[mc].values
        det = np.isfinite(mag)
        not_sat = mag > sat
        is_cat = ps[st].isin(['catalog', 'catalog+gaia']).values
        if kind == 'magerr':
            err = ps[ec].values
            snr_ok = np.isfinite(err) & (err < 1.0857 / SNR_MIN)
        else:
            snr = ps[ec].values
            snr_ok = np.isfinite(snr) & (snr > SNR_MIN)
        band_ok = det & not_sat & is_cat & snr_ok
        mask &= band_ok
        print(f'  after {name:<6} (det+sat>{sat}+SNR>{SNR_MIN:.0f}+catdet): '
              f'{mask.sum():,} remain')

    sub = ps[mask].copy()
    print(f'\nClean SED sample: {len(sub):,} stars (all 9 bands clean)')

    # Add a SNR per band column for convenience, and a derived magerr for JWST
    for name, mc, ec, kind, st, sat in BANDS:
        if kind == 'snr':
            snr = sub[ec].values
            sub[f'magerr_{name}_derived'] = np.where(snr > 0, 1.0857 / snr, np.nan)

    # Report magnitude span of the clean sample per band
    print('\nClean-sample magnitude span per band:')
    print(f'{"band":<7} {"min":>7} {"median":>7} {"max":>7}')
    for name, mc, ec, kind, st, sat in BANDS:
        v = sub[mc].values
        print(f'{name:<7} {np.nanmin(v):>7.2f} {np.nanmedian(v):>7.2f} {np.nanmax(v):>7.2f}')

    # Save
    out_parq = OUT / 'sed_clean_sample_v01.parquet'
    sub.to_parquet(out_parq, index=False)
    sub.to_csv(OUT / 'sed_clean_sample_v01.csv', index=False)
    print(f'\nWrote {out_parq.name} ({len(sub):,} rows × {len(sub.columns)} cols) + .csv')

    # Composition
    print('\nComposition:')
    print(f'  AGN/QSO:  {int(sub["is_agn_qso"].sum()):,}')
    print(f'  stars:    {int((~sub["is_agn_qso"]).sum()):,}')
    print('  primary_source:')
    print(sub['primary_source'].value_counts().to_string())


if __name__ == '__main__':
    main()
