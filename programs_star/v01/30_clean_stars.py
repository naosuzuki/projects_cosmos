#!/usr/bin/env python
"""
30_clean_stars.py — strict-cut clean stellar catalog (v01).

Filters star_master_v01 to a high-confidence subset:
  * 3-epoch PM measurement (HST + JWST + Euclid)
  * n_bands_detected ≥ 4
  * Tighter sharp on the primary detection (we re-pull from good_stars)
    sharp ∈ [0.5, 0.75]
  * PM total < 500 mas/yr (reject outliers)
  * PM ra error < 5 mas/yr (trustworthy fit)

Outputs:
  csvfiles_star/clean_stars_v01.parquet
  csvfiles_star/clean_stars_v01.csv
  htmls/star_v01/clean_pm_quiver.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'

SHARP_LO, SHARP_HI = 0.50, 0.75
MIN_BANDS = 4
MAX_PM    = 500.0  # mas/yr
MAX_PM_ERR = 5.0


def main():
    master = pd.read_parquet(OUT / 'star_master_v01.parquet')
    print(f'master      : {len(master):,}')

    # Primary star is taken from JWST F115W good_stars (sharpest PSF, reference frame).
    # Re-attach sharpness from good_stars_F115W within 0.10" of ra_ref.
    fb = pd.read_parquet(OUT / 'good_stars_F115W.parquet')
    cref = SkyCoord(master['ra_ref'].values * u.deg, master['dec_ref'].values * u.deg)
    cf   = SkyCoord(fb['ra'].values * u.deg,        fb['dec'].values * u.deg)
    idx_ref, idx_f, sep, _ = search_around_sky(cref, cf, 0.10 * u.arcsec)
    pairs = pd.DataFrame({'idx_ref': idx_ref, 'idx_f': idx_f, 'sep': sep.arcsec})
    pairs = pairs.sort_values(['idx_ref', 'sep']).drop_duplicates('idx_ref', keep='first')
    sharp = np.full(len(master), np.nan, dtype=np.float32)
    sharp[pairs['idx_ref'].values] = fb['sharpness'].values[pairs['idx_f'].values]
    master['sharp_F115W'] = sharp

    # If F115W is missing, fall back to VIS sharpness.
    mask_no_J = ~np.isfinite(master['sharp_F115W'])
    if mask_no_J.any():
        vb = pd.read_parquet(OUT / 'good_stars_VIS.parquet')
        cref2 = SkyCoord(master['ra_ref'].values[mask_no_J] * u.deg,
                         master['dec_ref'].values[mask_no_J] * u.deg)
        cv = SkyCoord(vb['ra'].values * u.deg, vb['dec'].values * u.deg)
        idx_r, idx_v, sep, _ = search_around_sky(cref2, cv, 0.20 * u.arcsec)
        p2 = pd.DataFrame({'idx_r': idx_r, 'idx_v': idx_v, 'sep': sep.arcsec})
        p2 = p2.sort_values(['idx_r', 'sep']).drop_duplicates('idx_r', keep='first')
        ref_idx_global = np.where(mask_no_J)[0]
        col = master['sharp_F115W'].values.copy()
        col[ref_idx_global[p2['idx_r'].values]] = vb['sharpness'].values[p2['idx_v'].values]
        master['sharp_F115W'] = col

    # Apply cuts
    cut_sharp = master['sharp_F115W'].between(SHARP_LO, SHARP_HI)
    cut_bands = master['n_bands_detected'] >= MIN_BANDS
    cut_pm    = master['pm_tot_mas_yr'].abs() < MAX_PM
    cut_err   = (master['pm_ra_err'].abs() < MAX_PM_ERR) & (master['pm_dec_err'].abs() < MAX_PM_ERR)
    cut_epoch = master['n_epochs'] >= 2

    breakdown = {
        'has 2+ epochs'      : int(cut_epoch.sum()),
        'sharp ∈ [0.5,0.75]' : int(cut_sharp.sum()),
        'n_bands ≥ 4'        : int(cut_bands.sum()),
        '|µ| < 500 mas/yr'   : int(cut_pm.sum()),
        'PM err < 5 mas/yr'  : int(cut_err.sum()),
        'ALL'                : int((cut_sharp & cut_bands & cut_pm & cut_err & cut_epoch).sum()),
    }
    print('\nCut breakdown:')
    for k, v in breakdown.items():
        print(f'  {k:<22} {v:>8,d}')

    clean = master[cut_sharp & cut_bands & cut_pm & cut_err & cut_epoch].copy()
    out_pq = OUT / 'clean_stars_v01.parquet'
    clean.to_parquet(out_pq, index=False)
    print(f'\nWrote {out_pq}  ({len(clean):,} stars)')

    lite_cols = ['euclid_id','hst_id','jwst_id','ra_ref','dec_ref',
                 'n_epochs','n_bands_detected','sharp_F115W',
                 'pm_ra_mas_yr','pm_dec_mas_yr','pm_ra_err','pm_dec_err','pm_tot_mas_yr',
                 'mag_F814W','mag_F115W','mag_F150W','mag_F277W','mag_F444W',
                 'mag_VIS','mag_NIR_Y','mag_NIR_J','mag_NIR_H']
    lite_cols = [c for c in lite_cols if c in clean.columns]
    clean[lite_cols].to_csv(OUT / 'clean_stars_v01.csv', index=False)
    print(f'Wrote {OUT/"clean_stars_v01.csv"}')

    # Quiver
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(clean['ra_ref'], clean['dec_ref'],
              clean['pm_ra_mas_yr'], clean['pm_dec_mas_yr'],
              clean['pm_tot_mas_yr'].clip(0, 100),
              cmap='viridis', scale=6000, width=0.0017)
    ax.set_xlim(clean['ra_ref'].max() + 0.02, clean['ra_ref'].min() - 0.02)
    ax.set_ylim(clean['dec_ref'].min() - 0.02, clean['dec_ref'].max() + 0.02)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Clean star PM (sharp 0.5-0.75, ≥4 bands, σ_µ < 5)  n={len(clean):,}')
    plt.tight_layout()
    plt.savefig(HTML / 'clean_pm_quiver.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"clean_pm_quiver.png"}')

    # PM histogram for clean subset
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean['pm_tot_mas_yr'].clip(0, 100), bins=40, color='C0')
    ax.set_xlabel('|μ| (mas/yr)')
    ax.set_ylabel('N')
    ax.set_title(f'Clean star PM magnitude  n={len(clean):,}')
    plt.tight_layout()
    plt.savefig(HTML / 'clean_pm_hist.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"clean_pm_hist.png"}')


if __name__ == '__main__':
    main()
