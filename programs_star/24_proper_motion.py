#!/usr/bin/env python
"""
24_proper_motion.py  —  Step 5: PM from per-pair displacements.

Combines the three pair tables from 23_crossmatch_pm.py to build a per-star
proper-motion catalog.  Euclid VIS is the reference epoch (2024.5).

For each Euclid star, look up:
  HST F814W  match (~19 yr earlier)
  JWST F115W match (~0.5 yr earlier)

If both HST + JWST + Euclid positions exist:
  Weighted linear fit ra(t) = ra0 + µα*·t, dec(t) = dec0 + µδ·t
  Per-mission positional weight = 1/(σ_pos)² where σ_pos is rough centroid
  uncertainty (0.05″ default, 0.15″ for saturated).
If only HST + Euclid:
  µα* = (ra_E - ra_H) cosδ / Δt,   µδ = (dec_E - dec_H) / Δt
Same for JWST + Euclid (much larger error per yr).

Outputs:
  csvfiles_star/proper_motion_v01.parquet
  csvfiles_star/proper_motion_v01.csv      (subset cols)
  htmls/star_v01/pm_quiver.png             (vector map)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

EPOCH = {'HST': 2005.0, 'JWST': 2024.0, 'Euclid': 2024.5}

# centroid positional uncertainties (arcsec). Tuned for v01; refine later.
SIGMA_POS = {'HST': 0.020, 'JWST': 0.015, 'Euclid': 0.050}
SIGMA_POS_SAT = {'HST': 0.10, 'JWST': 0.08, 'Euclid': 0.20}


def _sigma(mission, is_sat):
    return SIGMA_POS_SAT[mission] if bool(is_sat) else SIGMA_POS[mission]


def main():
    p_he = pd.read_parquet(OUT / 'pairs_HST_Euclid.parquet')
    p_je = pd.read_parquet(OUT / 'pairs_JWST_Euclid.parquet')
    p_hj = pd.read_parquet(OUT / 'pairs_HST_JWST.parquet')

    print(f'pairs HST-Euclid  = {len(p_he):,}')
    print(f'pairs JWST-Euclid = {len(p_je):,}')
    print(f'pairs HST-JWST    = {len(p_hj):,}')

    # Tag pair tables so we can union them under a common index
    he = p_he[['A_star_id','B_star_id','A_ra','A_dec','B_ra','B_dec',
               'A_is_saturated','B_is_saturated','sep_mas','dra_mas','ddec_mas']].copy()
    he.columns = ['hst_id','euclid_id','ra_H','dec_H','ra_E','dec_E',
                  'sat_H','sat_E','sep_HE_mas','dra_HE_mas','ddec_HE_mas']
    # Dedup: per Euclid star keep only the closest HST match
    he = he.sort_values('sep_HE_mas').drop_duplicates('euclid_id', keep='first').reset_index(drop=True)

    je = p_je[['A_star_id','B_star_id','A_ra','A_dec','B_ra','B_dec',
               'A_is_saturated','B_is_saturated','sep_mas','dra_mas','ddec_mas']].copy()
    je.columns = ['jwst_id','euclid_id','ra_J','dec_J','ra_E2','dec_E2',
                  'sat_J','sat_E2','sep_JE_mas','dra_JE_mas','ddec_JE_mas']
    je = je.sort_values('sep_JE_mas').drop_duplicates('euclid_id', keep='first').reset_index(drop=True)

    # Outer-merge on euclid_id
    df = pd.merge(he, je, on='euclid_id', how='outer')
    print(f'merged unique Euclid stars: {df["euclid_id"].nunique():,}; total rows {len(df):,}')

    # Decide reference (Euclid) ra/dec from either source
    df['ra_ref']  = df['ra_E'].fillna(df['ra_E2'])
    df['dec_ref'] = df['dec_E'].fillna(df['dec_E2'])
    df['sat_E_ref'] = df['sat_E'].fillna(df['sat_E2'])

    # Build per-star arrays of (epoch, ra-ra_ref [mas], dec-dec_ref [mas], sigma)
    cosd = np.cos(np.deg2rad(df['dec_ref'].values))
    # HST relative offsets (HST minus ref)
    dra_H  = (df['ra_H'].values  - df['ra_ref'].values)  * cosd * 3.6e6
    ddec_H = (df['dec_H'].values - df['dec_ref'].values) * 3.6e6
    # JWST
    dra_J  = (df['ra_J'].values  - df['ra_ref'].values)  * cosd * 3.6e6
    ddec_J = (df['dec_J'].values - df['dec_ref'].values) * 3.6e6

    # vectorised linear fit per row.  Use closed-form weighted linear regression.
    # t in years (relative to Euclid epoch).
    tH = EPOCH['HST']    - EPOCH['Euclid']    # ≈ -19.5
    tJ = EPOCH['JWST']   - EPOCH['Euclid']    # ≈ -0.5
    tE = 0.0

    sigH = np.array([_sigma('HST', s)    for s in df['sat_H'].fillna(False)])      * 1000.0  # mas
    sigJ = np.array([_sigma('JWST', s)   for s in df['sat_J'].fillna(False)])      * 1000.0
    sigE = np.array([_sigma('Euclid', s) for s in df['sat_E_ref'].fillna(False)])  * 1000.0

    wH = np.where(np.isfinite(dra_H), 1.0 / sigH**2, 0.0)
    wJ = np.where(np.isfinite(dra_J), 1.0 / sigJ**2, 0.0)
    wE = 1.0 / sigE**2  # Euclid origin: dra=0 by construction

    # Weighted least-squares: each row's slope µ = Σ w t (y) / Σ w t²  - intercept
    # With intercept fit:
    #   A = [[Σw, Σwt], [Σwt, Σwt²]],  b_ra = [Σw·y_ra, Σwt·y_ra]
    # We have at most 3 points per row.  Build sums element-wise.

    def weighted_slope(yH, yJ, yE_zero=True):
        SW   = wH + wJ + wE
        Swt  = wH * tH + wJ * tJ + wE * tE
        Swtt = wH * tH * tH + wJ * tJ * tJ + wE * tE * tE
        Swy  = wH * yH + wJ * yJ + wE * 0.0
        Swty = wH * tH * yH + wJ * tJ * yJ + wE * tE * 0.0
        det = SW * Swtt - Swt * Swt
        with np.errstate(divide='ignore', invalid='ignore'):
            slope     = (SW * Swty - Swt * Swy) / det          # mas / yr
            intercept = (Swtt * Swy - Swt * Swty) / det
            var_slope = SW / det                                # mas² / yr²
        return slope, intercept, np.sqrt(var_slope)

    mu_ra,  ic_ra,  sig_mra  = weighted_slope(np.where(np.isfinite(dra_H), dra_H, 0.0),
                                              np.where(np.isfinite(dra_J), dra_J, 0.0))
    mu_dec, ic_dec, sig_mdec = weighted_slope(np.where(np.isfinite(ddec_H), ddec_H, 0.0),
                                              np.where(np.isfinite(ddec_J), ddec_J, 0.0))

    df['has_HST']  = df['hst_id'].notna()
    df['has_JWST'] = df['jwst_id'].notna()
    df['n_epochs'] = df['has_HST'].astype(int) + df['has_JWST'].astype(int) + 1

    # 1-epoch (Euclid only) → PM undefined
    one_epoch = df['n_epochs'] == 1
    mu_ra  [one_epoch.values] = np.nan
    mu_dec [one_epoch.values] = np.nan
    sig_mra[one_epoch.values] = np.nan
    sig_mdec[one_epoch.values] = np.nan

    df['pm_ra_mas_yr']  = mu_ra
    df['pm_dec_mas_yr'] = mu_dec
    df['pm_ra_err']     = sig_mra
    df['pm_dec_err']    = sig_mdec
    df['pm_tot_mas_yr'] = np.sqrt(df['pm_ra_mas_yr']**2 + df['pm_dec_mas_yr']**2)

    pq = OUT / 'proper_motion_v01.parquet'
    df.to_parquet(pq, index=False)
    print(f'Wrote {pq}')
    df[['euclid_id','hst_id','jwst_id','ra_ref','dec_ref','n_epochs',
        'pm_ra_mas_yr','pm_dec_mas_yr','pm_ra_err','pm_dec_err',
        'pm_tot_mas_yr','sat_H','sat_J','sat_E_ref']].to_csv(OUT / 'proper_motion_v01.csv', index=False)
    print(f'Wrote {OUT/"proper_motion_v01.csv"}')

    # Quiver plot: stars within reasonable PM range (< 200 mas/yr) and 3-epoch only
    sel = (df['n_epochs'] == 3) & (df['pm_tot_mas_yr'] < 200) & np.isfinite(df['pm_ra_mas_yr'])
    sub = df[sel]
    print(f'PM quiver: {len(sub):,} 3-epoch stars w/ |µ|<200 mas/yr')

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = 100.0  # quiver scale (vector length per axis unit)
    ax.quiver(sub['ra_ref'], sub['dec_ref'],
              sub['pm_ra_mas_yr'], sub['pm_dec_mas_yr'],
              sub['pm_tot_mas_yr'],
              cmap='viridis', scale=sc * 60, width=0.0015)
    ax.set_xlim(sub['ra_ref'].max() + 0.02, sub['ra_ref'].min() - 0.02)   # RA reversed
    ax.set_ylim(sub['dec_ref'].min() - 0.02, sub['dec_ref'].max() + 0.02)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title(f'COSMOS proper motions  (Euclid frame, HST+JWST+Euclid; n={len(sub):,})')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(HTML / 'pm_quiver.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"pm_quiver.png"}')

    # Histogram of |µ|
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.loc[df['n_epochs'] == 3, 'pm_tot_mas_yr'].clip(0, 300),
            bins=60, color='C0', alpha=0.75)
    ax.set_xlabel('|μ| (mas/yr)')
    ax.set_ylabel('N stars')
    ax.set_title(f'COSMOS PM magnitude — n={(df["n_epochs"]==3).sum():,} 3-epoch')
    plt.tight_layout()
    plt.savefig(HTML / 'pm_hist.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"pm_hist.png"}')


if __name__ == '__main__':
    main()
