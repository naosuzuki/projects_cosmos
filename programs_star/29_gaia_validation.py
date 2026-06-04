#!/usr/bin/env python
"""
29_gaia_validation.py — Gaia DR3 cross-validation of star_master_v01.

Pulls Gaia DR3 stars in the COSMOS bounding box (149.0 < ra < 151.1,
1.2 < dec < 3.3, G < 21.5 to keep query size sane).  Cross-matches to
the unified catalog (Euclid-frame positions at epoch 2024.5) after
propagating Gaia DR3 (epoch 2016.0) positions with Gaia PM to 2024.5.

Outputs:
  csvfiles_star/gaia_dr3_cosmos.parquet      raw Gaia query (cached)
  csvfiles_star/gaia_match_v01.parquet       per-star cross-match table
  htmls/star_v01/gaia_pm_compare.png         our PM vs Gaia PM scatter
  htmls/star_v01/gaia_pos_compare.png        positional residuals histogram
  htmls/star_v01/gaia_completeness.png       recovery vs G mag
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

warnings.filterwarnings('ignore')

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

GAIA_CACHE = OUT / 'gaia_dr3_cosmos.parquet'
REF_EPOCH  = 2024.5      # Euclid VIS centroid epoch used in step 5
GAIA_EPOCH = 2016.0      # Gaia DR3 reference epoch


def pull_gaia():
    if GAIA_CACHE.exists():
        print(f'Loading cached {GAIA_CACHE}')
        return pd.read_parquet(GAIA_CACHE)

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'

    query = (
        "SELECT source_id, ra, dec, ref_epoch, "
        "pmra, pmra_error, pmdec, pmdec_error, "
        "parallax, parallax_error, "
        "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, "
        "astrometric_excess_noise, astrometric_chi2_al, "
        "ruwe, classprob_dsc_combmod_star "
        "FROM gaiadr3.gaia_source "
        "WHERE ra BETWEEN 149.0 AND 151.1 "
        "  AND dec BETWEEN 1.2 AND 3.3 "
        "  AND phot_g_mean_mag < 21.5"
    )
    print('Submitting Gaia DR3 query...')
    job = Gaia.launch_job_async(query)
    tab = job.get_results()
    df = tab.to_pandas()
    print(f'  retrieved {len(df):,} rows')
    df.to_parquet(GAIA_CACHE, index=False)
    return df


def propagate_to_epoch(df, target_epoch):
    """Move Gaia (ra, dec) at ref_epoch to target_epoch using (pmra, pmdec)."""
    out = df.copy()
    dt = target_epoch - out['ref_epoch'].fillna(GAIA_EPOCH).values  # yr
    cosd = np.cos(np.deg2rad(out['dec'].values))
    # PM in mas/yr → deg
    pmra_d  = (out['pmra'].fillna(0.0).values / 3.6e6)
    pmdec_d = (out['pmdec'].fillna(0.0).values / 3.6e6)
    out['ra_prop']  = out['ra'].values  + pmra_d  * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def main():
    gaia = pull_gaia()
    gaia = propagate_to_epoch(gaia, REF_EPOCH)
    print(f'Gaia DR3 in box: {len(gaia):,}')

    master = pd.read_parquet(OUT / 'star_master_v01.parquet')
    print(f'star_master_v01: {len(master):,}')

    cg = SkyCoord(gaia['ra_prop'].values * u.deg, gaia['dec_prop'].values * u.deg)
    cm = SkyCoord(master['ra_ref'].values * u.deg, master['dec_ref'].values * u.deg)

    # match radius 0.3" — Gaia → master.  Should be tight, since positions are
    # PM-propagated to the same epoch.
    idx_g, idx_m, sep, _ = search_around_sky(cg, cm, 0.3 * u.arcsec)
    print(f'  Gaia ↔ master pairs within 0.3": {len(idx_g):,}')

    pairs = pd.DataFrame({'idx_g': idx_g, 'idx_m': idx_m, 'sep_as': sep.arcsec})
    pairs = pairs.sort_values(['idx_g', 'sep_as']).drop_duplicates('idx_g', keep='first')
    print(f'  unique Gaia matches: {pairs["idx_g"].nunique():,}')

    g_sub = gaia.iloc[pairs['idx_g'].values].reset_index(drop=True)
    m_sub = master.iloc[pairs['idx_m'].values].reset_index(drop=True)
    join = pd.concat([
        g_sub[['source_id','ra','dec','ra_prop','dec_prop','pmra','pmdec',
               'pmra_error','pmdec_error','parallax','phot_g_mean_mag',
               'phot_bp_mean_mag','phot_rp_mean_mag','ruwe',
               'classprob_dsc_combmod_star']].add_prefix('gaia_'),
        m_sub[['euclid_id','ra_ref','dec_ref','n_epochs','pm_ra_mas_yr',
               'pm_dec_mas_yr','pm_ra_err','pm_dec_err','pm_tot_mas_yr',
               'mag_F814W','mag_F115W','mag_F277W','mag_VIS','mag_NIR_J']],
    ], axis=1)
    join['sep_as']     = pairs['sep_as'].values
    join['dpm_ra']     = join['pm_ra_mas_yr'].values  - join['gaia_pmra'].values
    join['dpm_dec']    = join['pm_dec_mas_yr'].values - join['gaia_pmdec'].values

    out_pq = OUT / 'gaia_match_v01.parquet'
    join.to_parquet(out_pq, index=False)
    print(f'Wrote {out_pq}')

    # Completeness vs G mag (independent of cross-match outcome)
    cg_all = SkyCoord(gaia['ra_prop'].values * u.deg, gaia['dec_prop'].values * u.deg)
    idx_a, _, _, _ = search_around_sky(cg_all, cm, 0.3 * u.arcsec)
    recovered = np.zeros(len(gaia), dtype=bool)
    recovered[np.unique(idx_a)] = True
    gaia['recovered'] = recovered

    g_mags = gaia['phot_g_mean_mag'].values
    bins = np.arange(14, 22.5, 0.5)
    n_total = np.histogram(g_mags, bins=bins)[0]
    n_rec   = np.histogram(g_mags[recovered], bins=bins)[0]
    completeness = np.divide(n_rec, n_total, out=np.zeros_like(n_total, dtype=float), where=n_total > 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step((bins[:-1] + bins[1:]) / 2, completeness, where='mid', color='C0', lw=2)
    ax.set_xlabel('Gaia G (mag)')
    ax.set_ylabel('Recovery fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'COSMOS star catalog v01 — recovery of Gaia DR3  (n_gaia={len(gaia):,})')
    plt.tight_layout()
    plt.savefig(HTML / 'gaia_completeness.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"gaia_completeness.png"}')

    # PM comparison (3-epoch only for fair test)
    sel = (join['n_epochs'] == 3) & np.isfinite(join['gaia_pmra']) & np.isfinite(join['pm_ra_mas_yr'])
    sub = join[sel]
    print(f'PM scatter: {len(sub):,} 3-epoch matched stars with Gaia PM')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for axx, x, y, lbl in [
        (ax[0], sub['gaia_pmra'],  sub['pm_ra_mas_yr'],  r'$\mu_{\alpha\!*}$'),
        (ax[1], sub['gaia_pmdec'], sub['pm_dec_mas_yr'], r'$\mu_\delta$'),
    ]:
        lim = 100
        axx.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        axx.scatter(x.clip(-lim, lim), y.clip(-lim, lim), s=2, alpha=0.3)
        axx.set_xlabel(f'Gaia DR3 {lbl} (mas/yr)')
        axx.set_ylabel(f'Our {lbl} (mas/yr)')
        axx.set_xlim(-lim, lim); axx.set_ylim(-lim, lim)
        med = (y - x).median(); std = (y - x).std()
        axx.set_title(f'median Δ = {med:.2f}, σ = {std:.2f} mas/yr  (n={len(sub):,})')
    plt.tight_layout()
    plt.savefig(HTML / 'gaia_pm_compare.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"gaia_pm_compare.png"}')

    # Positional residuals histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(join['sep_as'] * 1000, bins=40, color='C2')
    ax.set_xlabel('Gaia ↔ master separation (mas, after PM-propagation)')
    ax.set_ylabel('N')
    ax.set_title(f'Gaia–master positional offsets (n={len(join):,}, all matched)')
    plt.tight_layout()
    plt.savefig(HTML / 'gaia_pos_compare.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"gaia_pos_compare.png"}')

    # Summary numbers
    print('\nSummary:')
    print(f'  Gaia DR3 stars in COSMOS box (G<21.5)  : {len(gaia):,}')
    print(f'  recovered by our catalog              : {recovered.sum():,}  ({100*recovered.mean():.1f} %)')
    print(f'  3-epoch matched (vs Gaia PM)          : {len(sub):,}')
    if len(sub):
        print(f'  median Δpm_ra* (ours - Gaia)          : {(sub["pm_ra_mas_yr"] - sub["gaia_pmra"]).median():.2f} mas/yr')
        print(f'  median Δpm_dec (ours - Gaia)          : {(sub["pm_dec_mas_yr"] - sub["gaia_pmdec"]).median():.2f} mas/yr')


if __name__ == '__main__':
    main()
