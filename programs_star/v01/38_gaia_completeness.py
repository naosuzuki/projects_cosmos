#!/usr/bin/env python
"""
38_gaia_completeness.py — verify that DAO + classification gates recover
every Gaia DR3 star in each band, and diagnose any misses.

For each Gaia DR3 star (classprob_star ≥ 0.5):
  1. Propagate (ra, dec) with Gaia PM to that band's epoch.
  2. Find the nearest DAO detection (full dao_<BAND>.parquet, no morpho
     cut) within MATCH_AS = 0.5″.
  3. Mark which catalog (good / saturated_v2 / raw-DAO-only / missing).

Output
------
  csvfiles_star/gaia_recovery_per_band.csv      per-Gaia-source-per-band table
  csvfiles_star/gaia_recovery_summary.csv       counts per (band, status)
  csvfiles_star/gaia_missed.csv                 Gaia stars NOT in DAO output
  htmls/star_v01/gaia_recovery.png              recovery vs G mag, per band
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'

BANDS = [
    ('F814W', 2005.0),
    ('F115W', 2024.0),
    ('VIS',   2024.5),
    ('NIR_J', 2024.5),
]
MATCH_AS = 0.5    # tight (we've propagated with Gaia PM already)
REF_EPOCH_GAIA = 2016.0


def propagate(gaia: pd.DataFrame, target_epoch: float) -> pd.DataFrame:
    out = gaia.copy()
    ref_epoch = out['ref_epoch'].fillna(REF_EPOCH_GAIA).values
    dt = target_epoch - ref_epoch
    cosd = np.cos(np.deg2rad(out['dec'].values))
    pmra_d  = out['pmra'].fillna(0.0).values  / 3.6e6
    pmdec_d = out['pmdec'].fillna(0.0).values / 3.6e6
    out['ra_prop']  = out['ra'].values  + pmra_d  * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def match_to_dao(gaia: pd.DataFrame, dao: pd.DataFrame):
    """For each Gaia source, return (matched_bool, dao_idx, sep_arcsec) of
    the closest DAO source within MATCH_AS."""
    cg = SkyCoord(gaia['ra_prop'].values * u.deg, gaia['dec_prop'].values * u.deg)
    cd = SkyCoord(dao['ra'].values * u.deg, dao['dec'].values * u.deg)
    idx_g, idx_d, sep, _ = search_around_sky(cg, cd, MATCH_AS * u.arcsec)
    if len(idx_g) == 0:
        return (np.zeros(len(gaia), dtype=bool),
                np.full(len(gaia), -1, dtype=np.int64),
                np.full(len(gaia), np.nan))
    df = pd.DataFrame({'g': idx_g, 'd': idx_d, 's': sep.arcsec})
    df = df.sort_values(['g', 's']).drop_duplicates('g', keep='first')
    matched   = np.zeros(len(gaia), dtype=bool)
    dao_idx   = np.full(len(gaia), -1, dtype=np.int64)
    sep_arr   = np.full(len(gaia), np.nan)
    matched[df['g'].values]  = True
    dao_idx[df['g'].values]  = df['d'].values
    sep_arr[df['g'].values]  = df['s'].values
    return matched, dao_idx, sep_arr


def main():
    gaia = pd.read_parquet(OUT / 'gaia_dr3_cosmos.parquet')
    gaia_stars = gaia[gaia['classprob_dsc_combmod_star'] >= 0.5].reset_index(drop=True)
    print(f'Gaia DR3 stars (classprob ≥ 0.5): {len(gaia_stars):,}')

    summary_rows = []
    per_band = {}

    for band, epoch in BANDS:
        print(f'\n--- {band} (epoch {epoch}) ---')
        g_at = propagate(gaia_stars, epoch)

        # Load DAO (all detections, not just good_stars)
        dao = pd.read_parquet(OUT / 'dao' / f'dao_{band}.parquet')

        m_dao, idx_dao, sep_dao = match_to_dao(g_at, dao)
        n_match_dao = int(m_dao.sum())
        print(f'  DAO match within {MATCH_AS}″: {n_match_dao:,} / {len(g_at):,}  '
              f'({100 * m_dao.mean():.1f} %)')

        # Categorise: good / saturated / raw-only / missing
        good = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
        sat  = pd.read_parquet(OUT / f'saturated_stars_v2_{band}.parquet')
        good_ids = set(good['star_id'].tolist()) if 'star_id' in good.columns else set()
        sat_ids  = set(sat['star_id'].tolist()) if 'star_id' in sat.columns else set()

        # Build star_id for raw DAO from band + tile + index (matches step 3 convention)
        # In step 3, star_id = f"{band}_{tile}_{src_idx:07d}" where src_idx is the
        # index in dao_<band>.parquet.  So for a DAO row index i, star_id = band + tile_i + i
        dao = dao.reset_index().rename(columns={'index': 'dao_idx'})
        dao['star_id'] = (band + '_' + dao['tile'].astype(str) + '_'
                          + dao['dao_idx'].astype(int).map(lambda i: f'{i:07d}'))

        # Match Gaia → DAO and look up the star_id
        match_status = np.full(len(g_at), 'missing', dtype=object)
        matched_sid  = np.full(len(g_at), '',        dtype=object)
        for i in np.where(m_dao)[0]:
            sid = dao['star_id'].iat[idx_dao[i]]
            matched_sid[i] = sid
            if sid in good_ids:
                match_status[i] = 'good'
            elif sid in sat_ids:
                match_status[i] = 'sat_v2'
            else:
                match_status[i] = 'dao_raw_only'

        g_at['band']         = band
        g_at['match_status'] = match_status
        g_at['matched_sid']  = matched_sid
        g_at['sep_dao_arcsec'] = sep_dao
        # Pull matched DAO morphology for diagnostics
        sharp_arr = np.full(len(g_at), np.nan)
        round1_arr = np.full(len(g_at), np.nan)
        peak_arr  = np.full(len(g_at), np.nan)
        snr_arr   = np.full(len(g_at), np.nan)
        for i in np.where(m_dao)[0]:
            row = dao.iloc[idx_dao[i]]
            sharp_arr[i]  = row['sharpness']
            round1_arr[i] = row['roundness1']
            peak_arr[i]   = row['peak']
            snr_arr[i]    = row['peak'] / row['sky_std']
        g_at['dao_sharpness']  = sharp_arr
        g_at['dao_roundness1'] = round1_arr
        g_at['dao_peak']       = peak_arr
        g_at['dao_snr']        = snr_arr

        # Save per-band detail
        per_band[band] = g_at[['source_id', 'ra', 'dec', 'phot_g_mean_mag',
                               'pmra', 'pmdec', 'parallax', 'ruwe', 'ra_prop', 'dec_prop',
                               'band', 'match_status', 'matched_sid',
                               'sep_dao_arcsec',
                               'dao_sharpness', 'dao_roundness1',
                               'dao_peak', 'dao_snr']]

        # Per-status counts
        uniq, cnt = np.unique(match_status, return_counts=True)
        cat_counts = dict(zip(uniq.tolist(), cnt.tolist()))
        for status in ['good', 'sat_v2', 'dao_raw_only', 'missing']:
            n = cat_counts.get(status, 0)
            summary_rows.append({'band': band, 'status': status, 'n': int(n),
                                  'frac': float(n / len(g_at))})
            print(f'    {status:<15s}: {n:>6,d}  ({100*n/len(g_at):.1f} %)')

    # Save
    full_df = pd.concat(list(per_band.values()), ignore_index=True)
    full_df.to_csv(OUT / 'gaia_recovery_per_band.csv', index=False)
    print(f'\nWrote {OUT/"gaia_recovery_per_band.csv"}  ({len(full_df):,} rows)')

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / 'gaia_recovery_summary.csv', index=False)
    print(f'Wrote {OUT/"gaia_recovery_summary.csv"}')
    print('\nSummary:')
    print(sdf.pivot(index='band', columns='status', values='n').to_string())

    # Missed Gaia stars (in ANY band) — high-priority list for cut refinement
    missed = full_df[full_df['match_status'].isin(['missing'])][
        ['source_id', 'band', 'phot_g_mean_mag', 'pmra', 'pmdec', 'parallax',
         'ra', 'dec']].copy()
    missed.to_csv(OUT / 'gaia_missed.csv', index=False)
    print(f'\nMissed Gaia stars per band:')
    print(missed.groupby('band').size().to_string())

    # ---- Recovery vs G mag plot ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.arange(14, 22.5, 0.5)
    centers = (bins[:-1] + bins[1:]) / 2

    for i, (band, _) in enumerate(BANDS):
        sub = full_df[full_df['band'] == band]
        ax_ = ax[i // 2, i % 2]
        n_total = np.histogram(sub['phot_g_mean_mag'], bins=bins)[0]
        for status, color in [('good', 'C0'), ('sat_v2', 'C2'), ('dao_raw_only', 'C1'),
                              ('missing', 'C3')]:
            n_st = np.histogram(sub[sub['match_status'] == status]['phot_g_mean_mag'],
                                 bins=bins)[0]
            frac = np.divide(n_st, n_total, out=np.zeros_like(n_st, dtype=float), where=n_total > 0)
            ax_.step(centers, frac, where='mid', color=color, label=status, lw=1.5)
        ax_.set_xlabel('Gaia G (mag)')
        ax_.set_ylabel('fraction')
        ax_.set_ylim(-0.05, 1.05)
        ax_.set_title(f'{band} — total Gaia stars={len(sub):,}')
        if i == 0:
            ax_.legend(loc='center left', fontsize=9)
    plt.tight_layout()
    plt.savefig(HTML / 'gaia_recovery.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"gaia_recovery.png"}')


if __name__ == '__main__':
    main()
