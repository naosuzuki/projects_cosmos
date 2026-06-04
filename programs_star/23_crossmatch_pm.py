#!/usr/bin/env python
"""
23_crossmatch_pm.py  —  Step 4: brightness-aware cross-match (PM-tolerant).

For three mission pairs:
  HST F814W (epoch ~2005) ↔ Euclid VIS (~2024.5)   baseline ~19 yr
  JWST F115W (~2024)      ↔ Euclid VIS (~2024.5)    baseline ~0  yr
  HST F814W              ↔ JWST F115W (~2024)       baseline ~19 yr

Per-mission positions:
  Use the band with the sharpest PSF: HST→F814W, JWST→F115W, Euclid→VIS.
  Pool good_stars + saturated_stars in that band.

Match radius is a function of brightness AND saturation flag, expressed as
the maximum plausible proper motion times the epoch baseline:

  saturated         → pm_max = 250 mas/yr   (some nearby fast movers)
  SNR > 100         → pm_max = 100 mas/yr
  30 < SNR ≤ 100    → pm_max =  50 mas/yr
  10 < SNR ≤ 30     → pm_max =  25 mas/yr
  SNR ≤ 10          → pm_max =  15 mas/yr
  Floor at 0.30″ to allow for centroid + WCS slop.

Outputs (under csvfiles_star/):
  pairs_HST_Euclid.parquet, pairs_JWST_Euclid.parquet, pairs_HST_JWST.parquet
  Each row: starA columns + starB columns + sep_mas, dra_mas, ddec_mas,
            ambig_arcsec (second-closest sep), match_radius_used.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

PRIMARY_BAND = {'HST': 'F814W', 'JWST': 'F115W', 'Euclid': 'VIS'}

EPOCH = {'HST': 2005.0, 'JWST': 2024.0, 'Euclid': 2024.5}

PM_MAX_SAT  = 250.0   # mas/yr  (saturated → very bright nearby stars)
PM_MAX_SNR  = [(100, 100.0), (30, 50.0), (10, 25.0), (-np.inf, 15.0)]  # (snr_cut, pm_max)
RADIUS_FLOOR_AS = 0.30


def _pm_max_for(snr, is_sat):
    """Return pm_max (mas/yr) for a given (snr, is_saturated_bool) row."""
    if is_sat:
        return PM_MAX_SAT
    for cut, pm in PM_MAX_SNR:
        if snr > cut:
            return pm
    return PM_MAX_SNR[-1][1]


def load_mission_stars(mission: str):
    band = PRIMARY_BAND[mission]
    good = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
    sat  = pd.read_parquet(OUT / f'saturated_stars_{band}.parquet')
    good['is_saturated'] = False
    sat ['is_saturated'] = True
    if 'is_good' not in sat.columns:
        sat['is_good'] = False
    df = pd.concat([good, sat], ignore_index=True)
    df['mission'] = mission
    return df


def match_pair(a: pd.DataFrame, b: pd.DataFrame, baseline_yr: float, max_seplimit_as: float = 6.0):
    """Match catalog A → catalog B with per-source mag-dep radius.
    Returns DataFrame with one row per matched A source."""
    ca = SkyCoord(ra=a['ra'].values * u.deg, dec=a['dec'].values * u.deg)
    cb = SkyCoord(ra=b['ra'].values * u.deg, dec=b['dec'].values * u.deg)

    idx_a, idx_b, sep, _ = search_around_sky(ca, cb, max_seplimit_as * u.arcsec)
    # idx_a indexes ca, idx_b indexes cb (module-level fn — unambiguous).

    sep_as = sep.arcsec
    # Per A-source: pick closest B
    pairs = pd.DataFrame({'idx_a': idx_a, 'idx_b': idx_b, 'sep_as': sep_as})
    pairs = pairs.sort_values(['idx_a', 'sep_as']).reset_index(drop=True)
    # closest per idx_a
    first = pairs.groupby('idx_a', as_index=False).first()
    # second closest for ambiguity
    second = pairs.groupby('idx_a', as_index=False).nth(1).rename(columns={'sep_as': 'ambig_arcsec'})[['idx_a','ambig_arcsec']]
    out = first.merge(second, on='idx_a', how='left')
    out['ambig_arcsec'] = out['ambig_arcsec'].fillna(np.inf)

    # Per-source mag-dep radius (uses A's snr + sat flag)
    snr_a = a['snr'].values if 'snr' in a.columns else (a['peak'].values / a['sky_std'].values)
    sat_a = a['is_saturated'].values
    pm_max_per = np.array([_pm_max_for(s, bool(t)) for s, t in zip(snr_a, sat_a)])
    radius_per = np.maximum(pm_max_per * baseline_yr / 1000.0, RADIUS_FLOOR_AS)
    out['match_radius_used'] = radius_per[out['idx_a'].values]
    out['pm_max_used'] = pm_max_per[out['idx_a'].values]
    out = out[out['sep_as'] <= out['match_radius_used']].reset_index(drop=True)

    # Project (Δra cosδ, Δdec) at midpoint
    ra_a   = a['ra'].values[out['idx_a'].values];   dec_a = a['dec'].values[out['idx_a'].values]
    ra_b   = b['ra'].values[out['idx_b'].values];   dec_b = b['dec'].values[out['idx_b'].values]
    cosd   = np.cos(np.deg2rad((dec_a + dec_b) / 2.0))
    out['dra_mas']  = (ra_b - ra_a) * cosd * 3.6e6   # b - a (later epoch minus earlier)
    out['ddec_mas'] = (dec_b - dec_a) * 3.6e6
    out['sep_mas']  = out['sep_as'] * 1000.0

    # Carry forward useful columns from a, b
    keep = ['star_id', 'ra', 'dec', 'mag', 'peak', 'snr', 'sharpness',
            'roundness1', 'roundness2', 'is_saturated', 'tile']
    a_sub = a[keep].iloc[out['idx_a'].values].add_prefix('A_').reset_index(drop=True)
    b_sub = b[keep].iloc[out['idx_b'].values].add_prefix('B_').reset_index(drop=True)
    out   = pd.concat([out.reset_index(drop=True), a_sub, b_sub], axis=1)
    return out


def main():
    hst    = load_mission_stars('HST')
    jwst   = load_mission_stars('JWST')
    euclid = load_mission_stars('Euclid')
    print(f'HST     n_stars (good+sat) = {len(hst):,}')
    print(f'JWST    n_stars (good+sat) = {len(jwst):,}')
    print(f'Euclid  n_stars (good+sat) = {len(euclid):,}')

    pairs = []
    # HST → Euclid
    print('\nHST → Euclid  (baseline 19 yr, mag-dep radius)')
    p1 = match_pair(hst, euclid, baseline_yr=EPOCH['Euclid'] - EPOCH['HST'])
    p1['pair_kind'] = 'HST-Euclid'
    p1['baseline_yr'] = EPOCH['Euclid'] - EPOCH['HST']
    p1.to_parquet(OUT / 'pairs_HST_Euclid.parquet', index=False)
    print(f'  {len(p1):>6,d} pairs')
    pairs.append(p1)

    # JWST → Euclid (contemporaneous: floor radius)
    print('JWST → Euclid (baseline 0.5 yr, floor 0.3")')
    p2 = match_pair(jwst, euclid, baseline_yr=max(EPOCH['Euclid'] - EPOCH['JWST'], 1.0))
    p2['pair_kind'] = 'JWST-Euclid'
    p2['baseline_yr'] = EPOCH['Euclid'] - EPOCH['JWST']
    p2.to_parquet(OUT / 'pairs_JWST_Euclid.parquet', index=False)
    print(f'  {len(p2):>6,d} pairs')
    pairs.append(p2)

    # HST → JWST
    print('HST → JWST  (baseline 19 yr, mag-dep radius)')
    p3 = match_pair(hst, jwst, baseline_yr=EPOCH['JWST'] - EPOCH['HST'])
    p3['pair_kind'] = 'HST-JWST'
    p3['baseline_yr'] = EPOCH['JWST'] - EPOCH['HST']
    p3.to_parquet(OUT / 'pairs_HST_JWST.parquet', index=False)
    print(f'  {len(p3):>6,d} pairs')
    pairs.append(p3)


if __name__ == '__main__':
    main()
