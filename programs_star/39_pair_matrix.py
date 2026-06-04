#!/usr/bin/env python
"""
39_pair_matrix.py — 4 × 4 PAIR-STAR matrix for astrometry measurement.

Cell [row, col] (symmetric) = # of stars detected in BOTH band(row) and
band(col), using the brightness-aware cross-match radius from step 4.

  primary bands:
    HST          → F814W   (epoch 2005)
    JWST         → F115W   (2024)
    Euclid-VIS   → VIS     (2024.5)
    Euclid-NISP  → NIR-J   (2024.5)

Pool = good_stars + saturated_stars (DAO step-3 catalog only — not the
broader saturated_v2; we use the DAO-clean sample to measure astrometric
offsets reliably).

Output
------
  csvfiles_star/pair_matrix_snr5.csv     all good+sat (SNR>5 implicit by step 3)
  csvfiles_star/pair_matrix_snr30.csv    SNR > 30   sharp 0.5-0.75 cleaner
  csvfiles_star/pair_matrix_snr100.csv   SNR > 100  brightest only
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

BANDS = [
    ('HST',         'F814W', 2005.0),
    ('JWST',        'F115W', 2024.0),
    ('Euclid-VIS',  'VIS',   2024.5),
    ('Euclid-NISP', 'NIR_J', 2024.5),
]

PM_MAX_SAT  = 250.0
PM_MAX_SNR  = [(100, 100.0), (30, 50.0), (10, 25.0), (-np.inf, 15.0)]


def _pm_max_for(snr, is_sat):
    if is_sat: return PM_MAX_SAT
    for cut, pm in PM_MAX_SNR:
        if snr > cut: return pm
    return PM_MAX_SNR[-1][1]


def load_pool(band):
    g = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
    s = pd.read_parquet(OUT / f'saturated_stars_{band}.parquet')
    g['is_saturated'] = False
    s['is_saturated'] = True
    return pd.concat([g, s], ignore_index=True)


def pair_count(a, b, baseline_yr):
    """Number of pairs within brightness-aware radius (closest match per
    A-source).  Symmetric in expectation; we still report A→B count."""
    if len(a) == 0 or len(b) == 0:
        return 0
    ca = SkyCoord(a['ra'].values * u.deg, a['dec'].values * u.deg)
    cb = SkyCoord(b['ra'].values * u.deg, b['dec'].values * u.deg)
    max_lim = 6.0
    idx_a, idx_b, sep, _ = search_around_sky(ca, cb, max_lim * u.arcsec)
    if len(idx_a) == 0:
        return 0
    sep_as = sep.arcsec
    snr_a = a['snr'].values
    sat_a = a['is_saturated'].values if 'is_saturated' in a.columns else np.zeros(len(a), bool)
    pm_max = np.array([_pm_max_for(s, bool(t)) for s, t in zip(snr_a, sat_a)])
    radius = np.maximum(pm_max * baseline_yr / 1000.0, 0.30)
    keep = sep_as <= radius[idx_a]
    # closest A→B
    df = pd.DataFrame({'a': idx_a[keep], 'b': idx_b[keep], 's': sep_as[keep]})
    df = df.sort_values(['a', 's']).drop_duplicates('a', keep='first')
    # also dedup on B side (mutual best)
    df = df.sort_values(['b', 's']).drop_duplicates('b', keep='first')
    return len(df)


def main():
    for SNR_CUT, suffix in [(5, 'snr5'), (30, 'snr30'), (100, 'snr100')]:
        print(f'\n=== Pair-star matrix (SNR > {SNR_CUT}{", sharp 0.5-0.75" if SNR_CUT > 5 else ""}) ===')
        pools = {}
        for _, band, _ in BANDS:
            full = load_pool(band)
            if SNR_CUT > 5:
                a = full[(full['snr'] > SNR_CUT) & (full['sharpness'].between(0.5, 0.75))].copy()
            else:
                a = full.copy()
            pools[band] = a.reset_index(drop=True)

        n = len(BANDS)
        mat = pd.DataFrame(0, index=[r[0] for r in BANDS], columns=[r[0] for r in BANDS])
        for i, (mi_lbl, mi_band, mi_epoch) in enumerate(BANDS):
            mat.loc[mi_lbl, mi_lbl] = len(pools[mi_band])
            for j, (mj_lbl, mj_band, mj_epoch) in enumerate(BANDS):
                if i >= j: continue
                baseline = max(abs(mj_epoch - mi_epoch), 1.0)
                n_pair = pair_count(pools[mi_band], pools[mj_band], baseline)
                mat.loc[mi_lbl, mj_lbl] = n_pair
                mat.loc[mj_lbl, mi_lbl] = n_pair   # symmetric

        print(mat.to_string())
        mat.to_csv(OUT / f'pair_matrix_{suffix}.csv')
        print(f'  → {OUT/f"pair_matrix_{suffix}.csv"}')


if __name__ == '__main__':
    main()
