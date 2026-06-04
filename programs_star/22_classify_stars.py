#!/usr/bin/env python
"""
22_classify_stars.py  —  Step 3: classify DAO detections per band.

Reads per-band parquets written by 21_dao_detect.py and applies:

  GOOD STAR (pass G2-G4 + SNR ≥ 5):
    0.40 ≤ sharpness ≤ 0.85
    |roundness1| ≤ 0.50
    |roundness2| ≤ 0.50
    peak / sky_std ≥ 5.0

  SATURATED STAR (clipped peak, kept for PM):
    sharpness < 0.40                         (broad/flat-topped)
    |roundness1| ≤ 1.0  AND  |roundness2| ≤ 1.0
    peak > 0.9 × max(peak | good-stars) within the same tile
    (per-tile threshold so it adapts to background level)

Everything else is discarded as galaxies / noise / blends.

Outputs (under csvfiles_star/):
  good_stars_<BAND>.parquet         — passing G2-G4 & SNR≥5
  saturated_stars_<BAND>.parquet    — clipped bright stars
  classify_summary.csv              — counts per (band, tile, class)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/suzuki/github/projects_cosmos')
DAO  = ROOT / 'csvfiles_star' / 'dao'
OUT  = ROOT / 'csvfiles_star'
OUT.mkdir(exist_ok=True)

BANDS = ['F814W','F115W','F150W','F277W','F444W','VIS','NIR_Y','NIR_J','NIR_H']

# Master CLAUDE.md §2 morphology gates
SHARPLO, SHARPHI = 0.40, 0.85
ROUND_GOOD      = 0.50    # |r1|, |r2| ≤ this
ROUND_SAT       = 1.00    # looser for saturated
SNR_MIN         = 5.0
SAT_PEAK_FRAC   = 0.90    # peak > this × max(peak | good) → saturated candidate


def classify_band(band: str):
    f = DAO / f'dao_{band}.parquet'
    if not f.exists():
        print(f'  [{band}] no aggregated parquet'); return None
    df = pd.read_parquet(f)
    if df.empty:
        print(f'  [{band}] empty'); return None
    df['snr'] = df['peak'] / df['sky_std']

    # good star mask
    good = (
        (df['sharpness']  >= SHARPLO) & (df['sharpness']  <= SHARPHI) &
        (df['roundness1'].abs() <= ROUND_GOOD) &
        (df['roundness2'].abs() <= ROUND_GOOD) &
        (df['snr']        >= SNR_MIN)
    )
    df['is_good'] = good

    # Per-tile saturation threshold from the brightest good-star peak
    rows_sat = []
    rows_good = df[good].copy()
    rows_good['band'] = band

    sat_threshold_by_tile = (
        df.loc[good].groupby('tile')['peak'].max() * SAT_PEAK_FRAC
    ).to_dict()

    sat_mask = (
        (df['sharpness']  < SHARPLO) &
        (df['roundness1'].abs() <= ROUND_SAT) &
        (df['roundness2'].abs() <= ROUND_SAT)
    )
    # Apply per-tile peak threshold
    tile_thresh = df['tile'].map(sat_threshold_by_tile).fillna(np.inf)
    sat_mask &= df['peak'] > tile_thresh
    rows_sat = df[sat_mask].copy()
    rows_sat['band'] = band
    rows_sat['is_saturated'] = True

    # Add stable IDs (band_tile_idx) — vectorised, works for empty frames too
    def _make_ids(d):
        d = d.reset_index().rename(columns={'index': 'src_idx'})
        if len(d) == 0:
            d['star_id'] = pd.Series([], dtype=str)
        else:
            d['star_id'] = (band + '_' + d['tile'].astype(str) + '_'
                            + d['src_idx'].astype(int).map(lambda i: f'{i:07d}'))
        return d
    rows_good = _make_ids(rows_good)
    rows_sat  = _make_ids(rows_sat)

    pg = OUT / f'good_stars_{band}.parquet'
    ps = OUT / f'saturated_stars_{band}.parquet'
    rows_good.to_parquet(pg, index=False)
    rows_sat.to_parquet(ps, index=False)
    print(f'  [{band:<6}] n_dao={len(df):>9,d}  n_good={len(rows_good):>7,d}  '
          f'n_sat={len(rows_sat):>5,d}  → {pg.name}, {ps.name}')
    return dict(band=band, n_dao=len(df), n_good=len(rows_good), n_sat=len(rows_sat))


def main():
    summary = []
    for band in BANDS:
        r = classify_band(band)
        if r: summary.append(r)
    spath = OUT / 'classify_summary.csv'
    pd.DataFrame(summary).to_csv(spath, index=False)
    print(f'\nWrote {spath}')

    print('\nTotals:')
    if summary:
        sdf = pd.DataFrame(summary)
        print(sdf.to_string(index=False))


if __name__ == '__main__':
    main()
