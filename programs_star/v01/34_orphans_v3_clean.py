#!/usr/bin/env python
"""
34_orphans_v3_clean.py — third-pass orphan filter to remove saturated /
bright-star contaminants that survived v2.

Issues with v2 thumbnails (Nao 2026-05-26):
  Most "top orphans" were bright saturated stars with diffraction spikes —
  excellent astrometry, but obviously NOT SN candidates.

Improvements implemented here:

  1. Mag-scaled Gaia veto:
        G < 14   → 30″ radius   (very bright; long spikes)
        14-16    → 15″
        16-18    →  5″
        18-21    →  2″
     A DAO detection within the corresponding radius of any Gaia star is
     flagged `gaia_neighbour=True` and dropped from the orphan list.

  2. Peak-based saturation:
     Per-band, compute the 99th-percentile peak among good_stars (a proxy
     for the saturation cliff).  Any orphan with peak > this threshold is
     dropped (likely a clipped bright source DAO classified as PSF-like
     by accident).

  3. Local-bright-neighbour test:
     For each orphan, if there is ANY DAO good_star with peak > 99.5-pctl
     within 30″, drop it (contamination from a nearby very bright object's
     diffraction halo).

  4. Recompute orphan_highconf with these filters, then regenerate top-30
     per category for the diagnostic thumbnails.

Outputs
-------
  csvfiles_star/orphans_v3.parquet                  fully filtered
  csvfiles_star/orphans_highconf_v3.parquet         + SNR>20 + sharp 0.5-0.75
  csvfiles_star/asteroid_candidates_v3.csv
  csvfiles_star/modern_transient_candidates_v3.csv
  csvfiles_star/orphan_matrix_v3_snr30.csv
  htmls/star_v01/orphans_thumbs.html                regenerated for v3
  htmls/star_v01/orphan_thumbs/...                  (existing dir, new pngs)
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

# Gaia neighbour veto: (G_max, radius_arcsec)
GAIA_VETO_LADDER = [
    (14.0, 30.0),
    (16.0, 15.0),
    (18.0,  5.0),
    (21.0,  2.0),
]

PEAK_SAT_PCTL = 99.0   # per-band peak percentile above which we flag as
                       # near-saturation noise
LOCAL_BRIGHT_PCTL = 99.5   # extreme peaks within LOCAL_RADIUS_AS contaminate
LOCAL_RADIUS_AS   = 30.0   # diffraction halo around very bright sources

REF_EPOCH_GAIA = 2016.0

# Map orphans `found_in` → primary band file (so we can compute peak/sharp)
FOUND_TO_BAND = {
    'HST F814W':  'F814W',
    'JWST F115W': 'F115W',
    'Euclid VIS': 'VIS',
}

# Source-band epochs for Gaia propagation
SOURCE_EPOCH = {
    'F814W': 2005.0,
    'F115W': 2024.0,
    'VIS':   2024.5,
}


def propagate_gaia(gaia: pd.DataFrame, target_epoch: float) -> pd.DataFrame:
    out = gaia.copy()
    ref_epoch = out['ref_epoch'].fillna(REF_EPOCH_GAIA).values
    dt = target_epoch - ref_epoch
    cosd = np.cos(np.deg2rad(out['dec'].values))
    pmra_d  = out['pmra'].fillna(0.0).values / 3.6e6
    pmdec_d = out['pmdec'].fillna(0.0).values / 3.6e6
    out['ra_prop']  = out['ra'].values + pmra_d * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def tag_gaia_neighbours(df: pd.DataFrame, gaia: pd.DataFrame, source_band: str) -> np.ndarray:
    """For each row in df, return True if it's within the mag-scaled veto
    radius of any Gaia star at the source epoch."""
    target_epoch = SOURCE_EPOCH.get(source_band, 2020.0)
    g = propagate_gaia(gaia, target_epoch)
    veto = np.zeros(len(df), dtype=bool)
    co = SkyCoord(df['ra'].values * u.deg, df['dec'].values * u.deg)
    g_prev = -np.inf
    for g_max, r_as in GAIA_VETO_LADDER:
        bright = g[(g['phot_g_mean_mag'] > g_prev) &
                    (g['phot_g_mean_mag'] <= g_max)]
        if len(bright) == 0:
            g_prev = g_max
            continue
        cb = SkyCoord(bright['ra_prop'].values * u.deg,
                      bright['dec_prop'].values * u.deg)
        idx_o, idx_b, sep, _ = search_around_sky(co, cb, r_as * u.arcsec)
        veto[np.unique(idx_o)] = True
        g_prev = g_max
    return veto


def main():
    print('Loading...')
    orph = pd.read_parquet(OUT / 'orphans_refined.parquet')
    gaia = pd.read_parquet(OUT / 'gaia_dr3_cosmos.parquet')
    print(f'  orphans_refined: {len(orph):,}')
    print(f'  gaia DR3       : {len(gaia):,}')

    # Build per-band peak thresholds (P_sat) and local-bright neighbour catalogs
    band_sat_thresh = {}
    band_neighbour_peaks = {}   # source positions to query for local bright
    for band in ('F814W', 'F115W', 'VIS'):
        g = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
        if g.empty:
            continue
        p99   = float(np.percentile(g['peak'], PEAK_SAT_PCTL))
        p99_5 = float(np.percentile(g['peak'], LOCAL_BRIGHT_PCTL))
        band_sat_thresh[band] = p99
        # Local bright neighbours are detections with peak > p99_5
        bright = g[g['peak'] > p99_5][['ra', 'dec', 'peak']].copy()
        band_neighbour_peaks[band] = bright
        print(f'  {band:<5}: P_{PEAK_SAT_PCTL}={p99:.3g}, P_{LOCAL_BRIGHT_PCTL}={p99_5:.3g}  ({len(bright):,} extreme-peak sources)')

    # Process per `found_in`
    veto_gaia       = np.zeros(len(orph), dtype=bool)
    veto_peak       = np.zeros(len(orph), dtype=bool)
    veto_local      = np.zeros(len(orph), dtype=bool)

    for found_in, group in orph.groupby('found_in'):
        band = FOUND_TO_BAND.get(found_in)
        if band is None:
            continue
        idx = group.index.values
        sub = orph.loc[idx].copy()
        print(f'\nProcessing {found_in} ({band})  n={len(sub):,}')

        # 1. Mag-scaled Gaia veto
        gaia_v = tag_gaia_neighbours(sub, gaia, band)
        veto_gaia[idx] = gaia_v
        print(f'  gaia neighbour veto : {int(gaia_v.sum()):>7,d} ({100*gaia_v.mean():.1f} %)')

        # 2. Peak-based saturation
        if band in band_sat_thresh:
            peak_v = sub['peak'].values > band_sat_thresh[band]
            veto_peak[idx] = peak_v
            print(f'  peak > P{PEAK_SAT_PCTL}      : {int(peak_v.sum()):>7,d} ({100*peak_v.mean():.1f} %)')

        # 3. Local bright neighbour
        if band in band_neighbour_peaks and len(band_neighbour_peaks[band]):
            bn = band_neighbour_peaks[band]
            co = SkyCoord(sub['ra'].values * u.deg, sub['dec'].values * u.deg)
            cb = SkyCoord(bn['ra'].values * u.deg, bn['dec'].values * u.deg)
            idx_o, _, _, _ = search_around_sky(co, cb, LOCAL_RADIUS_AS * u.arcsec)
            loc_v = np.zeros(len(sub), dtype=bool)
            loc_v[np.unique(idx_o)] = True
            veto_local[idx] = loc_v
            print(f'  local bright nbr     : {int(loc_v.sum()):>7,d} ({100*loc_v.mean():.1f} %)')

    orph['veto_gaia']       = veto_gaia
    orph['veto_peak']       = veto_peak
    orph['veto_local']      = veto_local
    orph['veto_any']        = veto_gaia | veto_peak | veto_local

    n_any = int(orph['veto_any'].sum())
    print(f'\nVeto totals:')
    print(f'  any reason   : {n_any:>10,d}  ({100*n_any/len(orph):.1f} %)')
    print(f'  gaia only    : {int((veto_gaia & ~veto_peak & ~veto_local).sum()):>10,d}')
    print(f'  peak only    : {int((veto_peak & ~veto_gaia & ~veto_local).sum()):>10,d}')
    print(f'  local only   : {int((veto_local & ~veto_gaia & ~veto_peak).sum()):>10,d}')

    orph_v3 = orph[~orph['veto_any']].copy()
    print(f'\norphans v3   : {len(orph_v3):,}')

    orph.to_parquet(OUT / 'orphans_v3_with_flags.parquet', index=False)
    orph_v3.to_parquet(OUT / 'orphans_v3.parquet', index=False)

    # High-confidence v3
    if 'sharpness' not in orph_v3.columns:
        # orphans table doesn't carry sharpness from step 6; pull from good_stars
        # by (star_id) lookup
        sharp_map = {}
        for band, file in [('F814W','good_stars_F814W.parquet'),
                            ('F115W','good_stars_F115W.parquet'),
                            ('VIS',  'good_stars_VIS.parquet')]:
            d = pd.read_parquet(OUT / file)[['star_id','sharpness']]
            sharp_map[band] = d.set_index('star_id')['sharpness'].to_dict()

        def lookup_sharp(row):
            band = FOUND_TO_BAND.get(row.found_in, '')
            return sharp_map.get(band, {}).get(row.star_id, np.nan)

        orph_v3['sharpness'] = orph_v3.apply(lookup_sharp, axis=1)

    hi = orph_v3[(orph_v3['snr'] > 20) &
                 (orph_v3['sharpness'].between(0.5, 0.75))].copy()
    print(f'orphans_highconf v3  : {len(hi):,}')
    hi.to_parquet(OUT / 'orphans_highconf_v3.parquet', index=False)

    # Candidate lists
    ast = orph_v3[
        (orph_v3['found_in']=='HST F814W') &
        orph_v3['missing_jwst'] & orph_v3['missing_euclid'] &
        orph_v3['in_jwst_pix'] & orph_v3['in_euclid_fp'] &
        (orph_v3['snr'] > 30)
    ].copy()
    ast['ra_r']  = (ast['ra']  * 36000).round() / 36000
    ast['dec_r'] = (ast['dec'] * 36000).round() / 36000
    ast = ast.sort_values('snr', ascending=False).drop_duplicates(['ra_r','dec_r']).drop(columns=['ra_r','dec_r'])
    ast.to_csv(OUT / 'asteroid_candidates_v3.csv', index=False)
    print(f'asteroid v3          : {len(ast):,}')

    sn = orph_v3[
        orph_v3['found_in'].isin(['JWST F115W','Euclid VIS']) &
        orph_v3['missing_hst'] & orph_v3['in_hst_pix'] &
        (orph_v3['snr'] > 30)
    ].copy()
    sn['ra_r']  = (sn['ra']  * 36000).round() / 36000
    sn['dec_r'] = (sn['dec'] * 36000).round() / 36000
    sn = sn.sort_values('snr', ascending=False).drop_duplicates(['ra_r','dec_r']).drop(columns=['ra_r','dec_r'])
    sn.to_csv(OUT / 'modern_transient_candidates_v3.csv', index=False)
    print(f'modern transient v3  : {len(sn):,}')

    # v3 breakdown
    print('\nOrphan v3 breakdown:')
    print(orph_v3.groupby('found_in').size().to_string())


if __name__ == '__main__':
    main()
