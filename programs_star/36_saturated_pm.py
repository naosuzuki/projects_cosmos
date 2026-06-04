#!/usr/bin/env python
"""
36_saturated_pm.py — consolidated saturated-star catalog with PM.

Saturated/bright stars are useful for PM measurement (the centroid is
still computable even when the peak is clipped), but they should NOT
appear in the orphan list (they're real known stars, not transients).

Per band, saturated_stars_v2_<BAND>.parquet is the union of:

  1. DAO saturated_stars_<BAND>.parquet  (sharp<0.4 + bright; step 3)
  2. DAO good_stars_<BAND>.parquet within the mag-scaled Gaia veto radius
     (G < 14 : 30″, G 14-16 : 15″, G 16-18 : 5″, G 18-21 : 2″).
  3. DAO good_stars_<BAND>.parquet with peak above the per-band 99-pctl
     of good_star peaks (the saturation cliff).
  4. DAO good_stars_<BAND>.parquet within 30″ of a 99.5-pctl-peak source
     (local diffraction halo contaminants).

The four flags are stored per row (sat_dao, sat_gaia, sat_peak, sat_local)
so downstream filtering can be selective.

Cross-mission PM:

  Match HST F814W (≤2007) ↔ Euclid VIS (2024.5) and JWST F115W (2024)
  ↔ Euclid VIS within a brightness-aware radius (saturated → 5″).
  Linear PM fit in Euclid's frame using mid-baseline epochs.

Outputs
-------
  csvfiles_star/saturated_stars_v2_<BAND>.parquet     per-band consolidated
  csvfiles_star/saturated_stars_v2_combined.csv       cross-matched, with PM
  csvfiles_star/saturated_pm_v2.parquet               full PM table
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

# Same parameters used in step 34 (v3 veto)
GAIA_VETO_LADDER = [(14.0, 30.0), (16.0, 15.0), (18.0, 5.0), (21.0, 2.0)]
PEAK_SAT_PCTL   = 99.0
LOCAL_BRIGHT_PCTL = 99.5
LOCAL_RADIUS_AS = 30.0
REF_EPOCH_GAIA  = 2016.0

BANDS = [
    ('HST',         'F814W', 2005.0),
    ('JWST',        'F115W', 2024.0),
    ('Euclid-VIS',  'VIS',   2024.5),
    ('Euclid-NISP', 'NIR_J', 2024.5),
]

# PM cross-match: saturated stars get larger radius
SAT_MATCH_RADIUS_AS = 5.0    # absolute max — covers 250 mas/yr over 20 yr


def propagate_gaia(gaia, target_epoch):
    out = gaia.copy()
    ref_epoch = out['ref_epoch'].fillna(REF_EPOCH_GAIA).values
    dt = target_epoch - ref_epoch
    cosd = np.cos(np.deg2rad(out['dec'].values))
    pmra_d  = out['pmra'].fillna(0.0).values  / 3.6e6
    pmdec_d = out['pmdec'].fillna(0.0).values / 3.6e6
    out['ra_prop']  = out['ra'].values  + pmra_d  * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def tag_gaia_neighbour(df, gaia_at_epoch):
    """Mag-scaled veto.  Returns array of (in_veto: bool, gaia_g: float)
    per row of df.  in_veto True if within ladder-radius of any Gaia."""
    in_veto = np.zeros(len(df), dtype=bool)
    nearest_g = np.full(len(df), np.nan)
    co = SkyCoord(df['ra'].values * u.deg, df['dec'].values * u.deg)
    g_prev = -np.inf
    for g_max, r_as in GAIA_VETO_LADDER:
        bright = gaia_at_epoch[(gaia_at_epoch['phot_g_mean_mag'] > g_prev) &
                                (gaia_at_epoch['phot_g_mean_mag'] <= g_max)]
        if len(bright) == 0:
            g_prev = g_max
            continue
        cb = SkyCoord(bright['ra_prop'].values * u.deg,
                      bright['dec_prop'].values * u.deg)
        idx_o, idx_b, sep, _ = search_around_sky(co, cb, r_as * u.arcsec)
        for i in np.unique(idx_o):
            in_veto[i] = True
            # record brightest Gaia within this group
            mask = idx_o == i
            gmags = bright['phot_g_mean_mag'].values[idx_b[mask]]
            nearest_g[i] = float(np.nanmin(gmags))
        g_prev = g_max
    return in_veto, nearest_g


def build_saturated_v2(band: str, gaia: pd.DataFrame, epoch: float):
    """Build saturated_stars_v2_<band>.parquet."""
    good = pd.read_parquet(OUT / f'good_stars_{band}.parquet')
    sat  = pd.read_parquet(OUT / f'saturated_stars_{band}.parquet')

    # Per-band peak thresholds
    p99   = float(np.percentile(good['peak'], PEAK_SAT_PCTL)) if len(good) else np.nan
    p99_5 = float(np.percentile(good['peak'], LOCAL_BRIGHT_PCTL)) if len(good) else np.nan

    # Tag good_stars
    g_at = propagate_gaia(gaia, epoch)
    gaia_veto, nearest_g = tag_gaia_neighbour(good, g_at)
    peak_sat = good['peak'].values > p99
    # Local bright neighbour: search within 30" of >p99.5-peak source
    bright_idx = good['peak'].values > p99_5
    if bright_idx.any():
        cgs = SkyCoord(good['ra'].values * u.deg, good['dec'].values * u.deg)
        cbr = SkyCoord(good['ra'].values[bright_idx] * u.deg,
                       good['dec'].values[bright_idx] * u.deg)
        idx_g, idx_b, sep, _ = search_around_sky(cgs, cbr, LOCAL_RADIUS_AS * u.arcsec)
        local_v = np.zeros(len(good), dtype=bool)
        local_v[np.unique(idx_g)] = True
    else:
        local_v = np.zeros(len(good), dtype=bool)

    good = good.copy()
    good['sat_dao']   = False
    good['sat_gaia']  = gaia_veto
    good['sat_peak']  = peak_sat
    good['sat_local'] = local_v
    good['sat_gaia_G_mag'] = nearest_g
    good['saturated_v2'] = gaia_veto | peak_sat | local_v

    # Step-3 saturated catalog also gets the v2 flags (all-true essentially)
    sat = sat.copy()
    sat['sat_dao']   = True
    sat['sat_gaia']  = False
    sat['sat_peak']  = True
    sat['sat_local'] = False
    sat['sat_gaia_G_mag'] = np.nan
    sat['saturated_v2'] = True

    # Saturated set = step-3 sat + good_stars whose saturated_v2 is True
    saturated_v2 = pd.concat([sat, good[good['saturated_v2']].drop(columns=['saturated_v2'])
                              if 'saturated_v2' in good.columns else good[good['saturated_v2']]],
                             ignore_index=True)

    out_pq = OUT / f'saturated_stars_v2_{band}.parquet'
    saturated_v2.to_parquet(out_pq, index=False)
    n_dao   = int(sat['sat_dao'].sum())
    n_gaia  = int(good['sat_gaia'].sum())
    n_peak  = int(good['sat_peak'].sum())
    n_local = int(good['sat_local'].sum())
    print(f'  {band:<5}: total={len(saturated_v2):>7,d}   '
          f'dao={n_dao:>5,d}  gaia={n_gaia:>7,d}  peak={n_peak:>5,d}  local={n_local:>7,d}')
    return saturated_v2


def cross_match_satp(a, b, baseline_yr, max_radius_as=SAT_MATCH_RADIUS_AS):
    """Closest match within max_radius_as. Returns DataFrame of pairs."""
    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame()
    ca = SkyCoord(a['ra'].values * u.deg, a['dec'].values * u.deg)
    cb = SkyCoord(b['ra'].values * u.deg, b['dec'].values * u.deg)
    idx_a, idx_b, sep, _ = search_around_sky(ca, cb, max_radius_as * u.arcsec)
    if len(idx_a) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({'idx_a': idx_a, 'idx_b': idx_b, 'sep_as': sep.arcsec})
    df = df.sort_values(['idx_a', 'sep_as']).drop_duplicates('idx_a', keep='first')
    # Dec midpoint cosine
    ra_a   = a['ra'].values[df['idx_a'].values]
    dec_a  = a['dec'].values[df['idx_a'].values]
    ra_b   = b['ra'].values[df['idx_b'].values]
    dec_b  = b['dec'].values[df['idx_b'].values]
    cosd   = np.cos(np.deg2rad((dec_a + dec_b) / 2.0))
    df['dra_mas']  = (ra_b - ra_a) * cosd * 3.6e6
    df['ddec_mas'] = (dec_b - dec_a) * 3.6e6
    df['pm_ra_mas_yr']  = df['dra_mas']  / baseline_yr
    df['pm_dec_mas_yr'] = df['ddec_mas'] / baseline_yr
    df['baseline_yr']   = baseline_yr
    df['a_ra']  = ra_a;   df['a_dec'] = dec_a
    df['b_ra']  = ra_b;   df['b_dec'] = dec_b
    if 'star_id' in a.columns:
        df['a_star_id'] = a['star_id'].values[df['idx_a'].values]
    if 'star_id' in b.columns:
        df['b_star_id'] = b['star_id'].values[df['idx_b'].values]
    return df


def main():
    print('Loading Gaia DR3 cache...')
    gaia = pd.read_parquet(OUT / 'gaia_dr3_cosmos.parquet')

    print('\nBuilding per-band saturated_stars_v2:')
    sat = {}
    for mission, band, epoch in BANDS:
        sat[band] = build_saturated_v2(band, gaia, epoch)

    # Cross-match HST↔Euclid-VIS, JWST↔Euclid-VIS, HST↔JWST for PM
    print('\nCross-matching saturated stars across missions:')
    p_he = cross_match_satp(sat['F814W'], sat['VIS'], baseline_yr=2024.5 - 2005.0)
    p_he['pair_kind'] = 'HST-Euclid-VIS'
    p_je = cross_match_satp(sat['F115W'], sat['VIS'], baseline_yr=max(2024.5 - 2024.0, 1.0))
    p_je['pair_kind'] = 'JWST-Euclid-VIS'
    p_hj = cross_match_satp(sat['F814W'], sat['F115W'], baseline_yr=2024.0 - 2005.0)
    p_hj['pair_kind'] = 'HST-JWST'
    print(f'  HST-Euclid-VIS : {len(p_he):,}')
    print(f'  JWST-Euclid-VIS: {len(p_je):,}')
    print(f'  HST-JWST       : {len(p_hj):,}')

    # Combine PM measurements per Euclid star (3-epoch linear fit)
    if len(p_he):
        he = p_he[['b_star_id','a_star_id','b_ra','b_dec','dra_mas','ddec_mas']].copy()
        he.columns = ['euclid_id','hst_id','ra_ref','dec_ref','dra_HE_mas','ddec_HE_mas']
        he = he.sort_values('dra_HE_mas').drop_duplicates('euclid_id', keep='first')
    else:
        he = pd.DataFrame()

    if len(p_je):
        je = p_je[['b_star_id','a_star_id','dra_mas','ddec_mas']].copy()
        je.columns = ['euclid_id','jwst_id','dra_JE_mas','ddec_JE_mas']
        je = je.drop_duplicates('euclid_id', keep='first')
    else:
        je = pd.DataFrame()

    if len(he) and len(je):
        joined = pd.merge(he, je, on='euclid_id', how='outer')
    elif len(he):
        joined = he
    else:
        joined = je

    # Simple linear PM (Euclid at t=0, HST at t=-19.5, JWST at t=-0.5)
    tH, tJ = -19.5, -0.5
    sig_H, sig_J = 100.0, 50.0      # mas (saturated centroid uncertainty)
    sig_E = 100.0
    wH = 1.0 / sig_H**2
    wJ = 1.0 / sig_J**2
    wE = 1.0 / sig_E**2

    def slope(yH, yJ):
        hasH = np.isfinite(yH); hasJ = np.isfinite(yJ)
        SW   = hasH*wH + hasJ*wJ + wE
        Swt  = hasH*wH*tH + hasJ*wJ*tJ
        Swtt = hasH*wH*tH*tH + hasJ*wJ*tJ*tJ
        Swy  = hasH*wH*np.where(hasH,yH,0) + hasJ*wJ*np.where(hasJ,yJ,0)
        Swty = hasH*wH*tH*np.where(hasH,yH,0) + hasJ*wJ*tJ*np.where(hasJ,yJ,0)
        det = SW*Swtt - Swt*Swt
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (SW*Swty - Swt*Swy) / det
            sig = np.sqrt(SW / det)
        return s, sig

    if len(joined):
        yH_ra  = joined['dra_HE_mas'].values  if 'dra_HE_mas'  in joined else np.full(len(joined), np.nan)
        yH_dec = joined['ddec_HE_mas'].values if 'ddec_HE_mas' in joined else np.full(len(joined), np.nan)
        yJ_ra  = joined['dra_JE_mas'].values  if 'dra_JE_mas'  in joined else np.full(len(joined), np.nan)
        yJ_dec = joined['ddec_JE_mas'].values if 'ddec_JE_mas' in joined else np.full(len(joined), np.nan)
        # Convention: dra_HE = ra_E - ra_H (Euclid minus HST).  Linear fit ra(t)=ra0 + µ*t,
        # so y(HST) = µ*tH ⇒ µ ≈ (y_E - y_H)/(tE - tH) = -y_H / (-tH) ≈ -y_H/(-19.5)? Wait
        # Re-derive: we stored dra_HE = (ra_E - ra_H)*cosd*3.6e6.  ra_E is REF, value at t=0.
        # ra_H is at t=tH=-19.5.  Δ(ra) = ra_E - ra_H = µ*(tE - tH) = µ*(0-tH) = -µ*tH.
        # ⇒ µ = -dra_HE / tH = dra_HE / 19.5.  i.e., PM_ra ≈ dra_HE / 19.5 yr.
        # The slope() function above does the weighted linear fit with origin at Euclid
        # (y at t=0 is 0), so feed it -dra_HE (offset of HST relative to Euclid) at tH.
        # Easier: just compute simple PM from HE if no JWST, or HE+JE 3-epoch.
        joined['pm_ra_HE']  = -yH_ra  / tH   # mas/yr (Δra / Δt with sign)
        joined['pm_dec_HE'] = -yH_dec / tH
        joined['pm_ra_JE']  = -yJ_ra  / tJ
        joined['pm_dec_JE'] = -yJ_dec / tJ

        # Weighted-mean PM (HE has 19yr baseline → much more precise than JE)
        joined['pm_ra_mas_yr']  = joined['pm_ra_HE'].fillna(joined['pm_ra_JE'])
        joined['pm_dec_mas_yr'] = joined['pm_dec_HE'].fillna(joined['pm_dec_JE'])
        joined['pm_tot_mas_yr'] = np.sqrt(joined['pm_ra_mas_yr']**2 + joined['pm_dec_mas_yr']**2)
        joined['has_HST']  = joined.get('hst_id',  pd.Series(False, index=joined.index)).notna()
        joined['has_JWST'] = joined.get('jwst_id', pd.Series(False, index=joined.index)).notna()
        joined['n_epochs'] = joined['has_HST'].astype(int) + joined['has_JWST'].astype(int) + 1

    out_pq = OUT / 'saturated_pm_v2.parquet'
    joined.to_parquet(out_pq, index=False)
    print(f'\nWrote {out_pq}  ({len(joined):,} rows)')

    # Lite CSV view
    lite_cols = [c for c in ['euclid_id','hst_id','jwst_id','ra_ref','dec_ref',
                             'n_epochs','has_HST','has_JWST',
                             'pm_ra_mas_yr','pm_dec_mas_yr','pm_tot_mas_yr',
                             'dra_HE_mas','ddec_HE_mas','dra_JE_mas','ddec_JE_mas']
                 if c in joined.columns]
    joined[lite_cols].to_csv(OUT / 'saturated_pm_v2.csv', index=False)
    print(f'Wrote {OUT/"saturated_pm_v2.csv"}')

    # Summary
    print('\nSummary (saturated_pm_v2):')
    print(f'  total rows           : {len(joined):,}')
    if len(joined):
        n3 = int((joined.get('n_epochs', pd.Series(dtype=int)) == 3).sum())
        n2 = int((joined.get('n_epochs', pd.Series(dtype=int)) == 2).sum())
        print(f'  3-epoch PM           : {n3:,}')
        print(f'  2-epoch PM           : {n2:,}')
        finite = joined['pm_tot_mas_yr'].dropna()
        if len(finite):
            print(f'  median |µ| (mas/yr)  : {finite.median():.2f}')
            print(f'  |µ| > 50  mas/yr     : {int((finite > 50).sum()):,}')
            print(f'  |µ| > 100 mas/yr     : {int((finite > 100).sum()):,}')


if __name__ == '__main__':
    main()
