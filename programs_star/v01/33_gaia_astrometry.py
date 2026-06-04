#!/usr/bin/env python
"""
33_gaia_astrometry.py  —  Gaia DR3 astrometry check per band/mission,
                          STARS vs GALAXIES split, plus a bright-Gaia
                          saturated-star veto on the orphan catalog.

Inputs
------
  csvfiles_star/gaia_dr3_cosmos.parquet         (cached Gaia DR3 cone)
  csvfiles_star/good_stars_<BAND>.parquet
  csvfiles_star/saturated_stars_<BAND>.parquet
  csvfiles_star/orphans_refined.parquet         (from step 8)

Output
------
  csvfiles_star/gaia_astrometry_v2.csv          per band × {star, galaxy}
  csvfiles_star/orphans_v2.parquet              bright-Gaia veto applied
  csvfiles_star/orphan_matrix_v2.csv            refreshed 4×4 matrix
  htmls/star_v01/gaia_astrometry_v2.png         scatter Δα, Δδ per band

Classification
--------------
Gaia DR3 stars     : classprob_dsc_combmod_star ≥ 0.5
Gaia DR3 galaxies  : classprob_dsc_combmod_star <  0.5  (mostly QSOs/galaxies)

Bright-Gaia veto for orphans
----------------------------
Bright Gaia STAR  ≡  phot_g_mean_mag < G_BRIGHT (default 18.0)
Any DAO good/sat detection within VETO_RADIUS_AS (default 2.0″) of a bright
Gaia star is flagged `gaia_bright=True` and removed from the v2 orphan list
(was saturated/clipped, not an actual transient).
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
HTML.mkdir(parents=True, exist_ok=True)

# (mission_label, primary_band, epoch_yr)
BANDS = [
    ('HST',         'F814W', 2005.0),
    ('JWST',        'F115W', 2024.0),
    ('Euclid-VIS',  'VIS',   2024.5),
    ('Euclid-NISP', 'NIR_J', 2024.5),
]

MATCH_AS    = 0.5     # Gaia ↔ DAO match radius for astrometric statistics
G_BRIGHT    = 18.0    # bright Gaia veto threshold
VETO_RADIUS = 2.0     # arcsec around bright Gaia star


def propagate_gaia(gaia: pd.DataFrame, target_epoch: float) -> pd.DataFrame:
    """Move Gaia DR3 positions to target_epoch using DR3 PMs.  NaNs in PM
    are treated as 0 mas/yr (galaxy / QSO; PM not measured)."""
    out = gaia.copy()
    ref_epoch = out['ref_epoch'].fillna(2016.0).values
    dt = target_epoch - ref_epoch
    cosd = np.cos(np.deg2rad(out['dec'].values))
    pmra_d  = out['pmra'].fillna(0.0).values  / 3.6e6
    pmdec_d = out['pmdec'].fillna(0.0).values / 3.6e6
    out['ra_prop']  = out['ra'].values  + pmra_d  * dt / cosd
    out['dec_prop'] = out['dec'].values + pmdec_d * dt
    return out


def load_dao(primary: str) -> pd.DataFrame:
    g = pd.read_parquet(OUT / f'good_stars_{primary}.parquet')
    s = pd.read_parquet(OUT / f'saturated_stars_{primary}.parquet')
    g['is_saturated'] = False
    s['is_saturated'] = True
    return pd.concat([g, s], ignore_index=True)


def astrometry_stats(ref_ra, ref_dec, dao_ra, dao_dec, ref_dec_for_cosd):
    """Returns (n_match, dra_mas, ddec_mas, sep_mas) arrays for closest matches."""
    if len(ref_ra) == 0 or len(dao_ra) == 0:
        return 0, np.array([]), np.array([]), np.array([])
    cref = SkyCoord(ref_ra * u.deg, ref_dec * u.deg)
    cdao = SkyCoord(dao_ra * u.deg, dao_dec * u.deg)
    idx_ref, idx_dao, sep, _ = search_around_sky(cref, cdao, MATCH_AS * u.arcsec)
    sep_as = sep.arcsec
    if len(idx_ref) == 0:
        return 0, np.array([]), np.array([]), np.array([])
    pairs = pd.DataFrame({'r': idx_ref, 'd': idx_dao, 's': sep_as})
    pairs = pairs.sort_values(['r','s']).drop_duplicates('r', keep='first')
    ra_a = ref_ra[pairs['r'].values];  dec_a = ref_dec[pairs['r'].values]
    ra_b = dao_ra[pairs['d'].values];  dec_b = dao_dec[pairs['d'].values]
    cosd = np.cos(np.deg2rad((dec_a + dec_b) / 2.0))
    dra  = (ra_b - ra_a) * cosd * 3.6e6   # DAO - Gaia, mas
    ddec = (dec_b - dec_a) * 3.6e6
    return len(pairs), dra, ddec, pairs['s'].values * 1000.0


def main():
    gaia = pd.read_parquet(OUT / 'gaia_dr3_cosmos.parquet')
    print(f'Loaded Gaia DR3: {len(gaia):,} sources in COSMOS box')

    # Split by DSC star prob
    star_mask = gaia['classprob_dsc_combmod_star'].fillna(-1.0) >= 0.5
    gal_mask  = ~star_mask & gaia['classprob_dsc_combmod_star'].notna()
    print(f'  Gaia STAR  (classprob ≥ 0.5): {int(star_mask.sum()):,}')
    print(f'  Gaia GAL/QSO (classprob < 0.5): {int(gal_mask.sum()):,}')
    print(f'  Gaia bright  (G < {G_BRIGHT}): {int((gaia["phot_g_mean_mag"] < G_BRIGHT).sum()):,}')

    rows = []
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(10, 3.0 * len(BANDS)))
    for irow, (mission, band, epoch) in enumerate(BANDS):
        dao = load_dao(band)
        ra_dao  = dao['ra'].values
        dec_dao = dao['dec'].values

        # Propagate Gaia to this epoch
        g_at = propagate_gaia(gaia, epoch)
        g_star = g_at[star_mask].reset_index(drop=True)
        g_gal  = g_at[gal_mask].reset_index(drop=True)

        for kind, gtab, col in [('star', g_star, 'C0'),
                                ('gal',  g_gal,  'C3')]:
            n, dra, ddec, sep = astrometry_stats(
                gtab['ra_prop'].values, gtab['dec_prop'].values,
                ra_dao, dec_dao, dec_dao)
            row = dict(
                mission=mission, band=band, kind=kind,
                n_gaia=len(gtab), n_match=int(n),
                dra_median_mas  = float(np.median(dra))  if n else np.nan,
                dra_std_mas     = float(np.std(dra))     if n else np.nan,
                ddec_median_mas = float(np.median(ddec)) if n else np.nan,
                ddec_std_mas    = float(np.std(ddec))    if n else np.nan,
                sep_median_mas  = float(np.median(sep))  if n else np.nan,
            )
            rows.append(row)
            ax = axes[irow, 0 if kind == 'star' else 1]
            if n:
                ax.scatter(dra, ddec, s=2, alpha=0.3, c=col)
                ax.axhline(0, color='k', lw=0.5)
                ax.axvline(0, color='k', lw=0.5)
                lim = 250
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_xlabel('Δα·cosδ  (mas)')
                ax.set_ylabel('Δδ  (mas)')
                ax.set_title(
                    f'{mission} {band} − Gaia {kind} (n={n:,})\n'
                    f'med Δα={row["dra_median_mas"]:.1f}  Δδ={row["ddec_median_mas"]:.1f}, '
                    f'σ={row["dra_std_mas"]:.1f}/{row["ddec_std_mas"]:.1f}')
            else:
                ax.text(0.5, 0.5, f'{mission} {band} - Gaia {kind}\nNO MATCHES',
                        ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(HTML / 'gaia_astrometry_v2.png', dpi=150)
    plt.close()
    print(f'Wrote {HTML/"gaia_astrometry_v2.png"}')

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / 'gaia_astrometry_v2.csv', index=False)
    print(f'Wrote {OUT/"gaia_astrometry_v2.csv"}')
    print('\nAstrometry stats (DAO - Gaia, mas, median ± std):')
    print(stats_df[['mission','band','kind','n_match',
                    'dra_median_mas','dra_std_mas',
                    'ddec_median_mas','ddec_std_mas']].to_string(index=False))

    # ----- Bright-Gaia saturated-star veto on orphans -----
    print('\n--- Bright Gaia veto on orphan list ---')
    orph = pd.read_parquet(OUT / 'orphans_refined.parquet')
    print(f'  orphans (refined): {len(orph):,}')

    bright = gaia[gaia['phot_g_mean_mag'] < G_BRIGHT].copy()
    print(f'  bright Gaia stars (G < {G_BRIGHT}): {len(bright):,}')

    # Propagate bright Gaia to a "middle" epoch for veto (orphans are detected
    # at various epochs; use 2015 ≈ midpoint between HST and modern epochs)
    bright_2015 = propagate_gaia(bright, 2015.0)
    cb = SkyCoord(bright_2015['ra_prop'].values * u.deg,
                  bright_2015['dec_prop'].values * u.deg)
    co = SkyCoord(orph['ra'].values * u.deg, orph['dec'].values * u.deg)
    idx_b, idx_o, sep, _ = search_around_sky(cb, co, VETO_RADIUS * u.arcsec)
    veto_mask = np.zeros(len(orph), dtype=bool)
    veto_mask[np.unique(idx_o)] = True
    orph['gaia_bright_veto'] = veto_mask
    n_veto = int(veto_mask.sum())
    print(f'  orphans within {VETO_RADIUS}″ of a bright Gaia star → vetoed: {n_veto:,}')

    orph_v2 = orph[~veto_mask].copy()
    print(f'  orphans v2 (after veto): {len(orph_v2):,}')

    out_pq = OUT / 'orphans_v2.parquet'
    orph_v2.to_parquet(out_pq, index=False)
    print(f'Wrote {out_pq}')

    # Breakdown
    print('\nOrphan v2 breakdown by found_in:')
    print(orph_v2.groupby('found_in').size().to_string())


if __name__ == '__main__':
    main()
