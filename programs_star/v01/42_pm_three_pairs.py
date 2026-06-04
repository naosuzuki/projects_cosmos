#!/usr/bin/env python
"""
42_pm_three_pairs.py — proper-motion quiver plots for HST→Euclid,
                       JWST→Euclid, HST→JWST, following the style of
                       projects_euclid/programs/20_pm_three_pairs_quiver.py

Method (carried over from projects_euclid):
  1. Restrict every catalogue to the HST ∩ JWST common pixel mask.
  2. Compute a galaxy frame-tie per pair = median residual of
     mutual-NN galaxy matches (galaxies don't move, so this isolates
     the static WCS offset between two frames).
  3. Per-star displacement minus frame tie / baseline → PM in mas/yr.
  4. Triple-confirmed stars: matched in both HST↔Euclid and HST↔JWST
     (the ACS gold-star anchor approach from script 20).
  5. Quiver maps coloured by |PM| (coolwarm); per-axis histograms.

Galaxy classifier per band:
  HST F814W   : DAO sharp > 0.85 (extended) AND |round| ≤ 0.5
  JWST F115W  : same morphology criterion
  Euclid VIS  : same morphology criterion
The DAO output is rich enough — we use the same morphology-based proxy
across all three.

Output
------
  csvfiles_star/triple_morphology_stars.csv
  htmls/star_v01/pm_quiver_HST_to_Euclid.png
  htmls/star_v01/pm_quiver_JWST_to_Euclid.png
  htmls/star_v01/pm_quiver_HST_to_JWST.png
  htmls/star_v01/pm_three_pairs_histograms.png
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy import units as u

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

# Epochs
T_HST    = 2005.0
T_JWST   = 2024.0
T_EUCLID = 2024.5

MATCH_AS = 1.0
SHARP_STAR_RANGE = (0.50, 0.75)   # tight stellar
# DAO sharpness for extended sources (galaxies) is LOWER than PSF-like.
# DAO sharplo=0.2 was applied at detection time, so 0.20-0.40 selects mostly
# extended; round must stay small (-0.5..0.5) to drop blends / asymmetric junk.
SHARP_GAL_RANGE  = (0.20, 0.40)
ROUND_GAL_MAX    = 0.5
SNR_STAR_MIN     = 5.0
SNR_GAL_MIN      = 5.0


def nmad(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


def mutual_nn(c_a, c_b, rad):
    if len(c_a) == 0 or len(c_b) == 0:
        return np.zeros(len(c_a), dtype=bool), np.zeros(len(c_a), dtype=int)
    a2b_idx, a2b_sep, _ = c_a.match_to_catalog_sky(c_b)
    b2a_idx, b2a_sep, _ = c_b.match_to_catalog_sky(c_a)
    ok_a = a2b_sep <= rad
    ok_b = b2a_sep <= rad
    mutual = ok_a & ok_b[a2b_idx] & (b2a_idx[a2b_idx] == np.arange(len(c_a)))
    return mutual, a2b_idx


def load_common_mask():
    """Load 3″/pix common HST∩JWST coverage mask + its WCS."""
    p = OUT / 'footprints' / 'common_footprint_v5.fits'
    with fits.open(p) as h:
        mask = h[0].data.astype(bool)
        wcs  = WCS(h[0].header)
    return mask, wcs


def inside_mask(ra, dec, mask, wcs):
    x, y = wcs.all_world2pix(ra, dec, 0)
    xi = np.round(x).astype(int); yi = np.round(y).astype(int)
    ny, nx = mask.shape
    ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    out = np.zeros(len(ra), dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def load_band_with_class(band: str, common_mask, common_wcs):
    """Load the FULL DAO output for `band` (so we have access to the
    extended-source sharp<0.40 regime), restrict to common mask, return
    (df_star, df_gal) with star/galaxy morphology flags."""
    dao = pd.read_parquet(OUT / 'dao' / f'dao_{band}.parquet')
    dao['snr'] = dao['peak'] / dao['sky_std']
    in_com = inside_mask(dao['ra'].values, dao['dec'].values, common_mask, common_wcs)
    df = dao[in_com].reset_index(drop=True)

    is_star = ((df['sharpness'].between(*SHARP_STAR_RANGE)) &
               (df['roundness1'].abs() <= 0.5) &
               (df['roundness2'].abs() <= 0.5) &
               (df['snr'] >= SNR_STAR_MIN))
    is_gal  = ((df['sharpness'].between(*SHARP_GAL_RANGE)) &
               (df['roundness1'].abs() <= ROUND_GAL_MAX) &
               (df['roundness2'].abs() <= ROUND_GAL_MAX) &
               (df['snr'] >= SNR_GAL_MIN))
    return df[is_star].reset_index(drop=True), df[is_gal].reset_index(drop=True)


def _sigclip_median(x, sigma=3.0, iters=3):
    x = np.asarray(x)
    for _ in range(iters):
        m  = np.nanmedian(x)
        sd = nmad(x)
        if not np.isfinite(sd) or sd == 0:
            return m
        x = x[np.abs(x - m) <= sigma * sd]
    return float(np.nanmedian(x))


def static_frame_tie(gal_A, gal_B, star_A, star_B):
    """Galaxy-based frame tie when galaxies available (≥30 matches), else
    sigma-clipped median of mutual-NN matched stars (true PM is symmetric,
    so the clipped median is dominated by the static frame offset)."""
    # Try galaxies first
    cA = SkyCoord(gal_A['ra'].values * u.deg, gal_A['dec'].values * u.deg)
    cB = SkyCoord(gal_B['ra'].values * u.deg, gal_B['dec'].values * u.deg)
    mut, idx = mutual_nn(cA, cB, MATCH_AS * u.arcsec)
    n_gal = int(mut.sum())
    if n_gal >= 30:
        rA = gal_A['ra'].values[mut];  dA = gal_A['dec'].values[mut]
        rB = gal_B['ra'].values[idx[mut]]; dB = gal_B['dec'].values[idx[mut]]
        cosd = np.cos(np.deg2rad(dB))
        return (float(np.median((rA - rB) * cosd * 3.6e6)),
                float(np.median((dA - dB) * 3.6e6)),
                n_gal, 'galaxy')
    # Fallback to sigma-clipped star residuals
    cA = SkyCoord(star_A['ra'].values * u.deg, star_A['dec'].values * u.deg)
    cB = SkyCoord(star_B['ra'].values * u.deg, star_B['dec'].values * u.deg)
    mut, idx = mutual_nn(cA, cB, MATCH_AS * u.arcsec)
    n_st = int(mut.sum())
    if n_st == 0:
        return 0.0, 0.0, 0, 'none'
    rA = star_A['ra'].values[mut];  dA = star_A['dec'].values[mut]
    rB = star_B['ra'].values[idx[mut]]; dB = star_B['dec'].values[idx[mut]]
    cosd = np.cos(np.deg2rad(dB))
    return (_sigclip_median((rA - rB) * cosd * 3.6e6),
            _sigclip_median((dA - dB) * 3.6e6),
            n_st, 'star_sigclip')


def main():
    print('Loading common-footprint mask...')
    common, common_wcs = load_common_mask()
    print(f'  common: {int(common.sum()):,} px '
          f'({common.sum()*(3.0/60)**2:.1f} arcmin²)')

    print('\nLoading per-band catalogs restricted to common region:')
    hst_star,  hst_gal  = load_band_with_class('F814W', common, common_wcs)
    jw_star,   jw_gal   = load_band_with_class('F115W', common, common_wcs)
    eu_star,   eu_gal   = load_band_with_class('VIS',   common, common_wcs)
    print(f'  HST  stars/gals : {len(hst_star):,} / {len(hst_gal):,}')
    print(f'  JWST stars/gals : {len(jw_star):,} / {len(jw_gal):,}')
    print(f'  Euclid VIS stars/gals : {len(eu_star):,} / {len(eu_gal):,}')

    print('\nStatic frame ties (mas; falls back to sigma-clipped stars if galaxies < 30):')
    ft_he_ra, ft_he_de, n_he, kind_he = static_frame_tie(hst_gal, eu_gal, hst_star, eu_star)
    ft_je_ra, ft_je_de, n_je, kind_je = static_frame_tie(jw_gal,  eu_gal, jw_star,  eu_star)
    ft_hj_ra, ft_hj_de, n_hj, kind_hj = static_frame_tie(hst_gal, jw_gal, hst_star, jw_star)
    print(f'  HST  - Euclid : ({ft_he_ra:+6.2f}, {ft_he_de:+6.2f}) mas  (N={n_he:,}, {kind_he})')
    print(f'  JWST - Euclid : ({ft_je_ra:+6.2f}, {ft_je_de:+6.2f}) mas  (N={n_je:,}, {kind_je})')
    print(f'  HST  - JWST   : ({ft_hj_ra:+6.2f}, {ft_hj_de:+6.2f}) mas  (N={n_hj:,}, {kind_hj})')

    # Build triple-confirmed star sample: HST star matched in BOTH Euclid AND JWST
    print('\nBuilding triple-confirmed star sample (HST matched in Euclid AND JWST):')
    c_h = SkyCoord(hst_star['ra'].values * u.deg, hst_star['dec'].values * u.deg)
    c_e = SkyCoord(eu_star['ra'].values  * u.deg, eu_star['dec'].values  * u.deg)
    c_j = SkyCoord(jw_star['ra'].values  * u.deg, jw_star['dec'].values  * u.deg)
    mut_he, idx_he = mutual_nn(c_h, c_e, MATCH_AS * u.arcsec)
    mut_hj, idx_hj = mutual_nn(c_h, c_j, MATCH_AS * u.arcsec)
    triple = mut_he & mut_hj
    print(f'  HST stars matched to BOTH Euclid and JWST: {int(triple.sum()):,}')

    sel_h = np.where(triple)[0]
    sel_e = idx_he[triple]
    sel_j = idx_hj[triple]

    # Per-star displacements minus frame tie / baseline
    t = pd.DataFrame()
    t['hst_ra']  = hst_star['ra'].values[sel_h]
    t['hst_dec'] = hst_star['dec'].values[sel_h]
    t['hst_mag'] = hst_star['mag'].values[sel_h]
    t['eu_ra']   = eu_star['ra'].values[sel_e]
    t['eu_dec']  = eu_star['dec'].values[sel_e]
    t['eu_mag']  = eu_star['mag'].values[sel_e]
    t['jw_ra']   = jw_star['ra'].values[sel_j]
    t['jw_dec']  = jw_star['dec'].values[sel_j]
    t['jw_mag']  = jw_star['mag'].values[sel_j]

    cos_d = np.cos(np.deg2rad(t['eu_dec'].values))

    # HST → Euclid (baseline = 19.5 yr)
    dt_he = T_EUCLID - T_HST
    raw_dra_he = (t['hst_ra'].values - t['eu_ra'].values) * cos_d * 3.6e6
    raw_dde_he = (t['hst_dec'].values - t['eu_dec'].values) * 3.6e6
    t['pm_ra_HE']  = -(raw_dra_he - ft_he_ra) / dt_he
    t['pm_dec_HE'] = -(raw_dde_he - ft_he_de) / dt_he
    t['pm_tot_HE'] = np.hypot(t['pm_ra_HE'], t['pm_dec_HE'])

    # JWST → Euclid (baseline = 0.5 yr — large noise)
    dt_je = T_EUCLID - T_JWST
    raw_dra_je = (t['jw_ra'].values - t['eu_ra'].values) * cos_d * 3.6e6
    raw_dde_je = (t['jw_dec'].values - t['eu_dec'].values) * 3.6e6
    t['pm_ra_JE']  = -(raw_dra_je - ft_je_ra) / dt_je
    t['pm_dec_JE'] = -(raw_dde_je - ft_je_de) / dt_je
    t['pm_tot_JE'] = np.hypot(t['pm_ra_JE'], t['pm_dec_JE'])

    # HST → JWST (19 yr)
    dt_hj = T_JWST - T_HST
    cos_d2 = np.cos(np.deg2rad(t['jw_dec'].values))
    raw_dra_hj = (t['hst_ra'].values - t['jw_ra'].values) * cos_d2 * 3.6e6
    raw_dde_hj = (t['hst_dec'].values - t['jw_dec'].values) * 3.6e6
    t['pm_ra_HJ']  = -(raw_dra_hj - ft_hj_ra) / dt_hj
    t['pm_dec_HJ'] = -(raw_dde_hj - ft_hj_de) / dt_hj
    t['pm_tot_HJ'] = np.hypot(t['pm_ra_HJ'], t['pm_dec_HJ'])

    print('\nPer-star PM stats (triple-confirmed, mas/yr):')
    for col, label in [('pm_ra_HE',  'PM RA  (HST→Euclid)'),
                       ('pm_dec_HE', 'PM Dec (HST→Euclid)'),
                       ('pm_tot_HE', '|PM|   (HST→Euclid)'),
                       ('pm_ra_JE',  'PM RA  (JWST→Euclid)'),
                       ('pm_dec_JE', 'PM Dec (JWST→Euclid)'),
                       ('pm_tot_JE', '|PM|   (JWST→Euclid)'),
                       ('pm_ra_HJ',  'PM RA  (HST→JWST)'),
                       ('pm_dec_HJ', 'PM Dec (HST→JWST)'),
                       ('pm_tot_HJ', '|PM|   (HST→JWST)')]:
        v = np.asarray(t[col])
        print(f'  {label:30s}: median={np.median(v):+6.2f}  NMAD={nmad(v):5.2f}  '
              f'95% = {np.percentile(v,95):+6.2f}')

    t.to_csv(OUT / 'triple_morphology_stars.csv', index=False)
    t.to_parquet(OUT / 'triple_morphology_stars.parquet', index=False)
    print(f'\nWrote triple_morphology_stars.csv  ({len(t):,} rows)')

    # ===== Quiver plots (3 pairs) =====
    rng = np.random.default_rng(0)
    n_show = min(1500, len(t))
    samp = rng.choice(len(t), n_show, replace=False)

    def quiver_panel(ax, t_sub, ra_col, dec_col, pm_ra_col, pm_de_col, pm_tot_col,
                     title, vmax=15.0, scale=300):
        q = ax.quiver(np.asarray(t_sub[ra_col]),
                      np.asarray(t_sub[dec_col]),
                      np.asarray(t_sub[pm_ra_col]),
                      np.asarray(t_sub[pm_de_col]),
                      np.asarray(t_sub[pm_tot_col]),
                      cmap='coolwarm', clim=(0, vmax),
                      scale=scale, width=0.0025, alpha=0.85)
        cb = plt.colorbar(q, ax=ax, label='|PM| (mas/yr)')
        ax.set_xlabel('RA (deg)'); ax.set_ylabel('Dec (deg)')
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

    fig, ax = plt.subplots(figsize=(9, 8))
    quiver_panel(ax, t.iloc[samp], 'eu_ra', 'eu_dec',
                 'pm_ra_HE', 'pm_dec_HE', 'pm_tot_HE',
                 f'PM  HST → Euclid  (19.5 yr baseline, N={len(t):,}; shown {n_show:,})',
                 vmax=15.0, scale=300)
    fig.tight_layout()
    fig.savefig(HTML / 'pm_quiver_HST_to_Euclid.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    quiver_panel(ax, t.iloc[samp], 'eu_ra', 'eu_dec',
                 'pm_ra_JE', 'pm_dec_JE', 'pm_tot_JE',
                 f'PM  JWST → Euclid  (0.5 yr baseline, N={len(t):,}; shown {n_show:,})',
                 vmax=120.0, scale=2000)
    fig.tight_layout()
    fig.savefig(HTML / 'pm_quiver_JWST_to_Euclid.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    quiver_panel(ax, t.iloc[samp], 'jw_ra', 'jw_dec',
                 'pm_ra_HJ', 'pm_dec_HJ', 'pm_tot_HJ',
                 f'PM  HST → JWST  (19 yr baseline, N={len(t):,}; shown {n_show:,})',
                 vmax=15.0, scale=300)
    fig.tight_layout()
    fig.savefig(HTML / 'pm_quiver_HST_to_JWST.png', dpi=140); plt.close(fig)
    print('Wrote 3 PM quiver PNGs')

    # ===== Histograms =====
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    bins_l = np.linspace(-30, 30, 80)
    bins_s = np.linspace(-300, 300, 80)
    for irow, (ra_key, dec_key, label, bins) in enumerate([
        ('pm_ra_HE',  'pm_dec_HE', 'HST → Euclid  (19.5 yr)', bins_l),
        ('pm_ra_JE',  'pm_dec_JE', 'JWST → Euclid (0.5 yr)',  bins_s),
        ('pm_ra_HJ',  'pm_dec_HJ', 'HST → JWST  (19 yr)',     bins_l),
    ]):
        for icol, (key, ax_lbl) in enumerate([
                (ra_key,  r'$\mu_\alpha\cos\delta$ (mas/yr)'),
                (dec_key, r'$\mu_\delta$ (mas/yr)')]):
            ax = axes[irow, icol]
            v = t[key].values
            ax.hist(v, bins=bins, color=f'C{irow}', histtype='step', lw=2)
            ax.axvline(0, color='0.6', lw=0.6, ls='--')
            ax.set_xlabel(ax_lbl)
            ax.set_title(f'{label}  med={np.median(v):+.2f}  NMAD={nmad(v):.2f}')
            ax.grid(alpha=0.3)
    fig.suptitle(f'Proper motion of {len(t):,} triple-confirmed stars\n'
                 f'(galaxy frame tie subtracted; common HST∩JWST region)')
    fig.tight_layout()
    fig.savefig(HTML / 'pm_three_pairs_histograms.png', dpi=140); plt.close(fig)
    print('Wrote pm_three_pairs_histograms.png')


if __name__ == '__main__':
    main()
