#!/usr/bin/env python
"""
57_step7_pm_v2.py — Step 7 v2: proper-motion measurement with Gaia-only
supplements and per-source errors.

Improvements over v1:
  1. SIGN CONVENTION (user's paper convention): for each pair
     (m1, m2) where m2 is the reference (more recent / Gaia-tied),
       Δra  = (ra_m1 - ra_m2) × cos(dec) × 3.6e6   (mas)
       Δdec = (dec_m1 - dec_m2) × 3.6e6            (mas)
       pmra = Δra  / (epoch_m1 - epoch_m2)         (mas/yr)
       pmdec = Δdec / (epoch_m1 - epoch_m2)         (mas/yr)
     Pairs:
       HST_Euclid   : HST  −  Euclid   (reference: Euclid)
       HST_JWST     : HST  −  JWST     (reference: JWST)
       JWST_Euclid  : JWST −  Euclid   (reference: Euclid)
  2. GAIA-ONLY SUPPLEMENTS: when a star is gaia_only_bright in one
     mission but catalog-detected in the other (the typical case
     for bright high-PM stars that saturate in HST/JWST), we add
     the source to the PM table using gaia_ra_prop on the saturated
     side.  These rows are tagged pm_method='gaia_supplement_<side>'
     and plotted with DOTTED quiver lines.
  3. THREE CATEGORIES per combo: all / stars (PS − AGN) / AGN/QSO.
     Independent quiver plot per category — 5 combos × 3 = 15.
  4. PM ERRORS: per-source σ_axis built from per-mission systematic +
     SNR-dependent centroid (σ_cent = FWHM_psf × magerr or FWHM/SNR).
     σ_pmra = √(σ_axis_m1² + σ_axis_m2²) / |Δt|
     σ_pmdec same.  σ_pmtot via uncertainty propagation.

Output:
  csvfiles_star/refined_<combo>_with_pm.parquet  — intersection + gaia supplements
                                                   with new PM and error cols.
  htmls/pm_v01/pm_<combo>_quiver_<category>.png  — 15 quivers
  htmls/pm_v01/pm_<combo>_scatter.png            — 3-panel scatter (rebuilt)
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

warnings.filterwarnings('ignore')

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'pm_v01'
HTML.mkdir(parents=True, exist_ok=True)
CW_PATH = '/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits'

# (combo_name, mission1, mission2, tag1, tag2, ep1, ep2)
# tag is the short suffix used in the refined catalog (e.g. 'hst').
# m1 - m2 with m2 the reference (more recent / Gaia-tied).
COMBOS_PM = [
    ('HST_Euclid_VIS',   'HST',  'Euclid_VIS',  'hst',  'vis',  2005.0, 2024.5),
    ('HST_Euclid_NISP',  'HST',  'Euclid_NISP', 'hst',  'nisp', 2005.0, 2024.5),
    ('JWST_Euclid_VIS',  'JWST', 'Euclid_VIS',  'jwst', 'vis',  2024.0, 2024.5),
    ('JWST_Euclid_NISP', 'JWST', 'Euclid_NISP', 'jwst', 'nisp', 2024.0, 2024.5),
    ('HST_JWST',         'HST',  'JWST',        'hst',  'jwst', 2005.0, 2024.0),
]

# Per-mission systematic positional uncertainty per axis (mas)
SIGMA_SYS = {'HST': 30.0, 'JWST': 10.0, 'Euclid_VIS': 15.0, 'Euclid_NISP': 30.0}
# Per-mission PSF FWHM (mas), used for SNR-dependent centroid noise
FWHM_PSF = {'HST': 100.0, 'JWST': 40.0, 'Euclid_VIS': 140.0, 'Euclid_NISP': 350.0}
# Per-mission magerr column / snr column used for σ_centroid
ERR_COL = {
    'HST':         ('cat_magerr_F814W', 'magerr'),
    'JWST':        ('cat_snr_F115W',    'snr'),
    'Euclid_VIS':  ('cat_magerr_VIS',   'magerr'),
    'Euclid_NISP': ('cat_magerr_NIR_Y', 'magerr'),
}


def load_cw_agn() -> pd.DataFrame:
    with fits.open(CW_PATH) as hdul:
        photo = hdul['PHOTOMETRY HOTCOLD AND SE++'].data
        lephare = hdul['LEPHARE'].data
        ids = np.asarray(photo['id']).astype(np.int64)
        ra  = np.asarray(photo['ra']).astype(float)
        dec = np.asarray(photo['dec']).astype(float)
        lp_type = np.asarray(lephare['type']).astype(np.int64)
        chandra = np.asarray(lephare['flag_chandra']).astype(float) > 0.5
    agn_mask = (lp_type == 2) | chandra
    return pd.DataFrame({
        'cw_id':  ids[agn_mask],
        'cw_ra':  ra[agn_mask],
        'cw_dec': dec[agn_mask],
    })


def sigma_axis(mission: str, err_arr: np.ndarray, err_kind: str) -> np.ndarray:
    """σ per axis in mas: sqrt(σ_sys² + σ_centroid²) using per-source err."""
    s_sys = SIGMA_SYS[mission]
    fwhm  = FWHM_PSF[mission]
    if err_kind == 'magerr':
        # σ_centroid ≈ FWHM × magerr (mas), with floor against zero
        s_cent = fwhm * np.where(np.isfinite(err_arr) & (err_arr > 0), err_arr, 0.05)
    else:  # snr
        s_cent = fwhm / np.where(np.isfinite(err_arr) & (err_arr > 0), err_arr, 5.0)
    return np.sqrt(s_sys**2 + s_cent**2)


def nmad(x):
    m = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - m))


def build_pm_table(combo_name, m1, m2, tag1, tag2, ep1, ep2, cw_agn):
    """Return DataFrame with one row per source available for PM."""
    # ── 1. Existing intersection (cat - cat) ──────────────────────────────
    int_df = pd.read_parquet(OUT / f'refined_{combo_name}_v03.parquet')
    base = pd.DataFrame({
        'pm_method':    np.full(len(int_df), 'cat_cat', dtype=object),
        'ra_m1':        int_df[f'cat_ra_{tag1}'].values,
        'dec_m1':       int_df[f'cat_dec_{tag1}'].values,
        'ra_m2':        int_df[f'cat_ra_{tag2}'].values,
        'dec_m2':       int_df[f'cat_dec_{tag2}'].values,
        'is_point_source': int_df['is_point_source'].values,
        'is_pt_m1':     int_df.get(f'is_point_source_{tag1}', pd.Series(False, index=int_df.index)).values,
        'is_pt_m2':     int_df.get(f'is_point_source_{tag2}', pd.Series(False, index=int_df.index)).values,
        'gaia_source_id': int_df['gaia_source_id'].values if 'gaia_source_id' in int_df.columns else np.nan,
        'gaia_g':       int_df.get('gaia_phot_g_mean_mag', pd.Series(np.nan, index=int_df.index)).values,
        'gaia_pmra':    int_df.get('gaia_pmra', pd.Series(np.nan, index=int_df.index)).values,
        'gaia_pmdec':   int_df.get('gaia_pmdec', pd.Series(np.nan, index=int_df.index)).values,
    })
    # Per-mission err for sigma
    err_col1, err_kind1 = ERR_COL[m1]
    err_col2, err_kind2 = ERR_COL[m2]
    base['err_m1']   = int_df.get(f'{err_col1}_{tag1}', pd.Series(np.nan, index=int_df.index)).values
    base['err_m2']   = int_df.get(f'{err_col2}_{tag2}', pd.Series(np.nan, index=int_df.index)).values
    # AGN flag — derive below after concat for uniform treatment
    base['jwst_id'] = int_df.get('jwst_id_jwst', pd.Series(np.nan, index=int_df.index)).values
    if 'cat_ra_jwst' in int_df.columns:
        base['_match_ra']  = int_df['cat_ra_jwst'].values
        base['_match_dec'] = int_df['cat_dec_jwst'].values
    else:
        base['_match_ra']  = int_df[f'cat_ra_{tag2}'].values
        base['_match_dec'] = int_df[f'cat_dec_{tag2}'].values

    # ── 2. Supplements: gaia_only_bright in one mission + catalog in other ─
    df1 = pd.read_parquet(OUT / f'cat_matched_{m1}_with_gaia.parquet')
    df2 = pd.read_parquet(OUT / f'cat_matched_{m2}_with_gaia.parquet')
    df1_gob = df1[df1['source_type'] == 'gaia_only_bright'].copy()
    df2_gob = df2[df2['source_type'] == 'gaia_only_bright'].copy()
    df1_cat = df1[df1['source_type'].isin(['catalog', 'catalog+gaia'])].copy()
    df2_cat = df2[df2['source_type'].isin(['catalog', 'catalog+gaia'])].copy()
    # Dedup multi-mapped Gaia ids: when catalog has multiple rows per Gaia
    # source (e.g., MER split detections), keep the closest-to-Gaia row.
    if 'gaia_sep_as' in df1_cat.columns:
        df1_cat = df1_cat.sort_values('gaia_sep_as').drop_duplicates(
            subset=['gaia_source_id'], keep='first')
    if 'gaia_sep_as' in df2_cat.columns:
        df2_cat = df2_cat.sort_values('gaia_sep_as').drop_duplicates(
            subset=['gaia_source_id'], keep='first')
    # m1 gaia_only_bright + m2 catalog
    sup1 = df1_gob.merge(df2_cat, on='gaia_source_id', suffixes=('_a', '_b'))
    if len(sup1):
        sup1_df = pd.DataFrame({
            'pm_method':       'gaia_m1',
            'ra_m1':           sup1['gaia_ra_prop_a'].values,    # propagated to m1 epoch
            'dec_m1':          sup1['gaia_dec_prop_a'].values,
            'ra_m2':           sup1['cat_ra_b'].values,
            'dec_m2':          sup1['cat_dec_b'].values,
            'is_point_source': True,        # gaia_only_bright is a confirmed Gaia source
            'is_pt_m1':        True,
            'is_pt_m2':        True,
            'gaia_source_id':  sup1['gaia_source_id'].values,
            'gaia_g':          sup1['phot_g_mean_mag_a'].values if 'phot_g_mean_mag_a' in sup1.columns else
                                sup1['gaia_phot_g_mean_mag_a' ].values if 'gaia_phot_g_mean_mag_a' in sup1.columns else np.nan,
            'gaia_pmra':       sup1.get('pmra_a', sup1.get('gaia_pmra_a', np.nan)).values
                                if 'pmra_a' in sup1.columns or 'gaia_pmra_a' in sup1.columns else np.nan,
            'gaia_pmdec':      sup1.get('pmdec_a', sup1.get('gaia_pmdec_a', np.nan)).values
                                if 'pmdec_a' in sup1.columns or 'gaia_pmdec_a' in sup1.columns else np.nan,
            'err_m1':          np.nan,      # no catalog measurement on m1 side
            'err_m2':          sup1.get(err_col2 + '_b', np.nan).values
                                if err_col2 + '_b' in sup1.columns else np.nan,
            'jwst_id':         sup1.get('jwst_id_a', sup1.get('jwst_id_b', np.nan)).values
                                if 'jwst_id_a' in sup1.columns or 'jwst_id_b' in sup1.columns else np.nan,
            '_match_ra':       sup1['cat_ra_b'].values,
            '_match_dec':      sup1['cat_dec_b'].values,
        })
        base = pd.concat([base, sup1_df], ignore_index=True)
    # m1 catalog + m2 gaia_only_bright
    sup2 = df1_cat.merge(df2_gob, on='gaia_source_id', suffixes=('_a', '_b'))
    if len(sup2):
        sup2_df = pd.DataFrame({
            'pm_method':       'gaia_m2',
            'ra_m1':           sup2['cat_ra_a'].values,
            'dec_m1':          sup2['cat_dec_a'].values,
            'ra_m2':           sup2['gaia_ra_prop_b'].values,
            'dec_m2':          sup2['gaia_dec_prop_b'].values,
            'is_point_source': True,
            'is_pt_m1':        True,
            'is_pt_m2':        True,
            'gaia_source_id':  sup2['gaia_source_id'].values,
            'gaia_g':          sup2.get('phot_g_mean_mag_b', sup2.get('gaia_phot_g_mean_mag_b', np.nan)).values
                                if 'phot_g_mean_mag_b' in sup2.columns or 'gaia_phot_g_mean_mag_b' in sup2.columns else np.nan,
            'gaia_pmra':       sup2.get('pmra_b', sup2.get('gaia_pmra_b', np.nan)).values
                                if 'pmra_b' in sup2.columns or 'gaia_pmra_b' in sup2.columns else np.nan,
            'gaia_pmdec':      sup2.get('pmdec_b', sup2.get('gaia_pmdec_b', np.nan)).values
                                if 'pmdec_b' in sup2.columns or 'gaia_pmdec_b' in sup2.columns else np.nan,
            'err_m1':          sup2.get(err_col1 + '_a', np.nan).values
                                if err_col1 + '_a' in sup2.columns else np.nan,
            'err_m2':          np.nan,
            'jwst_id':         sup2.get('jwst_id_a', sup2.get('jwst_id_b', np.nan)).values
                                if 'jwst_id_a' in sup2.columns or 'jwst_id_b' in sup2.columns else np.nan,
            '_match_ra':       sup2['cat_ra_a'].values,
            '_match_dec':      sup2['cat_dec_a'].values,
        })
        base = pd.concat([base, sup2_df], ignore_index=True)

    return base


def process_combo(args) -> dict:
    combo_name, m1, m2, tag1, tag2, ep1, ep2 = args
    cw_agn = load_cw_agn()
    t0 = time.time()
    print(f'[{combo_name}] starting (Δt = {ep1-ep2:+.2f} yr,  '
          f'sign: {m1}−{m2}=ref)')

    pm = build_pm_table(combo_name, m1, m2, tag1, tag2, ep1, ep2, cw_agn)
    n_total = len(pm)
    print(f'[{combo_name}]   table rows: {n_total:,}   '
          f'(cat_cat: {(pm.pm_method=="cat_cat").sum():,}, '
          f'gaia_m1: {(pm.pm_method=="gaia_m1").sum():,}, '
          f'gaia_m2: {(pm.pm_method=="gaia_m2").sum():,})')

    # ── Compute Δposition and PM (user's sign convention: m1 − m2) ────────
    cosd = np.cos(np.deg2rad(pm['dec_m1'].values))
    dra_mas  = (pm['ra_m1'].values  - pm['ra_m2'].values)  * cosd * 3.6e6
    ddec_mas = (pm['dec_m1'].values - pm['dec_m2'].values) * 3.6e6
    dt = ep1 - ep2   # negative for HST−Euclid etc.
    pmra  = dra_mas  / dt
    pmdec = ddec_mas / dt
    pmtot = np.sqrt(pmra**2 + pmdec**2)

    # ── Bad-match filter ───────────────────────────────────────────────────
    # Cap on plausible PM.  The fastest Gaia star in COSMOS has PM ≈ 497
    # mas/yr; allow up to 1000 mas/yr.  Anything larger must be a wrong
    # cross-match (Gaia source_id linked to two unrelated catalog rows).
    PM_MAX_PLAUSIBLE = 1000.0   # mas/yr
    bad = np.abs(pmtot) > PM_MAX_PLAUSIBLE
    n_bad = int(bad.sum())
    pmra[bad]  = np.nan
    pmdec[bad] = np.nan
    pmtot[bad] = np.nan
    print(f'[{combo_name}]   |pmtot|>{PM_MAX_PLAUSIBLE} outliers NaN\'d: '
          f'{n_bad} (likely wrong-source matches)')

    # ── PM errors ─────────────────────────────────────────────────────────
    err1_kind = ERR_COL[m1][1]
    err2_kind = ERR_COL[m2][1]
    s1 = sigma_axis(m1, pm['err_m1'].values, err1_kind)
    s2 = sigma_axis(m2, pm['err_m2'].values, err2_kind)
    # For gaia_m1 rows, m1 side error = Gaia astrometric (~1 mas/yr × |Δt|)
    is_gaia_m1 = (pm['pm_method'] == 'gaia_m1').values
    is_gaia_m2 = (pm['pm_method'] == 'gaia_m2').values
    s1[is_gaia_m1] = 5.0   # Gaia-propagated to m1 epoch — small intrinsic error
    s2[is_gaia_m2] = 5.0
    sigma_pos = np.sqrt(s1**2 + s2**2)        # per-axis position error budget
    pmra_err  = sigma_pos / abs(dt)
    pmdec_err = sigma_pos / abs(dt)
    with np.errstate(invalid='ignore', divide='ignore'):
        pmtot_err = np.where(
            pmtot > 0,
            np.sqrt(((pmra*pmra_err)**2 + (pmdec*pmdec_err)**2)) / pmtot,
            np.sqrt(pmra_err**2 + pmdec_err**2) / np.sqrt(2),
        )

    pm['dra_mas']  = dra_mas
    pm['ddec_mas'] = ddec_mas
    pm['dt_yr']    = dt
    pm['pmra']     = pmra
    pm['pmdec']    = pmdec
    pm['pmtot']    = pmtot
    pm['pmra_err'] = pmra_err
    pm['pmdec_err']= pmdec_err
    pm['pmtot_err']= pmtot_err

    # ── AGN/QSO flag ──────────────────────────────────────────────────────
    has_jwst = ('JWST' in (m1, m2))
    if has_jwst and pm['jwst_id'].notna().any():
        jid = pd.to_numeric(pm['jwst_id'], errors='coerce').astype('Int64')
        is_qso = jid.isin(cw_agn['cw_id'].astype('Int64')).values
    else:
        cq = SkyCoord(cw_agn['cw_ra'].values * u.deg, cw_agn['cw_dec'].values * u.deg)
        cc = SkyCoord(pm['_match_ra'].values * u.deg, pm['_match_dec'].values * u.deg)
        idx_q, idx_c, sep, _ = search_around_sky(cq, cc, 0.3*u.arcsec)
        is_qso = np.zeros(n_total, dtype=bool)
        is_qso[np.unique(idx_c)] = True

    pm['is_agn_qso']   = is_qso
    pm['is_star_only'] = pm['is_point_source'].values & ~is_qso

    # ── Summary ───────────────────────────────────────────────────────────
    lines = [f'Combo: {combo_name}  ({m1} − {m2})  Δt = {dt:+.2f} yr  '
              f'(reference = {m2})',
             f'N total: {n_total:,}   '
             f'cat_cat={int((pm.pm_method=="cat_cat").sum()):,}   '
             f'gaia_m1={int((pm.pm_method=="gaia_m1").sum()):,}   '
             f'gaia_m2={int((pm.pm_method=="gaia_m2").sum()):,}',
             '',
             f'{"category":<15} {"N":>8} {"med pmra":>10} {"med pmdec":>10} '
             f'{"NMAD pra":>9} {"NMAD pdec":>10} {"med pmtot":>10} {"95% pmtot":>10} {"max pmtot":>10}']
    lines.append('-' * 105)
    for cat_name, mask in [
        ('all',     np.ones(n_total, dtype=bool)),
        ('stars',   pm['is_star_only'].values),
        ('agn_qso', is_qso),
    ]:
        n = int(mask.sum())
        if n == 0:
            continue
        m_pra = np.nanmedian(pmra[mask]);    m_pdc = np.nanmedian(pmdec[mask])
        n_pra = nmad(pmra[mask]);            n_pdc = nmad(pmdec[mask])
        m_pt  = np.nanmedian(pmtot[mask]);   p95   = np.nanpercentile(pmtot[mask], 95)
        mx    = np.nanmax(pmtot[mask])
        lines.append(
            f'{cat_name:<15} {n:>8,} {m_pra:>+10.2f} {m_pdc:>+10.2f} '
            f'{n_pra:>9.2f} {n_pdc:>10.2f} {m_pt:>10.2f} {p95:>10.2f} {mx:>10.1f}'
        )
    summary = '\n'.join(lines)
    print(summary)
    (HTML / f'pm_{combo_name}_summary.txt').write_text(summary + '\n')

    # ── 3-panel scatter ───────────────────────────────────────────────────
    # axis limit: 99% of max(pmtot) so we see the high-PM tail
    lim = max(20.0, float(np.nanpercentile(pmtot, 99.5)) * 1.1) if np.any(np.isfinite(pmtot)) else 50.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    for ax, (cat, mask, col) in zip(axes, [
        ('all',     np.ones(n_total, dtype=bool), 'C7'),
        ('stars',   pm['is_star_only'].values,    'C0'),
        ('agn_qso', is_qso,                       'C3')]):
        cc_mask = mask & (pm['pm_method'].values == 'cat_cat')
        gs_mask = mask & (pm['pm_method'].values != 'cat_cat')
        n_cc = int(cc_mask.sum()); n_gs = int(gs_mask.sum())
        if n_cc:
            ax.scatter(pmra[cc_mask], pmdec[cc_mask], s=3, alpha=0.4,
                       color=col, rasterized=True, label=f'cat-cat (N={n_cc:,})')
        if n_gs:
            ax.scatter(pmra[gs_mask], pmdec[gs_mask], s=10, alpha=0.6,
                       facecolors='none', edgecolors=col, rasterized=True,
                       label=f'gaia-suppl (N={n_gs:,})')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_xlabel(f'pmra ({m1}−{m2}) cos δ  (mas/yr)')
        ax.set_title(f'{cat}  (N_total={int(mask.sum()):,})', fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper right')
    axes[0].set_ylabel(f'pmdec ({m1}−{m2})  (mas/yr)')
    fig.suptitle(f'{combo_name}  Δt = {dt:+.2f} yr  ({m1}−{m2}, reference = {m2})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(HTML / f'pm_{combo_name}_scatter.png', dpi=130)
    plt.close(fig)

    # ── 3 independent quivers per combo ───────────────────────────────────
    for cat_name, mask, col in [
        ('all',     np.ones(n_total, dtype=bool), 'C7'),
        ('stars',   pm['is_star_only'].values,    'C0'),
        ('agn_qso', is_qso,                       'C3')]:
        sub = pm[mask]
        sub_pmra = pmra[mask]; sub_pmdec = pmdec[mask]
        sub_pmtot = pmtot[mask]
        finite = np.isfinite(sub_pmra) & np.isfinite(sub_pmdec)
        sub = sub[finite]; sub_pmra = sub_pmra[finite]
        sub_pmdec = sub_pmdec[finite]; sub_pmtot = sub_pmtot[finite]
        if len(sub) == 0:
            continue
        is_cc = (sub['pm_method'].values == 'cat_cat')
        is_gs = ~is_cc
        # Quiver scale: 50 mas/yr per inch
        fig, ax = plt.subplots(figsize=(8, 7))
        scale = 1500
        # Downsample dense cat_cat if too many
        sub_cc = sub[is_cc]; sub_gs = sub[is_gs]
        cc_pmra = sub_pmra[is_cc]; cc_pmdec = sub_pmdec[is_cc]; cc_pmtot = sub_pmtot[is_cc]
        gs_pmra = sub_pmra[is_gs]; gs_pmdec = sub_pmdec[is_gs]; gs_pmtot = sub_pmtot[is_gs]
        if len(sub_cc) > 3000:
            sel = np.random.choice(len(sub_cc), 3000, replace=False)
            sub_cc = sub_cc.iloc[sel]
            cc_pmra = cc_pmra[sel]; cc_pmdec = cc_pmdec[sel]; cc_pmtot = cc_pmtot[sel]
        if len(sub_cc):
            ax.quiver(sub_cc['ra_m1'], sub_cc['dec_m1'], cc_pmra, cc_pmdec,
                      cc_pmtot, cmap='viridis', clim=(0, 50), scale=scale,
                      width=0.0015, alpha=0.7,
                      label=f'cat-cat (N={len(sub_cc):,})')
        if len(sub_gs):
            ax.quiver(sub_gs['ra_m1'], sub_gs['dec_m1'], gs_pmra, gs_pmdec,
                      gs_pmtot, cmap='plasma', clim=(0, 500), scale=scale,
                      width=0.002, alpha=0.9, linestyle='dotted',
                      linewidths=0.8, edgecolors='red',
                      label=f'gaia-suppl (N={len(sub_gs):,})')
        ax.invert_xaxis()
        ax.set_xlabel('RA (deg)'); ax.set_ylabel('Dec (deg)')
        ax.set_title(f'{combo_name} — {cat_name} '
                     f'(Δt={dt:+.2f} yr, sign: {m1}−{m2})  N={len(sub):,}')
        ax.set_aspect(1.0/np.cos(np.deg2rad(np.median(sub['dec_m1']))))
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(HTML / f'pm_{combo_name}_quiver_{cat_name}.png', dpi=130)
        plt.close(fig)

    # ── Save augmented refined catalog ───────────────────────────────────
    # Start from the original intersection so we preserve all its columns,
    # then append supplement rows with extended columns NaN'd.
    int_df = pd.read_parquet(OUT / f'refined_{combo_name}_v03.parquet')
    n_cat_cat = (pm['pm_method'].values == 'cat_cat').sum()
    assert n_cat_cat == len(int_df), \
        f'cat_cat row count mismatch: {n_cat_cat} vs {len(int_df)}'
    # Map cat_cat rows back to int_df order — same order since we built base
    # from int_df.  Concatenate the supplement rows after.
    out = int_df.copy()
    out['dra_mas']    = dra_mas[:len(int_df)]
    out['ddec_mas']   = ddec_mas[:len(int_df)]
    out['dt_yr']      = dt
    out['pmra']       = pmra[:len(int_df)]
    out['pmdec']      = pmdec[:len(int_df)]
    out['pmtot']      = pmtot[:len(int_df)]
    out['pmra_err']   = pmra_err[:len(int_df)]
    out['pmdec_err']  = pmdec_err[:len(int_df)]
    out['pmtot_err']  = pmtot_err[:len(int_df)]
    out['pm_method']  = pm['pm_method'].values[:len(int_df)]
    out['is_agn_qso'] = is_qso[:len(int_df)]
    out['is_star_only'] = pm['is_star_only'].values[:len(int_df)]
    # Supplement rows: build a thin DataFrame, fill columns we have
    sup_mask = pm['pm_method'].values != 'cat_cat'
    if sup_mask.any():
        sup = pm[sup_mask].copy()
        # Use a minimal column set
        sup_out = pd.DataFrame({c: np.nan for c in out.columns}, index=range(sup_mask.sum()))
        # We can fill cat_ra_<tag1>/_<tag2> if available (NaN on the gaia_only side)
        sup_out[f'cat_ra_{tag1}']  = sup['ra_m1'].values
        sup_out[f'cat_dec_{tag1}'] = sup['dec_m1'].values
        sup_out[f'cat_ra_{tag2}']  = sup['ra_m2'].values
        sup_out[f'cat_dec_{tag2}'] = sup['dec_m2'].values
        sup_out['gaia_source_id']  = sup['gaia_source_id'].values
        sup_out['gaia_phot_g_mean_mag'] = sup['gaia_g'].values
        sup_out['gaia_pmra']  = sup['gaia_pmra'].values
        sup_out['gaia_pmdec'] = sup['gaia_pmdec'].values
        sup_out['dra_mas']    = sup['dra_mas'].values
        sup_out['ddec_mas']   = sup['ddec_mas'].values
        sup_out['dt_yr']      = dt
        sup_out['pmra']       = sup['pmra'].values
        sup_out['pmdec']      = sup['pmdec'].values
        sup_out['pmtot']      = sup['pmtot'].values
        sup_out['pmra_err']   = sup['pmra_err'].values
        sup_out['pmdec_err']  = sup['pmdec_err'].values
        sup_out['pmtot_err']  = sup['pmtot_err'].values
        sup_out['pm_method']  = sup['pm_method'].values
        sup_out['is_agn_qso'] = sup['is_agn_qso'].values
        sup_out['is_star_only'] = sup['is_star_only'].values
        sup_out['is_point_source'] = sup['is_point_source'].values
        sup_out[f'detected_in_{tag1}'] = sup['pm_method'].values == 'gaia_m2'
        sup_out[f'detected_in_{tag2}'] = sup['pm_method'].values == 'gaia_m1'
        sup_out['tie_type'] = 'gaia_anchored'
        sup_out['source_type_pm'] = sup['pm_method'].values
        # Renumber star_ids for supplement rows
        sup_out['star_id'] = np.arange(out['star_id'].max() + 1,
                                       out['star_id'].max() + 1 + len(sup_out))
        out = pd.concat([out, sup_out], ignore_index=True)

    out_parq = OUT / f'refined_{combo_name}_v03_with_pm_clean.parquet'
    out.to_parquet(out_parq, index=False)
    csv_written = False
    if len(out) * len(out.columns) < 200_000_000:
        out.to_csv(OUT / f'refined_{combo_name}_v03_with_pm_clean.csv', index=False)
        csv_written = True

    elapsed = time.time() - t0
    print(f'[{combo_name}] wrote {out_parq.name}: {len(out):,} rows  in {elapsed:.1f}s')
    return {
        'combo': combo_name,
        'dt_yr': dt,
        'n_total': len(out),
        'n_cat_cat': int(n_cat_cat),
        'n_gaia_m1': int((pm['pm_method'].values == 'gaia_m1').sum()),
        'n_gaia_m2': int((pm['pm_method'].values == 'gaia_m2').sum()),
        'n_agn': int(is_qso.sum()),
        'n_stars': int(pm['is_star_only'].sum()),
        'elapsed_s': elapsed,
    }


def main():
    t0 = time.time()
    print('=' * 78)
    print('Step 7 v2 — PM with Gaia-only supplements + 3-cat quivers + errors')
    print('=' * 78)
    results = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(process_combo, args): args[0] for args in COMBOS_PM}
        for fut in as_completed(futures):
            results.append(fut.result())

    print()
    print('=' * 78)
    print('Summary')
    print('=' * 78)
    lines = [f'{"combo":<22} {"Δt":>7} {"n_tot":>8} {"cat-cat":>8} '
             f'{"gaia_m1":>8} {"gaia_m2":>8} {"n_star":>8} {"n_agn":>6} {"time":>6}']
    lines.append('-' * 95)
    for r in sorted(results, key=lambda x: -abs(x['dt_yr'])):
        lines.append(
            f'{r["combo"]:<22} {r["dt_yr"]:>6.2f}y {r["n_total"]:>8,} '
            f'{r["n_cat_cat"]:>8,} {r["n_gaia_m1"]:>8,} {r["n_gaia_m2"]:>8,} '
            f'{r["n_stars"]:>8,} {r["n_agn"]:>6,} {r["elapsed_s"]:>5.1f}s'
        )
    text = '\n'.join(lines)
    print(text)
    (HTML / 'pm_v2_summary_all.txt').write_text(text + '\n')
    print(f'\nWall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
