#!/usr/bin/env python
"""
56_step7_proper_motion.py — Step 7: proper-motion measurement on 6
refined cross-matched catalogs.

For each pair (mission1, mission2), compute apparent proper motion from
the two missions' catalog positions:
  Δra (mas) = (ra_m2 - ra_m1) × cos(dec) × 3.6e6
  Δdec(mas) = (dec_m2 - dec_m1) × 3.6e6
  pmra  = Δra  / (epoch_m2 - epoch_m1)   [mas/yr]
  pmdec = Δdec / (epoch_m2 - epoch_m1)   [mas/yr]
  pmtot = sqrt(pmra² + pmdec²)

Mission epochs:
  HST/ACS F814W   2005.0
  JWST COSMOS-Web 2024.0
  Euclid VIS      2024.5
  Euclid NISP     2024.5

Special case: Euclid_VIS × Euclid_NISP has Δt = 0; we report Δposition
(astrometric residual) but not PM.

Per combo, split into 3 categories:
  (a) all sources           — every row in refined catalog
  (b) stars   = is_point_source AND NOT AGN/QSO
  (c) agn_qso = AGN/QSO from COSMOS-Web (lp_type==2 OR flag_chandra)

Output per combo:
  csvfiles_star/refined_<combo>_with_pm.parquet   (full table + pm cols)
  htmls/pm_v01/pm_<combo>_scatter.png             (pmra vs pmdec 3-panel)
  htmls/pm_v01/pm_<combo>_quiver.png              (sky quiver, stars only)
  htmls/pm_v01/pm_<combo>_summary.txt             (median + NMAD per cat)
"""
from __future__ import annotations
import os
import sys
import time
import warnings
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

CW_PATH = '/Volumes/exdisk1/data/catalog/JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits'

# Each entry: (combo_name, mission1, mission2, tag1, tag2, epoch1, epoch2)
COMBOS_PM = [
    ('HST_Euclid_VIS',          'HST',         'Euclid_VIS',   'hst',  'vis',  2005.0, 2024.5),
    ('HST_Euclid_NISP',         'HST',         'Euclid_NISP',  'hst',  'nisp', 2005.0, 2024.5),
    ('JWST_Euclid_VIS',         'JWST',        'Euclid_VIS',   'jwst', 'vis',  2024.0, 2024.5),
    ('JWST_Euclid_NISP',        'JWST',        'Euclid_NISP',  'jwst', 'nisp', 2024.0, 2024.5),
    ('HST_JWST',                'HST',         'JWST',         'hst',  'jwst', 2005.0, 2024.0),
    # Euclid_VIS × Euclid_NISP skipped: Δt=0 (MER assigns one RA/Dec per source
    # across all Euclid bands, so the PM/residual is zero by construction).
]


def load_cw_agn() -> pd.DataFrame:
    """Load COSMOS-Web AGN/QSO list (id + ra/dec) for position-matching."""
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


def nmad(x):
    """Normalized MAD = 1.4826 × median(|x - median(x)|)."""
    m = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - m))


def _scatter_panel(ax, x, y, title, lim_mas, color='C0'):
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum()
    ax.scatter(x[ok], y[ok], s=2, alpha=0.4, rasterized=True, color=color)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-lim_mas, lim_mas)
    ax.set_ylim(-lim_mas, lim_mas)
    ax.set_aspect('equal')
    if n > 0:
        mra, mdec = np.nanmedian(x[ok]), np.nanmedian(y[ok])
        nra, ndec = nmad(x[ok]), nmad(y[ok])
        stats = f'N={n:,}\nmed=({mra:+.1f}, {mdec:+.1f})\nNMAD=({nra:.1f}, {ndec:.1f})'
        ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                va='top', ha='left', fontsize=8, family='monospace',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def process_combo(args) -> dict:
    combo_name, m1, m2, tag1, tag2, ep1, ep2 = args
    cw_agn = load_cw_agn()
    t0 = time.time()
    print(f'[{combo_name}] starting (Δt = {ep2-ep1:+.2f} yr)')

    df = pd.read_parquet(OUT / f'refined_{combo_name}.parquet')
    n_total = len(df)

    # Compute Δposition in mas
    ra1 = df[f'cat_ra_{tag1}'].values
    ra2 = df[f'cat_ra_{tag2}'].values
    dc1 = df[f'cat_dec_{tag1}'].values
    dc2 = df[f'cat_dec_{tag2}'].values
    cosd = np.cos(np.deg2rad(dc1))
    dra_mas  = (ra2 - ra1) * cosd * 3.6e6
    ddec_mas = (dc2 - dc1) * 3.6e6
    dt = ep2 - ep1

    if dt > 0:
        pmra  = dra_mas  / dt
        pmdec = ddec_mas / dt
        pmtot = np.sqrt(pmra**2 + pmdec**2)
    else:
        pmra  = np.full(n_total, np.nan)
        pmdec = np.full(n_total, np.nan)
        pmtot = np.full(n_total, np.nan)

    df['dra_mas']   = dra_mas
    df['ddec_mas']  = ddec_mas
    df['dt_yr']     = dt
    df[f'pmra_{combo_name}']  = pmra
    df[f'pmdec_{combo_name}'] = pmdec
    df[f'pmtot_{combo_name}'] = pmtot

    # ── AGN/QSO mask: JWST-id lookup if JWST in combo, else position match
    has_jwst = ('JWST' in (m1, m2))
    if has_jwst and 'jwst_id_jwst' in df.columns:
        jid = pd.to_numeric(df['jwst_id_jwst'], errors='coerce').astype('Int64')
        is_qso = jid.isin(cw_agn['cw_id'].astype('Int64')).values
    else:
        ra_col = f'cat_ra_{tag2}' if not np.all(np.isnan(ra2)) else f'cat_ra_{tag1}'
        dec_col = ra_col.replace('cat_ra_', 'cat_dec_')
        cc = SkyCoord(df[ra_col].values * u.deg, df[dec_col].values * u.deg)
        cq = SkyCoord(cw_agn['cw_ra'].values * u.deg,
                      cw_agn['cw_dec'].values * u.deg)
        idx_q, idx_c, sep, _ = search_around_sky(cq, cc, 0.3 * u.arcsec)
        is_qso = np.zeros(n_total, dtype=bool)
        is_qso[np.unique(idx_c)] = True

    is_ps    = df['is_point_source'].values.astype(bool)
    is_star  = is_ps & ~is_qso

    df['is_agn_qso'] = is_qso
    df['is_star_only'] = is_star

    # ── Per-category summary
    summary_lines = [f'Combo: {combo_name}  ({m1} → {m2})  Δt = {dt:.2f} yr',
                      f'N total: {n_total:,}',
                      f'N point_source: {is_ps.sum():,}',
                      f'N agn_qso     : {is_qso.sum():,}',
                      f'N stars_only  : {is_star.sum():,}',
                      '']

    if dt > 0:
        summary_lines.append(
            f'{"category":<12} {"N":>8} {"med pmra":>10} {"med pmdec":>10} '
            f'{"NMAD pmra":>10} {"NMAD pmdec":>11} {"med pmtot":>11} {"95% pmtot":>10}'
        )
        summary_lines.append('-' * 95)
        for cat_name, mask in [('all', np.ones(n_total, dtype=bool)),
                                ('stars',   is_star),
                                ('agn_qso', is_qso)]:
            n = int(mask.sum())
            if n == 0:
                continue
            mra = np.nanmedian(pmra[mask])
            mdc = np.nanmedian(pmdec[mask])
            nra = nmad(pmra[mask])
            ndc = nmad(pmdec[mask])
            mpt = np.nanmedian(pmtot[mask])
            p95 = np.nanpercentile(pmtot[mask], 95)
            summary_lines.append(
                f'{cat_name:<12} {n:>8,} {mra:>+10.2f} {mdc:>+10.2f} '
                f'{nra:>10.2f} {ndc:>11.2f} {mpt:>11.2f} {p95:>10.2f}'
            )
    else:
        summary_lines.append(
            f'{"category":<12} {"N":>8} {"med dra":>10} {"med ddec":>10} '
            f'{"NMAD dra":>10} {"NMAD ddec":>11}'
        )
        summary_lines.append('-' * 75)
        for cat_name, mask in [('all', np.ones(n_total, dtype=bool)),
                                ('stars',   is_star),
                                ('agn_qso', is_qso)]:
            n = int(mask.sum())
            if n == 0:
                continue
            mra = np.nanmedian(dra_mas[mask])
            mdc = np.nanmedian(ddec_mas[mask])
            nra = nmad(dra_mas[mask])
            ndc = nmad(ddec_mas[mask])
            summary_lines.append(
                f'{cat_name:<12} {n:>8,} {mra:>+10.2f} {mdc:>+10.2f} '
                f'{nra:>10.2f} {ndc:>11.2f}'
            )

    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    (HTML / f'pm_{combo_name}_summary.txt').write_text(summary_text + '\n')

    # ── 3-panel pmra-vs-pmdec scatter ─────────────────────────────────────
    if dt > 0:
        lim = max(50.0, float(np.nanpercentile(pmtot, 99)) * 1.1) if np.any(np.isfinite(pmtot)) else 50.0
        xlbl, ylbl = 'pmra cos δ  (mas/yr)', 'pmdec  (mas/yr)'
        xvals, yvals = pmra, pmdec
    else:
        lim = max(50.0, float(np.nanpercentile(np.hypot(dra_mas, ddec_mas), 99)) * 1.1)
        xlbl, ylbl = 'Δra cos δ  (mas)', 'Δdec  (mas)'
        xvals, yvals = dra_mas, ddec_mas

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    for ax, (cat, mask, col) in zip(axes, [
        ('all',     np.ones(n_total, dtype=bool), 'C7'),
        ('stars',   is_star, 'C0'),
        ('agn_qso', is_qso,  'C3')]):
        _scatter_panel(ax, xvals[mask], yvals[mask],
                       f'{cat}  (N={int(mask.sum()):,})', lim, color=col)
        ax.set_xlabel(xlbl)
    axes[0].set_ylabel(ylbl)
    fig.suptitle(f'{combo_name}  (Δt = {dt:.2f} yr,  {m1} → {m2})', fontsize=12)
    fig.tight_layout()
    fig.savefig(HTML / f'pm_{combo_name}_scatter.png', dpi=130)
    plt.close(fig)

    # ── Sky quiver (stars only) ────────────────────────────────────────────
    if dt > 0 and is_star.sum() > 0:
        ra_plot = ra1[is_star]
        dc_plot = dc1[is_star]
        u_arr = pmra[is_star]
        v_arr = pmdec[is_star]
        # Downsample if too many
        if len(ra_plot) > 5000:
            sel = np.random.choice(len(ra_plot), 5000, replace=False)
            ra_plot, dc_plot = ra_plot[sel], dc_plot[sel]
            u_arr, v_arr = u_arr[sel], v_arr[sel]
        fig, ax = plt.subplots(figsize=(8, 7))
        # Quiver scale: 100 mas/yr → 0.01 deg (~36"), reasonable on sky
        scale = 8000  # mas/yr per inch
        q = ax.quiver(ra_plot, dc_plot, u_arr, v_arr, np.hypot(u_arr, v_arr),
                      cmap='viridis', clim=(0, 50), scale=scale, width=0.0015,
                      alpha=0.7)
        ax.invert_xaxis()
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title(f'{combo_name} — stars-only PM (Δt={dt:.2f} yr, N={len(ra_plot):,})')
        plt.colorbar(q, ax=ax, label='|PM| (mas/yr)')
        ax.set_aspect(1.0/np.cos(np.deg2rad(np.median(dc_plot))))
        fig.tight_layout()
        fig.savefig(HTML / f'pm_{combo_name}_quiver.png', dpi=130)
        plt.close(fig)

    # ── Save augmented catalog ────────────────────────────────────────────
    out_parq = OUT / f'refined_{combo_name}_with_pm.parquet'
    df.to_parquet(out_parq, index=False)
    csv_written = False
    if len(df) * len(df.columns) < 200_000_000:
        df.to_csv(OUT / f'refined_{combo_name}_with_pm.csv', index=False)
        csv_written = True

    elapsed = time.time() - t0
    print(f'[{combo_name}] wrote {out_parq.name} in {elapsed:.1f}s')
    return {
        'combo': combo_name,
        'dt_yr': dt,
        'n_total': n_total,
        'n_point_source': int(is_ps.sum()),
        'n_agn_qso': int(is_qso.sum()),
        'n_stars_only': int(is_star.sum()),
        'elapsed_s': elapsed,
    }


def main():
    t0 = time.time()
    print('=' * 75)
    print('Step 7 — proper motion on 6 refined cross-matched catalogs')
    print('=' * 75)
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_combo, args): args[0] for args in COMBOS_PM}
        for fut in as_completed(futures):
            results.append(fut.result())

    print()
    print('=' * 75)
    print('Summary')
    print('=' * 75)
    lines = [f'{"combo":<28} {"Δt":>6} {"n_total":>9} {"n_ps":>8} {"n_agn":>6} {"n_star":>8} {"time":>6}']
    lines.append('-' * 80)
    for r in sorted(results, key=lambda x: -x['dt_yr']):
        lines.append(
            f'{r["combo"]:<28} {r["dt_yr"]:>5.2f}y {r["n_total"]:>9,} '
            f'{r["n_point_source"]:>8,} {r["n_agn_qso"]:>6,} '
            f'{r["n_stars_only"]:>8,} {r["elapsed_s"]:>5.1f}s'
        )
    text = '\n'.join(lines)
    print(text)
    (HTML / 'pm_summary_all.txt').write_text(text + '\n')
    print(f'\nWall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
