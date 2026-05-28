#!/usr/bin/env python
"""
58_pm_quiver_styled.py — regenerate Step 7 quiver plots in the style of
projects_euclid/programs/20_pm_three_pairs_quiver.py (the ACS→Euclid
plot shipped in the paper, papers/26_jwsteuclidhst_note/figures/
pm_quiver_acs_to_euclid.png).

Style:
  - cmap='coolwarm', clim=(0, vmax)  with vmax = 15 (long baseline) or 120 (short)
  - scale=300 (long) or 2000 (short),  width=0.0025, alpha=0.85
  - axhline(2.2, color="0.85", lw=0.4)   ← survey strip indicator
  - colorbar label "|PM| (mas/yr)"
  - x-axis = RA, inverted; y-axis = Dec; reference frame = the m2 mission
  - title format: "PM X→Y  (NN yr baseline, N=…; shown …)"
  - DOWN-SAMPLE to N=1500 like the original
  - GAIA SUPPLEMENTS plotted with linestyle=':' (dotted)

Inputs:  csvfiles_star/refined_<combo>_with_pm_v02.parquet
Outputs: htmls/pm_v01/pm_<combo>_quiver_<category>.png    (15 files,
                                                           overwritten)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'pm_v01'

# (combo, m1_label, m2_label, tag1, tag2, baseline_yr_str, vmax, scale)
COMBOS = [
    ('HST_Euclid_VIS',   'ACS',  'Euclid',  'hst',  'vis',   '19.5 yr',   15,  300),
    ('HST_Euclid_NISP',  'ACS',  'Euclid',  'hst',  'nisp',  '19.5 yr',   15,  300),
    ('HST_JWST',         'ACS',  'JWST',    'hst',  'jwst',  '19 yr',     15,  300),
    ('JWST_Euclid_VIS',  'JWST', 'Euclid',  'jwst', 'vis',   '0.5 yr',   120, 2000),
    ('JWST_Euclid_NISP', 'JWST', 'Euclid',  'jwst', 'nisp',  '0.5 yr',   120, 2000),
]
CATEGORIES = ['all', 'stars', 'agn_qso']
N_SHOW = 1500


def select_category(df: pd.DataFrame, cat: str) -> pd.Series:
    if cat == 'all':
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if cat == 'stars':
        return df['is_star_only'].fillna(False).astype(bool)
    if cat == 'agn_qso':
        return df['is_agn_qso'].fillna(False).astype(bool)
    raise ValueError(cat)


def get_ref_radec(df: pd.DataFrame, tag2: str, tag1: str) -> tuple[np.ndarray, np.ndarray]:
    """Use m2 (reference) cat positions when available, fall back to m1."""
    ra  = df[f'cat_ra_{tag2}'].values.copy()
    dec = df[f'cat_dec_{tag2}'].values.copy()
    miss = ~np.isfinite(ra)
    if f'cat_ra_{tag1}' in df.columns:
        ra[miss]  = df[f'cat_ra_{tag1}'].values[miss]
        dec[miss] = df[f'cat_dec_{tag1}'].values[miss]
    return ra, dec


def quiver_one(combo: str, m1: str, m2: str, tag1: str, tag2: str,
                baseline: str, vmax: float, scale: float) -> None:
    df = pd.read_parquet(OUT / f'refined_{combo}_with_pm_v02.parquet')
    # need finite PM
    finite = df['pmra'].notna() & df['pmdec'].notna()
    for cat in CATEGORIES:
        sel = select_category(df, cat) & finite
        sub = df[sel].copy()
        if len(sub) == 0:
            continue

        ra, dec = get_ref_radec(sub, tag2, tag1)
        pmra  = sub['pmra'].values
        pmdec = sub['pmdec'].values
        pmtot = sub['pmtot'].values
        is_cc = (sub['pm_method'].values == 'cat_cat')
        is_gs = ~is_cc

        # downsample cat_cat — preserve all gaia supplements (always shown)
        rng = np.random.default_rng(0)
        cc_idx = np.where(is_cc)[0]
        gs_idx = np.where(is_gs)[0]
        if len(cc_idx) > N_SHOW:
            cc_idx = rng.choice(cc_idx, N_SHOW, replace=False)
        n_show_total = len(cc_idx) + len(gs_idx)

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.axhline(2.2, color="0.85", lw=0.4, zorder=0)

        # cat_cat arrows: solid
        if len(cc_idx):
            q = ax.quiver(ra[cc_idx], dec[cc_idx],
                          pmra[cc_idx], pmdec[cc_idx], pmtot[cc_idx],
                          cmap='coolwarm', clim=(0, vmax),
                          scale=scale, width=0.0025, alpha=0.85,
                          zorder=2)
            cb = plt.colorbar(q, ax=ax, label="|PM| (mas/yr)")

        # gaia supplements: dotted, thinner, no extra edgecolor.  Cap PM
        # length for display only so very-high-PM bright stars don't
        # dominate; colour saturates at vmax as usual.
        if len(gs_idx):
            disp_cap = 3 * vmax     # 45 mas/yr for long baseline, 360 for short
            ptot = np.hypot(pmra[gs_idx], pmdec[gs_idx])
            scl = np.where(ptot > disp_cap, disp_cap / ptot, 1.0)
            u_disp = pmra[gs_idx] * scl
            v_disp = pmdec[gs_idx] * scl
            q2 = ax.quiver(ra[gs_idx], dec[gs_idx],
                           u_disp, v_disp, pmtot[gs_idx],
                           cmap='coolwarm', clim=(0, vmax),
                           scale=scale, width=0.0018, alpha=0.7,
                           linestyle=':', linewidths=0.6, zorder=3)
            if not len(cc_idx):
                cb = plt.colorbar(q2, ax=ax, label="|PM| (mas/yr)")

        ax.set_xlabel(f"RA (deg, {m2})")
        ax.set_ylabel(f"Dec (deg, {m2})")
        ax.invert_xaxis()
        n_total = int(sel.sum())
        title_tag = f' [{cat}]' if cat != 'all' else ''
        title = (f"PM {m1}→{m2}{title_tag}  ({baseline} baseline, "
                 f"N={n_total:,}; shown {n_show_total:,}"
                 + (f", gaia-supp {len(gs_idx):,}" if len(gs_idx) else "")
                 + ")")
        ax.set_title(title)

        fig.tight_layout()
        out = HTML / f'pm_{combo}_quiver_{cat}.png'
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'wrote {out.name}  (cat_cat: {len(cc_idx):,}, gaia-supp: {len(gs_idx):,})')


def main():
    HTML.mkdir(parents=True, exist_ok=True)
    print('Regenerating styled quivers (coolwarm, dec=2.2 line, dotted gaia)...')
    for combo, m1, m2, tag1, tag2, baseline, vmax, scale in COMBOS:
        print(f'\n── {combo}  ({m1}→{m2}, {baseline}, vmax={vmax}, scale={scale}) ──')
        quiver_one(combo, m1, m2, tag1, tag2, baseline, vmax, scale)
    print('\nDone.')


if __name__ == '__main__':
    main()
