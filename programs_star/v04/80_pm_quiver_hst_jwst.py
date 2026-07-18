#!/usr/bin/env python
"""
80_pm_quiver_hst_jwst.py — single-pair proper-motion quiver plot (styled).

Reproduces the v01 styled quiver from the corrected v02 pair catalogs with
Suzuki's matplotlib conventions (Times, inward ticks on all four sides) and
enlarged axis fonts.  Default pair is ACS->JWST; pass a combo key for any pair:

  python 80_pm_quiver_hst_jwst.py                    # ACS -> JWST
  python 80_pm_quiver_hst_jwst.py JWST_Euclid_VIS    # JWST -> Euclid
  python 80_pm_quiver_hst_jwst.py HST_Euclid_VIS     # ACS  -> Euclid

  data : csvfiles_star/v02/refined_<combo>_with_pm_v02.parquet
  out  : htmls/pm_v04/pm_<combo>_quiver_{all,stars,agn_qso}.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman'],
                     'mathtext.fontset': 'stix',
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 1.3,
                     'xtick.labelsize': 20, 'ytick.labelsize': 20,
                     'xtick.major.size': 9, 'ytick.major.size': 9,
                     'xtick.minor.size': 4.5, 'ytick.minor.size': 4.5,
                     'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
                     'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9})

LBL, TTL, CBLBL, CBTICK, NOTE = 30, 24, 26, 18, 17

ROOT = Path('/Users/suzuki/github/projects_cosmos')
DATA = ROOT / 'csvfiles_star' / 'v02'
OUT  = ROOT / 'htmls' / 'pm_v04'
N_SHOW = 1500
CATEGORIES = ['all', 'stars', 'agn_qso']

# combo -> (m1, m2, ref_tag, fallback_tag, baseline, vmax, scale)
COMBOS = {
    'HST_JWST':         ('ACS',  'JWST',   'jwst', 'hst',  '19 yr',   15,  300),
    'HST_Euclid_VIS':   ('ACS',  'Euclid', 'vis',  'hst',  '19.5 yr', 15,  300),
    'HST_Euclid_NISP':  ('ACS',  'Euclid', 'nisp', 'hst',  '19.5 yr', 15,  300),
    'JWST_Euclid_VIS':  ('JWST', 'Euclid', 'vis',  'jwst', '0.5 yr', 120, 2000),
    'JWST_Euclid_NISP': ('JWST', 'Euclid', 'nisp', 'jwst', '0.5 yr', 120, 2000),
}


def category_mask(df: pd.DataFrame, cat: str) -> pd.Series:
    if cat == 'all':
        return pd.Series(True, index=df.index)
    if cat == 'stars':
        return df['is_star_only'].fillna(False).astype(bool)
    if cat == 'agn_qso':
        return df['is_agn_qso'].fillna(False).astype(bool)
    raise ValueError(cat)


def ref_radec(df: pd.DataFrame, ref_tag: str, fb_tag: str):
    """Reference-mission cat positions, falling back where they are missing."""
    ra, dec = df[f'cat_ra_{ref_tag}'].values.copy(), df[f'cat_dec_{ref_tag}'].values.copy()
    miss = ~np.isfinite(ra)
    ra[miss], dec[miss] = df[f'cat_ra_{fb_tag}'].values[miss], df[f'cat_dec_{fb_tag}'].values[miss]
    return ra, dec


def quiver_one(combo: str, cat: str, vmax_override=None) -> None:
    m1, m2, ref_tag, fb_tag, baseline, vmax, scale = COMBOS[combo]
    if vmax_override is not None:               # force a specific |mu| colour range
        vmax = vmax_override
    df = pd.read_parquet(DATA / f'refined_{combo}_with_pm_v02.parquet')
    sel = category_mask(df, cat) & df['pmra'].notna() & df['pmdec'].notna()
    sub = df[sel].copy()
    if len(sub) == 0:
        print(f'  [{cat}] no rows — skip'); return

    ra, dec = ref_radec(sub, ref_tag, fb_tag)
    pmra, pmdec, pmtot = sub['pmra'].values, sub['pmdec'].values, sub['pmtot'].values
    is_cc = sub['pm_method'].values == 'cat_cat'
    cc, gs = np.where(is_cc)[0], np.where(~is_cc)[0]
    rng = np.random.default_rng(0)
    if len(cc) > N_SHOW:
        cc = rng.choice(cc, N_SHOW, replace=False)
    n_show = len(cc) + len(gs)

    fig, ax = plt.subplots(figsize=(11, 9.2))
    ax.axhline(2.2, color='0.88', lw=0.8, zorder=0)       # COSMOS survey strip
    head = dict(headwidth=4.2, headlength=5.2, headaxislength=4.6)

    q = None
    if len(cc):
        q = ax.quiver(ra[cc], dec[cc], pmra[cc], pmdec[cc], pmtot[cc],
                      cmap='coolwarm', clim=(0, vmax), scale=scale,
                      width=0.0028, alpha=0.9, zorder=2, **head)
    if len(gs):                                           # dotted, display-capped
        cap = 3 * vmax
        p = np.hypot(pmra[gs], pmdec[gs])
        s = np.where(p > cap, cap / p, 1.0)
        q2 = ax.quiver(ra[gs], dec[gs], pmra[gs] * s, pmdec[gs] * s, pmtot[gs],
                       cmap='coolwarm', clim=(0, vmax), scale=scale,
                       width=0.002, alpha=0.75, linestyle=':', linewidths=0.7,
                       zorder=3, **head)
        q = q or q2

    cb = plt.colorbar(q, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r'$|\mu|$  (mas yr$^{-1}$)', fontsize=CBLBL, labelpad=10)
    cb.ax.tick_params(labelsize=CBTICK, width=1.1, size=6)
    cb.outline.set_linewidth(1.1)

    ax.set_xlabel('R.A.  [deg]', fontsize=LBL, labelpad=10)
    ax.set_ylabel('Decl.  [deg]', fontsize=LBL, labelpad=10)
    ax.invert_xaxis()
    ax.tick_params(pad=7)

    tag = {'all': '', 'stars': '  (stars)', 'agn_qso': '  (AGN / QSO)'}.get(cat, '')
    ax.set_title(f'{m1} $\\rightarrow$ {m2} proper motion{tag}', fontsize=TTL, pad=14)
    note = (f'{baseline} baseline\nN = {int(sel.sum()):,} stars\n{n_show:,} shown'
            + (f'\nGaia-supp {len(gs):,}' if len(gs) else ''))
    ax.text(0.025, 0.975, note, transform=ax.transAxes, va='top', ha='left',
            fontsize=NOTE, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='0.75',
                      lw=1.0, alpha=0.92))

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'pm_{combo}_quiver_{cat}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  wrote {out.name}  (cat_cat {len(cc):,}, Gaia-supp {len(gs):,})')


def main(combo: str = 'HST_JWST', vmax_override=None) -> None:
    m1, m2, *_ = COMBOS[combo]
    print(f'{m1}->{m2} PM quiver from refined_{combo}_with_pm_v02.parquet'
          + (f'  (|mu| scale 0-{vmax_override:g})' if vmax_override else ''))
    for cat in CATEGORIES:
        quiver_one(combo, cat, vmax_override)
    print('Done.')


if __name__ == '__main__':
    _combo = sys.argv[1] if len(sys.argv) > 1 else 'HST_JWST'
    _vmax = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(_combo, _vmax)
