#!/usr/bin/env python
"""
81_pm_quiver_three_panels.py — three-panel proper-motion quiver figure.

Same style as 80_pm_quiver_hst_jwst.py, laid out as a 1x3 paper figure:
  ACS -> JWST   (19 yr)  |  ACS -> Euclid (19.5 yr)  |  JWST -> Euclid (0.5 yr)
The two long-baseline panels share the clean 0-15 mas/yr scale; the short
JWST->Euclid panel is noise-dominated (0-120 mas/yr) — the point of the
comparison.  Per-panel colorbars because the scales differ.

  data : csvfiles_star/v02/refined_<combo>_with_pm_v02.parquet
  out  : htmls/pm_v04/pm_three_panels_<category>.png
"""
from __future__ import annotations
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
                     'xtick.labelsize': 16, 'ytick.labelsize': 16,
                     'xtick.major.size': 8, 'ytick.major.size': 8,
                     'xtick.minor.size': 4, 'ytick.minor.size': 4,
                     'xtick.major.width': 1.2, 'ytick.major.width': 1.2})

ROOT = Path('/Users/suzuki/github/projects_cosmos')
DATA = ROOT / 'csvfiles_star' / 'v02'
OUT  = ROOT / 'htmls' / 'pm_v04'
N_SHOW = 1500
LBL, TTL, CBLBL, CBTICK, NOTE = 25, 23, 19, 15, 15

# combo, m1, m2, ref_tag, fallback_tag, baseline, vmax, scale
COMBOS = [
    ('HST_JWST',        'ACS',  'JWST',   'jwst', 'hst',  '19 yr',   15,  300),
    ('HST_Euclid_VIS',  'ACS',  'Euclid', 'vis',  'hst',  '19.5 yr', 15,  300),
    ('JWST_Euclid_VIS', 'JWST', 'Euclid', 'vis',  'jwst', '0.5 yr',  120, 2000),
]


def category_mask(df, cat):
    if cat == 'all':
        return pd.Series(True, index=df.index)
    if cat == 'stars':
        return df['is_star_only'].fillna(False).astype(bool)
    if cat == 'agn_qso':
        return df['is_agn_qso'].fillna(False).astype(bool)
    raise ValueError(cat)


def ref_radec(df, ref_tag, fb_tag):
    ra, dec = df[f'cat_ra_{ref_tag}'].values.copy(), df[f'cat_dec_{ref_tag}'].values.copy()
    miss = ~np.isfinite(ra)
    ra[miss], dec[miss] = df[f'cat_ra_{fb_tag}'].values[miss], df[f'cat_dec_{fb_tag}'].values[miss]
    return ra, dec


def panel(ax, combo, m1, m2, ref_tag, fb_tag, baseline, vmax, scale, cat, first):
    df = pd.read_parquet(DATA / f'refined_{combo}_with_pm_v02.parquet')
    sel = category_mask(df, cat) & df['pmra'].notna() & df['pmdec'].notna()
    sub = df[sel].copy()
    ra, dec = ref_radec(sub, ref_tag, fb_tag)
    pmra, pmdec, pmtot = sub['pmra'].values, sub['pmdec'].values, sub['pmtot'].values
    is_cc = sub['pm_method'].values == 'cat_cat'
    cc, gs = np.where(is_cc)[0], np.where(~is_cc)[0]
    rng = np.random.default_rng(0)
    if len(cc) > N_SHOW:
        cc = rng.choice(cc, N_SHOW, replace=False)

    ax.axhline(2.2, color='0.88', lw=0.8, zorder=0)
    head = dict(headwidth=4.2, headlength=5.2, headaxislength=4.6)
    q = None
    if len(cc):
        q = ax.quiver(ra[cc], dec[cc], pmra[cc], pmdec[cc], pmtot[cc],
                      cmap='coolwarm', clim=(0, vmax), scale=scale,
                      width=0.0032, alpha=0.9, zorder=2, **head)
    if len(gs):
        cap = 3 * vmax
        p = np.hypot(pmra[gs], pmdec[gs])
        s = np.where(p > cap, cap / p, 1.0)
        q2 = ax.quiver(ra[gs], dec[gs], pmra[gs] * s, pmdec[gs] * s, pmtot[gs],
                       cmap='coolwarm', clim=(0, vmax), scale=scale,
                       width=0.0024, alpha=0.75, linestyle=':', linewidths=0.7,
                       zorder=3, **head)
        q = q or q2
    cb = ax.figure.colorbar(q, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r'$|\mu|$  (mas yr$^{-1}$)', fontsize=CBLBL, labelpad=8)
    cb.ax.tick_params(labelsize=CBTICK, width=1.1, size=5)
    cb.outline.set_linewidth(1.1)

    ax.set_xlabel('R.A.  [deg]', fontsize=LBL, labelpad=8)
    if first:
        ax.set_ylabel('Decl.  [deg]', fontsize=LBL, labelpad=8)
    ax.invert_xaxis()
    ax.set_title(f'{m1} $\\rightarrow$ {m2}', fontsize=TTL, pad=12)
    med = np.median(pmtot)
    note = f'{baseline} baseline\nN = {int(sel.sum()):,}\nmed |$\\mu$| = {med:.1f}'
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va='top', ha='left',
            fontsize=NOTE, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.75',
                      lw=1.0, alpha=0.92))
    print(f'  {combo}: N={int(sel.sum()):,}  med|mu|={med:.1f}  '
          f'(cat_cat {len(cc):,}, gaia {len(gs):,})')


def main(cat='all'):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(26, 8.4), constrained_layout=True)
    print(f'three-panel PM quiver [{cat}]:')
    for ax, (combo, m1, m2, rt, fb, bl, vmax, scale) in zip(axes, COMBOS):
        panel(ax, combo, m1, m2, rt, fb, bl, vmax, scale, cat, ax is axes[0])
    out = OUT / f'pm_three_panels_{cat}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
