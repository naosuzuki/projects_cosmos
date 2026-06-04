#!/usr/bin/env python
"""
13_step1_plot_3way_matrix.py — 3×3 matrix of per-mission and pairwise
common areas for the v04 3-way (HST, JWST, Euclid VIS).

Reads the per-mission masks cached on the global 1″ COSMOS WCS by
12_step1_3way_common_area.py — no reprojection needed, so this runs
in seconds rather than the 20+ min the from-scratch reprojection takes.

Output:
  htmls/v04/step1/common_area_3way_matrix.png

The matrix:
                HST            JWST           VIS
  HST   [ HST only      ] [ HST ∩ JWST     ] [ HST ∩ VIS      ]
  JWST  [ HST ∩ JWST    ] [ JWST only      ] [ JWST ∩ VIS     ]
  VIS   [ HST ∩ VIS     ] [ JWST ∩ VIS     ] [ VIS only       ]

The diagonal shows each mission's eroded coverage union (JWST cell
shows the F115∩F150∩F277∩F444W intersection).  Off-diagonals show the
pairwise overlap.  Each cell title carries the area in deg².

A 10th annotation in the top-right corner gives the full 3-way area
(HST ∩ JWST ∩ VIS).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

V04   = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
GCOV  = V04 / 'coverage_global'
OUT_DIR = Path('/Users/suzuki/github/projects_cosmos/htmls/v04/step1')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'common_area_3way_matrix.png'

# Cached per-mission masks (already on the global 1″ WCS)
SOURCES = {
    'HST':  GCOV / 'HST_F814W_1arcsec.fits.gz',
    'JWST': GCOV / 'JWST_4band_intersect_1arcsec.fits.gz',
    'VIS':  GCOV / 'Euclid_VIS_1arcsec.fits.gz',
}
COLOURS = {'HST': '#d62728', 'JWST': '#1f77b4', 'VIS': '#2ca02c'}


def load_mask(p: Path) -> tuple[np.ndarray, WCS]:
    with fits.open(p) as h:
        return np.asarray(h[0].data, dtype=bool), WCS(h[0].header)


def main():
    print('Loading per-mission masks (cached on global 1″ WCS) ...')
    masks, wcs = {}, None
    for name, p in SOURCES.items():
        m, w = load_mask(p)
        masks[name] = m
        wcs = wcs or w
        print(f'  {name:5s}  {m.sum()/1e6:6.2f} Mpx  ≈ {m.sum()/3600**2:.4f} deg²')

    # Crop to the union bbox so we don't waste plot real estate on empty grid
    union = masks['HST'] | masks['JWST'] | masks['VIS']
    rows = np.any(union, axis=1).nonzero()[0]
    cols = np.any(union, axis=0).nonzero()[0]
    pad = 30
    y0, y1 = max(0, rows.min()-pad), min(union.shape[0], rows.max()+pad)
    x0, x1 = max(0, cols.min()-pad), min(union.shape[1], cols.max()+pad)
    print(f'  bbox y[{y0}:{y1}] x[{x0}:{x1}] ({y1-y0}×{x1-x0} px)')

    # Slice each mask to the bbox + a sliced WCS for axis ticks
    masks_s = {k: m[y0:y1, x0:x1] for k, m in masks.items()}
    wcs_s = wcs.slice((slice(y0, y1), slice(x0, x1)))

    names = ['HST', 'JWST', 'VIS']
    # ── Build the figure: 3×3 grid of subplots with WCS axes ───────────
    fig = plt.figure(figsize=(13, 13))
    for ir, row in enumerate(names):
        for ic, col in enumerate(names):
            ax = fig.add_subplot(3, 3, ir*3 + ic + 1, projection=wcs_s)
            if ir == ic:
                # Diagonal: single-mission footprint
                m = masks_s[row]
                title = f'{row} only'
                fill_colour = COLOURS[row]
            else:
                # Off-diagonal: pairwise intersection
                m = masks_s[row] & masks_s[col]
                title = f'{row} ∩ {col}'
                # Use the "smaller" mission's colour for the fill
                # (visually emphasises the bottleneck on each pair)
                fill_colour = COLOURS[row if masks_s[row].sum()
                                       <= masks_s[col].sum() else col]

            area_deg2 = float(m.sum()) / 3600**2
            area_amin2 = area_deg2 * 3600  # arcmin²

            ax.imshow(m.astype(int), origin='lower',
                      cmap=plt.matplotlib.colors.ListedColormap(['#f5f5f5', fill_colour]),
                      vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'{title}\n{area_deg2:.4f} deg² ({area_amin2:.1f} arcmin²)',
                         fontsize=10)
            # Only put RA/Dec labels on the outer rim
            if ir < 2:
                ax.coords[0].set_ticklabel_visible(False)
                ax.set_xlabel('')
            else:
                ax.set_xlabel('RA (J2000)', fontsize=9)
            if ic > 0:
                ax.coords[1].set_ticklabel_visible(False)
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Dec (J2000)', fontsize=9)
            ax.coords[0].set_major_formatter('d.dd')
            ax.coords[1].set_major_formatter('d.dd')
            ax.grid(color='k', alpha=0.15, ls=':')

    # ── 3-way intersection summary (overall title + annotation) ────────
    three_way = masks_s['HST'] & masks_s['JWST'] & masks_s['VIS']
    a3 = three_way.sum() / 3600**2
    fig.suptitle(f'v04 — 3-way HST ∩ JWST ∩ Euclid VIS common-area matrix\n'
                 f'full 3-way intersection = {a3:.4f} deg² '
                 f'({a3*3600:.1f} arcmin²)',
                 fontsize=14, y=1.00)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'\n[save] {OUT_PNG}  ({OUT_PNG.stat().st_size/1024:.0f} KB)')

    # Print a compact tabular summary too
    print('\n=== 3×3 area matrix (deg²) ===')
    header = '          ' + ''.join(f'{n:>10s}' for n in names)
    print(header)
    for r in names:
        row_str = f'{r:8s}  '
        for c in names:
            m = masks[r] & masks[c] if r != c else masks[r]
            row_str += f'{m.sum()/3600**2:10.4f}'
        print(row_str)


if __name__ == '__main__':
    main()
