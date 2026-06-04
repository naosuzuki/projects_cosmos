#!/usr/bin/env python
"""
11_step1_plot_footprint.py — RA/Dec map of the 4-way footprint.

Reads the per-mission coverage masks reprojected onto the global 1"
COSMOS grid (built by Step 1) and shows the 4-way HST ∩ JWST ∩ VIS ∩
NISP common-area footprint, overlaid with per-mission outlines so it
is visible which mission limits each edge of the common region.

Output: htmls/v04/step1/common_area_footprint.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

V04   = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
COV   = V04 / 'coverage'
GLOBAL_MASK = V04 / 'common_area_4way_1arcsec.fits.gz'
OUT_DIR = Path('/Users/suzuki/github/projects_cosmos/htmls/v04/step1')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'common_area_footprint.png'


def load_global_mask():
    """Load the 4-way intersection mask + its WCS."""
    with fits.open(GLOBAL_MASK) as h:
        mask = np.asarray(h[0].data, dtype=bool)
        wcs  = WCS(h[0].header)
    return mask, wcs


def per_mission_union(prefix: str, wcs_target, shape_target) -> np.ndarray:
    """OR together every tile mask whose filename starts with `prefix`
    after reprojecting onto the target WCS."""
    out = np.zeros(shape_target, dtype=bool)
    for f in sorted(COV.glob(f'{prefix}*.fits.gz')):
        with fits.open(f, memmap=True) as h:
            m = np.asarray(h[0].data, dtype=np.float32)
            w = WCS(h[0].header)
        rp, _ = reproject_interp((m, w), wcs_target,
                                 shape_out=shape_target,
                                 order='nearest-neighbor')
        out |= (np.nan_to_num(rp, nan=0) > 0.5)
    return out


def jwst_intersection(wcs_target, shape_target):
    """All 4 JWST bands AND'd together."""
    out = None
    for flt in ('F115W', 'F150W', 'F277W', 'F444W'):
        m = per_mission_union(f'JWST_{flt}_', wcs_target, shape_target)
        out = m if out is None else (out & m)
    return out


def nisp_intersection(wcs_target, shape_target):
    """All 3 NISP bands AND'd together."""
    out = None
    for flt in ('NIR-Y', 'NIR-J', 'NIR-H'):
        m = per_mission_union(f'Euclid_{flt}_', wcs_target, shape_target)
        out = m if out is None else (out & m)
    return out


def main():
    print('Loading global 4-way mask ...')
    four_way, wcs = load_global_mask()
    shape = four_way.shape
    print(f'  grid {shape[1]}x{shape[0]}  4-way pixels: {four_way.sum():,}')

    print('Building per-mission union/intersection masks for outline overlay ...')
    hst  = per_mission_union('HST_F814W_',   wcs, shape)
    jwst = jwst_intersection(wcs, shape)
    vis  = per_mission_union('Euclid_VIS_',  wcs, shape)
    nisp = nisp_intersection(wcs, shape)
    print(f'  HST   {hst.sum()/1e6:6.2f} Mpx')
    print(f'  JWST  {jwst.sum()/1e6:6.2f} Mpx')
    print(f'  VIS   {vis.sum()/1e6:6.2f} Mpx')
    print(f'  NISP  {nisp.sum()/1e6:6.2f} Mpx')
    print(f'  4-way {four_way.sum()/1e6:6.2f} Mpx -> {four_way.sum()/3600**2:.4f} deg^2')

    # ── Plot ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 9))
    ax  = fig.add_subplot(111, projection=wcs)

    # Filled fan-of-grey: 4-way region in solid colour.
    ax.imshow(four_way, origin='lower', cmap='Greys', alpha=0.55,
              interpolation='nearest', vmin=0, vmax=1.4)

    # Per-mission outlines (drawn in pixel coords; WCSAxes handles RA/Dec ticks)
    colours = {'HST':  '#d62728',  # red
               'JWST': '#1f77b4',  # blue
               'VIS':  '#2ca02c',  # green
               'NISP': '#ff7f0e'}  # orange
    for name, m in (('HST', hst), ('JWST', jwst), ('VIS', vis), ('NISP', nisp)):
        ax.contour(m, levels=[0.5], colors=colours[name],
                   linewidths=1.4, linestyles='-')

    # Legend (proxy artists since contour doesn't take label well)
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=k)
               for k, c in colours.items()]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='#555', ec='none', alpha=0.55,
                                 label='4-way intersection'))
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=10)

    ax.set_xlabel('RA  (ICRS, J2000)')
    ax.set_ylabel('Dec (ICRS, J2000)')
    area_deg2 = four_way.sum() / 3600**2
    ax.set_title(f'COSMOS v04 — 4-way HST $\\cap$ JWST $\\cap$ Euclid VIS $\\cap$ '
                 f'Euclid NISP footprint\n'
                 f'common area: {area_deg2:.4f} deg$^2$  '
                 f'(global mask sampled at 1″)')
    ax.grid(color='k', alpha=0.2, ls=':')

    # Annotate where the bottleneck is (NISP-Y has tiny coverage)
    ax.text(0.02, 0.02,
            'NISP outline shows where the 4-way overlap is geometrically limited',
            transform=ax.transAxes, fontsize=8, color='#555',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    print(f'\n[save] {OUT_PNG}  ({OUT_PNG.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
