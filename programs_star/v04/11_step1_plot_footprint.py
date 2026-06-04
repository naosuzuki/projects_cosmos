#!/usr/bin/env python
"""
11_step1_plot_footprint.py — RA/Dec map of the v04 footprint.

Uses the tile lookup table (RA/Dec polygon corners stored per tile) and
the persisted 3-way common-area WKT — NO FITS REOPEN, NO REPROJECTION.

Per-mission footprint = union of all per-tile 4-corner sky polygons:
    HST         = union over the 10 HST F814W tiles
    JWST        = INTERSECTION across F115/F150/F277/F444W (each filter
                  is the union of its 20 tile polygons first)
    Euclid VIS  = union over the 60 VIS MER tiles
    Euclid NISP = INTERSECTION across NIR-Y/J/H (each band is the
                  union of its 60 tile polygons first)

The 3-way common area itself is loaded from
    csvfiles_star/v04/common_area_3way.wkt
and shown as the filled grey region.

Output: htmls/v04/step1/common_area_footprint.png

Runs in seconds (vs the original ~30-50 min reprojection version).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt as shp_wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

V04           = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
LOOKUP        = V04 / 'tile_lookup.parquet'
THREE_WAY_WKT = V04 / 'common_area_3way.wkt'

OUT_DIR = Path('/Users/suzuki/github/projects_cosmos/htmls/v04/step1')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'common_area_footprint.png'

COLOURS = {'HST':  '#d62728',   # red
           'JWST': '#1f77b4',   # blue
           'VIS':  '#2ca02c',   # green
           'NISP': '#ff7f0e'}   # orange

# Drawing: VIS in the back as a thick solid line; NISP in front as a
# dashed line on top, so VIS green shows through the gaps in the dashes
# and both Euclid outlines are visible simultaneously.
DRAW_ORDER  = ['HST', 'JWST', 'VIS', 'NISP']
WIDTHS      = {'HST': 1.8, 'JWST': 1.8, 'VIS': 3.6, 'NISP': 1.8}
ALPHAS      = {'HST': 1.0, 'JWST': 1.0, 'VIS': 1.0, 'NISP': 1.0}
LINESTYLES  = {'HST': '-',  'JWST': '-',  'VIS': '-',  'NISP': '--'}


def _row_polygon(row) -> Polygon:
    """Build a Shapely polygon from a tile_lookup row's 4 corners (RA, Dec)."""
    ra  = np.asarray(row['corners_ra'])
    dec = np.asarray(row['corners_dec'])
    coords = list(zip(ra, dec))
    coords.append(coords[0])  # close the ring
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def mission_union(df: pd.DataFrame, mission: str, filt: str):
    """Union of all tile polygons for one (mission, filter)."""
    sub = df[(df['mission'] == mission) & (df['filter'] == filt)]
    polys = [_row_polygon(r) for _, r in sub.iterrows()]
    return unary_union(polys) if polys else Polygon()


def mission_intersection_across_filters(df: pd.DataFrame,
                                         mission: str,
                                         filters: list[str]):
    """For each filter, take the union over its tiles; then INTERSECT
    those per-filter unions to get the cross-band common footprint."""
    per_filter = [mission_union(df, mission, f) for f in filters]
    out = per_filter[0]
    for p in per_filter[1:]:
        out = out.intersection(p)
    return out


def draw_geom(ax, geom, **kw):
    """matplotlib helper: draw a (Multi)Polygon outline."""
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        polys = [geom]
    else:
        polys = list(geom.geoms)
    for poly in polys:
        xs, ys = zip(*poly.exterior.coords)
        ax.plot(xs, ys, **kw)
        for hole in poly.interiors:
            hxs, hys = zip(*hole.coords)
            ax.plot(hxs, hys, **kw)


def main():
    print('Reading tile_lookup.parquet ...')
    df = pd.read_parquet(LOOKUP)
    print(f'  {len(df)} tile rows')

    print('Building per-mission footprints via Shapely union/intersection ...')
    footprints = {
        'HST':  mission_union(df, 'HST', 'F814W'),
        'JWST': mission_intersection_across_filters(
                    df, 'JWST', ['F115W', 'F150W', 'F277W', 'F444W']),
        'VIS':  mission_union(df, 'Euclid', 'VIS'),
        'NISP': mission_intersection_across_filters(
                    df, 'Euclid', ['NIR-Y', 'NIR-J', 'NIR-H']),
    }
    for name, geom in footprints.items():
        print(f'  {name:5s} polygon area = {geom.area:.4f} deg², bounds={geom.bounds}')

    print(f'Loading 3-way common-area polygon from {THREE_WAY_WKT.name} ...')
    three_way = shp_wkt.loads(THREE_WAY_WKT.read_text())
    print(f'  3-way area = {three_way.area:.4f} deg²')

    # ── Plot ───────────────────────────────────────────────────────────
    # Axis extent: union bounding box of all footprints, with a 5% margin.
    all_geoms = list(footprints.values()) + [three_way]
    union_box = unary_union(all_geoms).bounds   # (minx, miny, maxx, maxy)
    margin_x = 0.05 * (union_box[2] - union_box[0])
    margin_y = 0.05 * (union_box[3] - union_box[1])

    fig, ax = plt.subplots(figsize=(10, 9))

    # 3-way: filled grey patches
    if three_way.geom_type == 'Polygon':
        threes = [three_way]
    else:
        threes = list(three_way.geoms)
    for p in threes:
        xs, ys = zip(*p.exterior.coords)
        ax.fill(xs, ys, color='#555', alpha=0.45, zorder=2)

    # Per-mission outlines in the documented order (NISP drawn last on top)
    z = {'HST': 3, 'JWST': 3, 'VIS': 4, 'NISP': 5}
    for name in DRAW_ORDER:
        draw_geom(ax, footprints[name],
                  color=COLOURS[name], lw=WIDTHS[name], alpha=ALPHAS[name],
                  linestyle=LINESTYLES[name], zorder=z[name])

    # Axes / labels
    ax.set_xlim(union_box[2] + margin_x, union_box[0] - margin_x)   # RA decreases L->R
    ax.set_ylim(union_box[1] - margin_y, union_box[3] + margin_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('RA (J2000, deg)')
    ax.set_ylabel('Dec (J2000, deg)')
    ax.grid(color='k', alpha=0.2, ls=':')

    # Legend (proxy artists so all four lines + the filled 3-way appear)
    handles = []
    for name in ('HST', 'JWST', 'VIS', 'NISP'):
        handles.append(plt.Line2D([], [], color=COLOURS[name],
                                    lw=WIDTHS[name], alpha=ALPHAS[name],
                                    linestyle=LINESTYLES[name],
                                    label=f'{name}  ({footprints[name].area:.3f} deg²)'))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='#555', ec='none', alpha=0.45,
                                  label=f'3-way HST∩JWST∩VIS  ({three_way.area:.3f} deg²)'))
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=10)

    ax.set_title(
        'COSMOS v04 — per-mission footprints + 3-way common area\n'
        f'(from tile_lookup polygons; full Euclid VIS extent shown)',
        fontsize=12)

    # Annotation on the discovery: NISP and VIS share the same MER tiles
    ax.text(0.02, 0.02,
            'VIS and NISP outlines coincide (same MER tile WCS per band) — '
            'VIS drawn as a thick solid line in the back, NISP drawn as a '
            'dashed line in front so both show through.',
            transform=ax.transAxes, fontsize=8, color='#555',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, bbox_inches='tight')
    print(f'\n[save] {OUT_PNG}  ({OUT_PNG.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
