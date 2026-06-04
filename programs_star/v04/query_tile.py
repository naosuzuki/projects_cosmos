#!/usr/bin/env python
"""
query_tile.py — convenience API over the Step 1 artefacts.

Downstream v04 code asks two kinds of questions about per-source
membership in the v04 footprint:

  1. "Given (RA, Dec), which mosaic tiles contain it (per mission)?"
     -> tiles_containing(ra, dec, mission=None, filter=None)
     Backed by csvfiles_star/v04/tile_lookup.parquet
     (4-corner polygon test, after a fast bbox prefilter).

  2. "Given (RA, Dec), is it inside the 3-way HST ∩ JWST ∩ Euclid VIS
     footprint?"
     -> in_v04_footprint(ra, dec)
     Backed by csvfiles_star/v04/common_area_3way.wkt
     (vectorised shapely point-in-polygon test).

Both lookups are cached at module import so that calling them
millions of times in inner loops is cheap, and crucially neither
ever reopens a per-tile FITS file.

Usage:
    from programs_star.v04.query_tile import (
        tiles_containing, mask_path_for, in_v04_footprint, footprint_polygon)

    # which tiles cover the COSMOS centre in JWST?
    df = tiles_containing(150.1, 2.2, mission='JWST')
    print(df[['filter', 'tile_id']])

    # vectorised footprint test on a source catalog
    import pandas as pd
    cat = pd.read_parquet('my_sources.parquet')
    cat['in_footprint'] = in_v04_footprint(cat['ra'].values, cat['dec'].values)

    # find the per-tile mask FITS for a specific (mission, filter, tile)
    path = mask_path_for('HST', 'F814W', 'B5')
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Lazy imports of shapely/matplotlib so just importing this module is cheap.

V04 = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
LOOKUP_PARQUET = V04 / 'tile_lookup.parquet'
FOOTPRINT_WKT  = V04 / 'common_area_3way.wkt'


# ──────────────────────────────────────────────────────────────────────
# Tile lookup
# ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _lookup() -> pd.DataFrame:
    """Module-cached read of the per-tile lookup parquet."""
    df = pd.read_parquet(LOOKUP_PARQUET)
    # corners_ra / corners_dec are stored as lists; keep as object arrays.
    return df


def _bbox_filter(df: pd.DataFrame, ra: float, dec: float) -> pd.DataFrame:
    return df[(df.ra_min <= ra) & (df.ra_max >= ra) &
              (df.dec_min <= dec) & (df.dec_max >= dec)]


def _point_in_polygon(ra: float, dec: float,
                      corners_ra: Iterable[float],
                      corners_dec: Iterable[float]) -> bool:
    """Ray-cast point-in-polygon for one (ra, dec) and one closed ring."""
    xs = np.asarray(corners_ra,  dtype=float)
    ys = np.asarray(corners_dec, dtype=float)
    n  = len(xs)
    inside = False
    j = n - 1
    for i in range(n):
        if ((ys[i] > dec) != (ys[j] > dec)) and \
           (ra < (xs[j] - xs[i]) * (dec - ys[i]) / (ys[j] - ys[i] + 1e-30) + xs[i]):
            inside = not inside
        j = i
    return inside


def tiles_containing(ra: float, dec: float,
                      mission: str | None = None,
                      filter: str  | None = None,
                      polygon: bool = True) -> pd.DataFrame:
    """Return rows of the tile lookup table whose tile contains (ra, dec).

    Parameters
    ----------
    ra, dec : float
        ICRS J2000 in degrees.
    mission : optional str
        Restrict to 'HST', 'JWST', or 'Euclid'.
    filter : optional str
        Restrict to a specific filter (e.g. 'F115W', 'VIS').
    polygon : bool
        If True (default), tighten the bbox match to the actual 4-corner
        sky polygon.  Otherwise the bbox match is returned as-is
        (cheaper, but ~0.1% over-inclusive at tile rotations).
    """
    df = _lookup()
    if mission is not None:
        df = df[df.mission == mission]
    if filter is not None:
        df = df[df.filter == filter]

    hits = _bbox_filter(df, ra, dec)
    if polygon and len(hits) > 0:
        inside = hits.apply(
            lambda r: _point_in_polygon(ra, dec,
                                         r['corners_ra'], r['corners_dec']),
            axis=1).values
        hits = hits[inside]
    return hits.reset_index(drop=True)


def mask_path_for(mission: str, filter: str, tile_id: str) -> str:
    """Return the absolute path of the per-tile coverage mask FITS."""
    df = _lookup()
    sub = df[(df.mission == mission) & (df.filter == filter) &
             (df.tile_id == tile_id)]
    if len(sub) == 0:
        raise KeyError(f'no tile lookup row for ({mission}, {filter}, {tile_id})')
    if len(sub) > 1:
        raise ValueError(f'ambiguous tile lookup ({mission}, {filter}, {tile_id})')
    return str(sub.iloc[0]['coverage_mask_path'])


# ──────────────────────────────────────────────────────────────────────
# 3-way footprint queries
# ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def footprint_polygon():
    """Lazy-load the 3-way HST∩JWST∩EuclidVIS polygon (Shapely)."""
    try:
        from shapely import wkt
    except ImportError as e:
        raise ImportError('shapely required for footprint queries '
                          '(pip install shapely)') from e
    if not FOOTPRINT_WKT.exists():
        raise FileNotFoundError(
            f'{FOOTPRINT_WKT} not built yet — run '
            f'programs_star/v04/12_step1_3way_common_area.py first')
    return wkt.loads(FOOTPRINT_WKT.read_text())


def in_v04_footprint(ra, dec) -> np.ndarray | bool:
    """Vectorised "is (ra, dec) inside the 3-way HST∩JWST∩EuclidVIS area?".

    Accepts either scalar floats or arrays of equal length; returns
    a scalar bool or a boolean array correspondingly.
    """
    from shapely import points
    poly = footprint_polygon()
    if np.isscalar(ra):
        from shapely.geometry import Point
        return poly.contains(Point(float(ra), float(dec)))
    ra  = np.asarray(ra,  dtype=float)
    dec = np.asarray(dec, dtype=float)
    pts = points(ra, dec)
    return poly.contains(pts)


# ──────────────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('--- Tile lookup smoke test ---')
    print(f'lookup table: {LOOKUP_PARQUET}  '
          f'{LOOKUP_PARQUET.stat().st_size/1024:.0f} KB')
    df = tiles_containing(150.1, 2.2)
    print(f'tiles_containing(150.1, 2.2): {len(df)} hits')
    print(df[['mission', 'filter', 'tile_id']].to_string(index=False))

    print('\n--- Footprint smoke test ---')
    if FOOTPRINT_WKT.exists():
        poly = footprint_polygon()
        n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
        print(f'footprint WKT: {FOOTPRINT_WKT}  '
              f'{FOOTPRINT_WKT.stat().st_size/1024:.0f} KB, '
              f'{n_pieces} polygon(s), area = {poly.area:.4f} deg²')
        # Pick a guaranteed-inside point (centroid of the largest sub-polygon)
        if n_pieces > 1:
            largest = max(poly.geoms, key=lambda p: p.area)
        else:
            largest = poly
        cen = largest.centroid
        sample = np.array([(150.10,  2.20),       # nominal COSMOS centre
                           (cen.x,   cen.y),       # centroid of largest piece
                           (149.50,  2.20),       # outside (west)
                           (150.50,  2.85)])      # outside (north)
        ras, decs = sample.T
        ins = in_v04_footprint(ras, decs)
        labels = ['cosmos centre', 'largest-piece centroid',
                  'outside (west)', 'outside (north)']
        for (r, d), b, lbl in zip(sample, ins, labels):
            print(f'  in_v04_footprint({r:8.4f}, {d:6.4f}) -> {bool(b):5}  ({lbl})')
    else:
        print(f'(footprint WKT not built yet at {FOOTPRINT_WKT})')
