#!/usr/bin/env python
"""
23_step2_sdss_3way_subset.py — extract the 3-way HST ∩ JWST ∩ Euclid VIS
subset of the SDSS DR17 catalog landed by 22_step2_sdss_pull.py.

Sister to 21_step2_gaia_dr3_3way_subset.py.

Inputs:
  /Volumes/exdisk1/data/SDSS/COSMOS/sdss_cosmos.parquet
  csvfiles_star/v04/common_area_3way.wkt
Outputs (alongside the parent):
  sdss_3way.parquet
  sdss_3way.csv
  sdss_3way.meta.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

SDSS_DIR  = Path('/Volumes/exdisk1/data/SDSS/COSMOS')
SDSS_FULL = SDSS_DIR / 'sdss_cosmos.parquet'
SDSS_3WAY = SDSS_DIR / 'sdss_3way.parquet'
SDSS_3WAY_CSV  = SDSS_DIR / 'sdss_3way.csv'
SDSS_3WAY_META = SDSS_DIR / 'sdss_3way.meta.json'

WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                'common_area_3way.wkt')


def main():
    t0 = time.time()

    print(f'Reading parent: {SDSS_FULL}')
    df = pd.read_parquet(SDSS_FULL)
    print(f'  {len(df):,} rows from the Euclid VIS polygon')

    print(f'Loading 3-way polygon: {WKT_3WAY}')
    poly = shp_wkt.loads(WKT_3WAY.read_text())
    n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
    print(f'  3-way MultiPolygon: {n_pieces} pieces, area = {poly.area:.4f} deg²')

    print('Point-in-polygon filter (vectorised) ...')
    pts = points(df['ra'].values, df['dec'].values)
    inside = poly.contains(pts)
    df3 = df[inside].reset_index(drop=True)
    print(f'  inside 3-way: {len(df3):,} / {len(df):,}  '
          f'({100*len(df3)/len(df):.1f}%)')

    df3.to_parquet(SDSS_3WAY, index=False)
    df3.to_csv(SDSS_3WAY_CSV, index=False)
    print(f'[save] {SDSS_3WAY.name}  '
          f'({SDSS_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')
    print(f'[save] {SDSS_3WAY_CSV.name}      '
          f'({SDSS_3WAY_CSV.stat().st_size/1024/1024:.1f} MB)')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':    round(time.time() - t0, 2),
        'parent_path':     str(SDSS_FULL),
        'parent_rows':     int(len(df)),
        'wkt_3way_path':   str(WKT_3WAY),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':  int(n_pieces),
        'rows_inside_3way': int(len(df3)),
        'fraction_kept':   round(len(df3) / len(df), 4),
        'columns':         list(df3.columns),
    }
    SDSS_3WAY_META.write_text(json.dumps(meta, indent=2))
    print(f'[save] {SDSS_3WAY_META.name}')

    print()
    print('=== SDSS DR17 — 3-way HST ∩ JWST ∩ Euclid VIS subset ===')
    print(f'  rows                : {len(df3):>6,}')
    n_star = (df3['type'] == 6).sum()
    n_gal  = (df3['type'] == 3).sum()
    print(f'  classified star (6) : {n_star:>6,}')
    print(f'  classified gal  (3) : {n_gal:>6,}')
    print(f'  clean photometry    : {(df3["clean"] == 1).sum():>6,}')
    print(f'  r < 22 (well-detect): {(df3["psfMag_r"] < 22).sum():>6,}')
    print(f'  r < 20 (bright)     : {(df3["psfMag_r"] < 20).sum():>6,}')
    print(f'  probPSF > 0.5       : {(df3["probPSF"] > 0.5).sum():>6,}')


if __name__ == '__main__':
    main()
