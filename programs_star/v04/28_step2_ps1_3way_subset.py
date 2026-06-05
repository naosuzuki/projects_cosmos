#!/usr/bin/env python
"""
28_step2_ps1_3way_subset.py — extract the 3-way HST ∩ JWST ∩ Euclid VIS
subset of the PS1 DR2 catalog landed by 27_step2_ps1_pull*.py.

Sister to 21_ (Gaia DR3 3-way) and 23_ (SDSS DR17 3-way).

Inputs:
  /Volumes/exdisk1/data/PanSTARRS/COSMOS/ps1_cosmos.parquet
  csvfiles_star/v04/common_area_3way.wkt
Outputs (alongside the parent):
  ps1_3way.parquet
  ps1_3way.csv
  ps1_3way.meta.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

PS1_DIR  = Path('/Volumes/exdisk1/data/PanSTARRS/COSMOS')
PS1_FULL = PS1_DIR / 'ps1_cosmos.parquet'
PS1_3WAY = PS1_DIR / 'ps1_3way.parquet'
PS1_3WAY_CSV  = PS1_DIR / 'ps1_3way.csv'
PS1_3WAY_META = PS1_DIR / 'ps1_3way.meta.json'

WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                'common_area_3way.wkt')


def main():
    t0 = time.time()

    print(f'Reading parent: {PS1_FULL}')
    df = pd.read_parquet(PS1_FULL)
    print(f'  {len(df):,} rows from the Euclid VIS polygon')

    print(f'Loading 3-way polygon: {WKT_3WAY}')
    poly = shp_wkt.loads(WKT_3WAY.read_text())
    n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
    print(f'  3-way MultiPolygon: {n_pieces} pieces, area = {poly.area:.4f} deg²')

    print('Point-in-polygon filter (vectorised) ...')
    # PS1 column names are raMean / decMean (not ra / dec)
    pts = points(df['raMean'].values, df['decMean'].values)
    inside = poly.contains(pts)
    df3 = df[inside].reset_index(drop=True)
    print(f'  inside 3-way: {len(df3):,} / {len(df):,}  '
          f'({100*len(df3)/len(df):.1f}%)')

    df3.to_parquet(PS1_3WAY, index=False)
    df3.to_csv(PS1_3WAY_CSV, index=False)
    print(f'[save] {PS1_3WAY.name}  '
          f'({PS1_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')
    print(f'[save] {PS1_3WAY_CSV.name}      '
          f'({PS1_3WAY_CSV.stat().st_size/1024/1024:.1f} MB)')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':    round(time.time() - t0, 2),
        'parent_path':     str(PS1_FULL),
        'parent_rows':     int(len(df)),
        'wkt_3way_path':   str(WKT_3WAY),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':  int(n_pieces),
        'rows_inside_3way': int(len(df3)),
        'fraction_kept':   round(len(df3) / len(df), 4),
        'columns':         list(df3.columns),
    }
    PS1_3WAY_META.write_text(json.dumps(meta, indent=2))
    print(f'[save] {PS1_3WAY_META.name}')

    print()
    print('=== PS1 DR2 MeanObject — 3-way HST ∩ JWST ∩ Euclid VIS subset ===')
    print(f'  rows                : {len(df3):>8,}')
    for b in ('g', 'r', 'i', 'z', 'y'):
        col = f'{b}MeanPSFMag'
        if col in df3.columns:
            n22 = ((df3[col] > 0) & (df3[col] < 22)).sum()
            n20 = ((df3[col] > 0) & (df3[col] < 20)).sum()
            print(f'  {b}<22 / {b}<20         : {n22:>8,} / {n20:>8,}')
    if 'nDetections' in df3.columns:
        print(f'  nDetections >= 5    : {(df3["nDetections"] >= 5).sum():>8,}')


if __name__ == '__main__':
    main()
