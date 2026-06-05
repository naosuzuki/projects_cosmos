#!/usr/bin/env python
"""
35_step2_galex_3way_subset.py — extract the 3-way HST ∩ JWST ∩ Euclid VIS
subset of the GALEX catalog landed by 34_.

Sister to 21_/23_/28_/31_.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

GX_DIR  = Path('/Volumes/exdisk1/data/catalog/GALEX/COSMOS')
GX_FULL = GX_DIR / 'galex_cosmos.parquet'
GX_3WAY = GX_DIR / 'galex_3way.parquet'
GX_3WAY_CSV  = GX_DIR / 'galex_3way.csv'
GX_3WAY_META = GX_DIR / 'galex_3way.meta.json'

WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/common_area_3way.wkt')


def main():
    t0 = time.time()
    print(f'Reading parent: {GX_FULL}')
    df = pd.read_parquet(GX_FULL)
    print(f'  {len(df):,} rows from the Euclid VIS polygon')

    poly = shp_wkt.loads(WKT_3WAY.read_text())
    n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
    print(f'3-way polygon: {n_pieces} pieces, area = {poly.area:.4f} deg²')

    pts = points(df['ra'].values, df['dec'].values)
    df3 = df[poly.contains(pts)].reset_index(drop=True)
    print(f'  inside 3-way: {len(df3):,} / {len(df):,}  '
          f'({100*len(df3)/len(df):.1f}%)')

    df3.to_parquet(GX_3WAY, index=False)
    df3.to_csv(GX_3WAY_CSV, index=False)
    print(f'[save] {GX_3WAY.name}  ({GX_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')

    meta = {
        'created_utc_iso':   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':      round(time.time()-t0, 2),
        'parent_path':       str(GX_FULL),
        'parent_rows':       int(len(df)),
        'rows_inside_3way':  int(len(df3)),
        'fraction_kept':     round(len(df3)/len(df), 4),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':    int(n_pieces),
        'columns':           list(df3.columns),
    }
    GX_3WAY_META.write_text(json.dumps(meta, indent=2))
    print(f'[save] {GX_3WAY_META.name}')

    print()
    print('=== GALEX — 3-way subset ===')
    print(f'  rows                  : {len(df3):>6,}')
    for b in ('nuv', 'fuv'):
        col = f'{b}_mag'
        if col in df3.columns:
            d = df3[col].dropna()
            n21 = (d < 21).sum(); n22 = (d < 22).sum()
            print(f'  {b.upper()}<21 / {b.upper()}<22         : {n21:>6,} / {n22:>6,}')


if __name__ == '__main__':
    main()
