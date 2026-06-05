#!/usr/bin/env python
"""
37_step2_unwise_3way_subset.py — 3-way HST ∩ JWST ∩ Euclid VIS subset
of the unWISE DR1 catalog landed by 36_.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

UW_DIR  = Path('/Volumes/exdisk1/data/catalog/unWISE/COSMOS')
UW_FULL = UW_DIR / 'unwise_cosmos.parquet'
UW_3WAY = UW_DIR / 'unwise_3way.parquet'
UW_3WAY_CSV  = UW_DIR / 'unwise_3way.csv'
UW_3WAY_META = UW_DIR / 'unwise_3way.meta.json'
WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/common_area_3way.wkt')


def main():
    t0 = time.time()
    print(f'Reading parent: {UW_FULL}')
    df = pd.read_parquet(UW_FULL)
    print(f'  {len(df):,} rows from the Euclid VIS polygon')

    poly = shp_wkt.loads(WKT_3WAY.read_text())
    n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
    print(f'3-way polygon: {n_pieces} pieces, area = {poly.area:.4f} deg²')

    pts = points(df['ra'].values, df['dec'].values)
    df3 = df[poly.contains(pts)].reset_index(drop=True)
    print(f'  inside 3-way: {len(df3):,} / {len(df):,}  '
          f'({100*len(df3)/len(df):.1f}%)')

    df3.to_parquet(UW_3WAY, index=False)
    df3.to_csv(UW_3WAY_CSV, index=False)
    print(f'[save] {UW_3WAY.name}  ({UW_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')

    meta = {
        'created_utc_iso':   time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':      round(time.time()-t0, 2),
        'parent_path':       str(UW_FULL),
        'parent_rows':       int(len(df)),
        'rows_inside_3way':  int(len(df3)),
        'fraction_kept':     round(len(df3)/len(df), 4),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':    int(n_pieces),
        'columns':           list(df3.columns),
    }
    UW_3WAY_META.write_text(json.dumps(meta, indent=2))

    print()
    print('=== unWISE — 3-way subset ===')
    print(f'  rows                : {len(df3):>6,}')
    print(f'  primary == 1        : {(df3["primary"]==1).sum():>6,}')
    print(f'  mag_w1_vg < 19      : {((df3["mag_w1_vg"]<19)&df3["mag_w1_vg"].notna()).sum():>6,}')
    print(f'  5σ W1 (flux/dflux>5): {((df3["flux_w1"]/df3["dflux_w1"])>5).sum():>6,}')


if __name__ == '__main__':
    main()
