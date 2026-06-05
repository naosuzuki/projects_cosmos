#!/usr/bin/env python
"""
31_step2_lsdr10_3way_subset.py — extract the 3-way HST ∩ JWST ∩ Euclid VIS
subset of the LS DR10 Tractor catalog landed by 30_step2_lsdr10_pull.py.

Sister to 21_/23_/28_ for Gaia DR3, SDSS DR17, PS1 DR2.

Inputs:
  /Volumes/exdisk1/data/catalog/DESI_Legacy/COSMOS/lsdr10_cosmos.parquet
  csvfiles_star/v04/common_area_3way.wkt
Outputs (alongside the parent):
  lsdr10_3way.parquet
  lsdr10_3way.csv
  lsdr10_3way.meta.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

LS_DIR  = Path('/Volumes/exdisk1/data/catalog/DESI_Legacy/COSMOS')
LS_FULL = LS_DIR / 'lsdr10_cosmos.parquet'
LS_3WAY = LS_DIR / 'lsdr10_3way.parquet'
LS_3WAY_CSV  = LS_DIR / 'lsdr10_3way.csv'
LS_3WAY_META = LS_DIR / 'lsdr10_3way.meta.json'

WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                'common_area_3way.wkt')


def main():
    t0 = time.time()

    print(f'Reading parent: {LS_FULL}')
    df = pd.read_parquet(LS_FULL)
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

    df3.to_parquet(LS_3WAY, index=False)
    df3.to_csv(LS_3WAY_CSV, index=False)
    print(f'[save] {LS_3WAY.name}  '
          f'({LS_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')
    print(f'[save] {LS_3WAY_CSV.name}      '
          f'({LS_3WAY_CSV.stat().st_size/1024/1024:.1f} MB)')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':    round(time.time() - t0, 2),
        'parent_path':     str(LS_FULL),
        'parent_rows':     int(len(df)),
        'wkt_3way_path':   str(WKT_3WAY),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':  int(n_pieces),
        'rows_inside_3way': int(len(df3)),
        'fraction_kept':   round(len(df3) / len(df), 4),
        'columns':         list(df3.columns),
    }
    LS_3WAY_META.write_text(json.dumps(meta, indent=2))
    print(f'[save] {LS_3WAY_META.name}')

    print()
    print('=== LS DR10 Tractor catalog — 3-way HST ∩ JWST ∩ Euclid VIS subset ===')
    print(f'  rows                : {len(df3):>6,}')
    if 'type' in df3.columns:
        for ty in ('PSF', 'REX', 'EXP', 'DEV', 'SER'):
            n = (df3['type'] == ty).sum()
            print(f'  type = {ty:<4s}         : {n:>6,}')
    # Flux → AB mag
    for b in ('g', 'r', 'i', 'z'):
        fcol = f'flux_{b}'
        if fcol in df3.columns:
            flux = df3[fcol].values
            ok = flux > 0
            mag = np.full(len(df3), np.nan)
            mag[ok] = 22.5 - 2.5*np.log10(flux[ok])
            n22 = ((mag < 22) & np.isfinite(mag)).sum()
            n24 = ((mag < 24) & np.isfinite(mag)).sum()
            print(f'  {b}<22 / {b}<24         : {n22:>6,} / {n24:>6,}')
    if 'ref_cat' in df3.columns:
        n_gaia = df3['ref_cat'].isin(['G2', 'GE']).sum()
        print(f'  Gaia-anchored        : {n_gaia:>6,}')


if __name__ == '__main__':
    main()
