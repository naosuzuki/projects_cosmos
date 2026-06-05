#!/usr/bin/env python
"""
21_step2_gaia_dr3_3way_subset.py — extract the 3-way subset of the
Gaia DR3 catalog landed by 20_step2_gaia_dr3_pull.py.

The 20_ pull is over the Euclid VIS polygon (3.91 deg^2, 15,328 sources).
For most downstream v04 work the relevant area is the smaller 3-way
HST ∩ JWST ∩ Euclid_VIS common-area footprint (0.5303 deg^2) — that's
where we have all three missions' photometry.  This script filters
the Gaia table down to that polygon and saves it alongside the
parent.

Inputs:
  /Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.parquet
  csvfiles_star/v04/common_area_3way.wkt

Outputs (alongside the parent):
  gaia_dr3_3way.parquet           canonical
  gaia_dr3_3way.csv               human-readable sidecar
  gaia_dr3_3way.meta.json         input + cut + row counts + summary

The 3-way polygon ships as a MultiPolygon (20 disconnected pieces
reflecting per-HST-tile structure); point-in-polygon is via shapely.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
from shapely import wkt as shp_wkt
from shapely import points

GAIA_DIR    = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS')
GAIA_FULL   = GAIA_DIR / 'gaia_dr3_cosmos.parquet'
GAIA_3WAY   = GAIA_DIR / 'gaia_dr3_3way.parquet'
GAIA_3WAY_CSV  = GAIA_DIR / 'gaia_dr3_3way.csv'
GAIA_3WAY_META = GAIA_DIR / 'gaia_dr3_3way.meta.json'

WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                'common_area_3way.wkt')


def main():
    t0 = time.time()

    print(f'Reading parent: {GAIA_FULL}')
    df = pd.read_parquet(GAIA_FULL)
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

    # ── Save ───────────────────────────────────────────────────────────
    df3.to_parquet(GAIA_3WAY, index=False)
    df3.to_csv(GAIA_3WAY_CSV, index=False)
    print(f'[save] {GAIA_3WAY.name}  '
          f'({GAIA_3WAY.stat().st_size/1024/1024:.1f} MB, {len(df3):,} rows)')
    print(f'[save] {GAIA_3WAY_CSV.name}      '
          f'({GAIA_3WAY_CSV.stat().st_size/1024/1024:.1f} MB)')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':    round(time.time() - t0, 2),
        'parent_path':     str(GAIA_FULL),
        'parent_rows':     int(len(df)),
        'wkt_3way_path':   str(WKT_3WAY),
        'polygon_area_deg2': round(poly.area, 4),
        'polygon_pieces':  int(n_pieces),
        'rows_inside_3way': int(len(df3)),
        'fraction_kept':   round(len(df3) / len(df), 4),
        'columns':         list(df3.columns),
    }
    GAIA_3WAY_META.write_text(json.dumps(meta, indent=2))
    print(f'[save] {GAIA_3WAY_META.name}')

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print('=== Gaia DR3 — 3-way HST ∩ JWST ∩ Euclid VIS subset ===')
    print(f'  rows                : {len(df3):>6,}')
    print(f'  with parallax > 0   : {(df3["parallax"] > 0).sum():>6,}')
    print(f'  with PM measured    : {df3[["pmra","pmdec"]].notna().all(axis=1).sum():>6,}')
    print(f'  G < 18 (bright)     : {(df3["phot_g_mean_mag"] < 18).sum():>6,}')
    print(f'  G < 14 (very bright): {(df3["phot_g_mean_mag"] < 14).sum():>6,}')
    print(f'  RUWE < 1.4 (clean)  : {(df3["ruwe"] < 1.4).sum():>6,}')
    print(f'  in_qso_candidates   : {df3["in_qso_candidates"].sum():>6,}')
    print(f'  in_galaxy_candidates: {df3["in_galaxy_candidates"].sum():>6,}')
    print(f'  non_single_star  >0 : {(df3["non_single_star"] > 0).sum():>6,}')


if __name__ == '__main__':
    main()
