#!/usr/bin/env python
"""
24_step2_sdss_spec_pull.py — Step 2.2b: pull SDSS DR17 SpecObj
(spectroscopy) over the Euclid VIS polygon + 3-way subset.

Companion to 22_step2_sdss_pull.py (PhotoPrimary photometry).
SDSS spectra add a redshift (`z`), classification (`class` =
STAR/GALAXY/QSO), spectral subclass, and per-pixel-noise summary
(`snMedian`).  Used in v04 Steps 4-5 as a truth label for
spectroscopically-confirmed stars / galaxies / QSOs.

Spatial selection: same two-stage as Gaia / PhotoPrimary
  1. SQL bbox at SkyServer  RA [148.85, 151.30], Dec [1.10, 3.40]
  2. Shapely point-in-polygon filter → Euclid VIS footprint
  3. Then a further point-in-polygon filter → 3-way HST∩JWST∩VIS

Outputs (to /Volumes/exdisk1/data/catalog/SDSS/COSMOS/):
  sdss_spec_cosmos.parquet         in VIS polygon
  sdss_spec_cosmos.csv
  sdss_spec_cosmos.query.sql
  sdss_spec_cosmos.meta.json
  sdss_spec_3way.parquet           in 3-way polygon
  sdss_spec_3way.csv
  sdss_spec_3way.meta.json

Columns retrieved from SpecObj:
  identifiers   : specObjID, bestObjID
  position      : ra, dec
  classification: class, subclass, z, zErr, zWarning
  quality       : snMedian
  observation   : plate, mjd, fiberID, programname, survey
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/catalog/SDSS/COSMOS')

SQL = f"""
SELECT
  specObjID, bestObjID,
  ra, dec,
  class, subclass, z, zErr, zWarning,
  snMedian,
  plate, mjd, fiberID, programname, survey
FROM SpecObj
WHERE ra  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND dec BETWEEN {DEC_MIN} AND {DEC_MAX}
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--data-release', type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base   = OUT_DIR / 'sdss_spec_cosmos'
    base3w = OUT_DIR / 'sdss_spec_3way'

    print(f'SDSS DR{args.data_release} SpecObj query:')
    print('-' * 60)
    print(SQL)
    print('-' * 60)
    print()

    (base.with_suffix('.query.sql')).write_text(SQL + '\n')

    if args.dry_run:
        print('[dry-run] not contacting SkyServer.  SQL saved.')
        return

    from astroquery.sdss import SDSS
    print(f'Submitting query to SkyServer DR{args.data_release} ...', flush=True)
    t0 = time.time()
    try:
        tbl = SDSS.query_sql(SQL, data_release=args.data_release, timeout=300)
    except Exception as e:
        print(f'SkyServer query FAILED: {e!r}', file=sys.stderr)
        sys.exit(1)
    dt = time.time() - t0
    if tbl is None or len(tbl) == 0:
        print('Empty result.')
        sys.exit(2)
    print(f'Returned {len(tbl):,} spectra in {dt:.1f}s.')

    import pandas as pd
    df = tbl.to_pandas()
    n_bbox = len(df)

    # ── Polygon filter: Euclid VIS ─────────────────────────────────────
    print(f'\nFiltering through Euclid VIS polygon ...')
    from shapely import points
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    LOOKUP = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/tile_lookup.parquet')
    tdf = pd.read_parquet(LOOKUP)
    vis = tdf[(tdf['mission'] == 'Euclid') & (tdf['filter'] == 'VIS')]
    vis_polys = []
    for _, r in vis.iterrows():
        ring = list(zip(r['corners_ra'], r['corners_dec']))
        ring.append(ring[0])
        p = Polygon(ring); p = p if p.is_valid else p.buffer(0)
        vis_polys.append(p)
    vis_polygon = unary_union(vis_polys)
    pts = points(df['ra'].values, df['dec'].values)
    df_vis = df[vis_polygon.contains(pts)].reset_index(drop=True)
    print(f'  in VIS  : {len(df_vis):,}/{n_bbox:,}  '
          f'({100*len(df_vis)/n_bbox:.1f}%)')

    df_vis.to_parquet(base.with_suffix('.parquet'), index=False)
    df_vis.to_csv(base.with_suffix('.csv'), index=False)
    print(f'[save] {base.name}.parquet  ({len(df_vis):,} rows)')

    # ── Polygon filter: 3-way ──────────────────────────────────────────
    from shapely import wkt as shp_wkt
    WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                    'common_area_3way.wkt')
    poly3 = shp_wkt.loads(WKT_3WAY.read_text())
    pts3 = points(df_vis['ra'].values, df_vis['dec'].values)
    df_3w = df_vis[poly3.contains(pts3)].reset_index(drop=True)
    print(f'  in 3-way: {len(df_3w):,}/{len(df_vis):,}  '
          f'({100*len(df_3w)/max(len(df_vis),1):.1f}%)')

    df_3w.to_parquet(base3w.with_suffix('.parquet'), index=False)
    df_3w.to_csv(base3w.with_suffix('.csv'), index=False)
    print(f'[save] {base3w.name}.parquet  ({len(df_3w):,} rows)')

    # ── Meta files ─────────────────────────────────────────────────────
    meta_vis = {
        'query_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds': round(dt, 2),
        'row_count_bbox_query':     int(n_bbox),
        'row_count_in_vis_polygon': int(len(df_vis)),
        'bbox_deg': {'ra_min': RA_MIN, 'ra_max': RA_MAX,
                     'dec_min': DEC_MIN, 'dec_max': DEC_MAX},
        'vis_polygon_deg2': round(vis_polygon.area, 4),
        'sdss_table':       f'DR{args.data_release} SpecObj',
        'columns':          list(df_vis.columns),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta_vis, indent=2))

    meta_3w = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'parent_path':     str(base.with_suffix('.parquet')),
        'parent_rows':     int(len(df_vis)),
        'wkt_3way_path':   str(WKT_3WAY),
        'polygon_area_deg2': round(poly3.area, 4),
        'rows_inside_3way': int(len(df_3w)),
        'columns':          list(df_3w.columns),
    }
    (base3w.with_suffix('.meta.json')).write_text(json.dumps(meta_3w, indent=2))

    # ── Summary ────────────────────────────────────────────────────────
    def breakdown(d, label):
        if len(d) == 0:
            print(f'\n=== {label}: empty ===')
            return
        print(f'\n=== {label} ===')
        print(f'  rows                : {len(d):>6,}')
        for cls in ('STAR', 'GALAXY', 'QSO'):
            n = (d['class'] == cls).sum()
            print(f'  class = {cls:<6s}     : {n:>6,}')
        print(f'  zWarning = 0 (good z): {(d["zWarning"] == 0).sum():>6,}')
        print(f'  snMedian > 5         : {(d["snMedian"] > 5).sum():>6,}')

    breakdown(df_vis, 'SDSS DR17 SpecObj in Euclid VIS polygon')
    breakdown(df_3w,  'SDSS DR17 SpecObj in 3-way subset')


if __name__ == '__main__':
    main()
