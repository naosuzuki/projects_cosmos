#!/usr/bin/env python
"""
22_step2_sdss_pull.py — Step 2.2: pull SDSS DR17 PhotoPrimary over the
v04 Euclid VIS polygon.

Mirrors the Gaia pull (20_step2_gaia_dr3_pull.py) in structure:
  1. SQL bbox query at SDSS SkyServer covering the full Euclid VIS extent
  2. Shapely point-in-polygon filter to the exact Euclid VIS footprint

SDSS contributes ground-based u/g/r/i/z PSF and model magnitudes for
~10^4 sources over COSMOS — independent zero-point chain
(Sloan 2.5 m + SDSS imaging camera) used in v04 Step 3 as one of the
three independent optical surveys (S4: SDSS / PS1 / DESI Legacy are
kept independent because cross-comparing their photometry is the
science deliverable).

Spatial selection: same bbox as the Gaia pull
  bbox  RA  148.85 -> 151.30   (~2.45 deg wide)
        Dec   1.10 ->   3.40   (~2.30 deg tall)
  area  ~5.6 deg^2 ADQL, then ~3.91 deg^2 after Euclid VIS polygon filter

Outputs (to /Volumes/exdisk1/data/SDSS/COSMOS/):
  sdss_cosmos.parquet         canonical (PhotoPrimary in VIS polygon)
  sdss_cosmos.csv             human-readable sidecar
  sdss_cosmos.query.sql       exact SDSS SkyServer SQL used
  sdss_cosmos.meta.json       timestamp + row counts + bbox + columns

Columns retrieved (28 of PhotoPrimary's ~470):
  identifiers     : objID, mode, type, clean, probPSF
  astrometry      : ra, dec, raErr, decErr
  PSF photometry  : psfMag_{u,g,r,i,z} (+Err)
  model photometry: modelMag_{u,g,r,i,z} (+Err)
  observation     : mjd, run, rerun, camcol, field
  quality         : flags
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/SDSS/COSMOS')

# SDSS SkyServer SQL.  PhotoPrimary is the deduplicated primary detections.
SQL = f"""
SELECT
  objID, mode, type, clean, probPSF,
  ra, dec, raErr, decErr,
  psfMag_u,    psfMag_g,    psfMag_r,    psfMag_i,    psfMag_z,
  psfMagErr_u, psfMagErr_g, psfMagErr_r, psfMagErr_i, psfMagErr_z,
  modelMag_u,    modelMag_g,    modelMag_r,    modelMag_i,    modelMag_z,
  modelMagErr_u, modelMagErr_g, modelMagErr_r, modelMagErr_i, modelMagErr_z,
  mjd, run, rerun, camcol, field, flags
FROM PhotoPrimary
WHERE ra  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND dec BETWEEN {DEC_MIN} AND {DEC_MAX}
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Print the SQL and exit; do not contact SkyServer.')
    p.add_argument('--data-release', type=int, default=17,
                   help='SDSS data release as an int (default 17 = DR17).')
    return p.parse_args()


def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'sdss_cosmos'

    print(f'SDSS DR{args.data_release} query:')
    print('-' * 60)
    print(SQL)
    print('-' * 60)
    print()

    (base.with_suffix('.query.sql')).write_text(SQL + '\n')

    if args.dry_run:
        print('[dry-run] not contacting SkyServer.  SQL saved to',
              base.with_suffix('.query.sql'))
        return

    from astroquery.sdss import SDSS
    print(f'Submitting query to SkyServer DR{args.data_release} ...', flush=True)
    t0 = time.time()
    try:
        tbl = SDSS.query_sql(SQL, data_release=args.data_release,
                              timeout=300)
    except Exception as e:
        print(f'SkyServer query FAILED: {e!r}', file=sys.stderr)
        sys.exit(1)
    dt = time.time() - t0
    if tbl is None or len(tbl) == 0:
        print('Empty result.  Check schema / DR availability.', file=sys.stderr)
        sys.exit(2)
    print(f'Returned {len(tbl):,} rows in {dt:.1f}s.')

    import pandas as pd
    df = tbl.to_pandas()

    # ── Polygon filter to exact Euclid VIS footprint ───────────────────
    n_bbox = len(df)
    print(f'\nFiltering {n_bbox:,} bbox rows through the exact Euclid VIS polygon ...')
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
        p = Polygon(ring)
        if not p.is_valid:
            p = p.buffer(0)
        vis_polys.append(p)
    vis_polygon = unary_union(vis_polys)
    print(f'  Euclid VIS polygon area : {vis_polygon.area:.4f} deg²')
    pts = points(df['ra'].values, df['dec'].values)
    inside = vis_polygon.contains(pts)
    df_in = df[inside].reset_index(drop=True)
    print(f'  inside VIS polygon : {len(df_in):,} / {n_bbox:,}  '
          f'({100*len(df_in)/n_bbox:.1f}%)')
    df = df_in

    parquet_path = base.with_suffix('.parquet')
    csv_path     = base.with_suffix('.csv')
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f'[save] {parquet_path.name}  '
          f'({parquet_path.stat().st_size/1024/1024:.1f} MB, {len(df):,} rows)')
    print(f'[save] {csv_path.name}      '
          f'({csv_path.stat().st_size/1024/1024:.1f} MB)')

    meta = {
        'query_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':  round(dt, 2),
        'row_count_bbox_query':     int(n_bbox),
        'row_count_in_vis_polygon': int(len(df)),
        'bbox_deg': {
            'ra_min':  RA_MIN,  'ra_max':  RA_MAX,
            'dec_min': DEC_MIN, 'dec_max': DEC_MAX,
            'area_approx_deg2': round((RA_MAX-RA_MIN)*(DEC_MAX-DEC_MIN), 2),
        },
        'vis_polygon_deg2':       round(vis_polygon.area, 4),
        'spatial_selection':      'SQL bbox at Euclid VIS extent, then '
                                  'Shapely point-in-polygon filter to the '
                                  'exact Euclid VIS footprint.',
        'sdss_table':    f'DR{args.data_release} PhotoPrimary',
        'columns':       list(df.columns),
        'parquet_path':  str(parquet_path),
        'sql_path':      str(base.with_suffix('.query.sql')),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))
    print(f'[save] sdss_cosmos.meta.json')

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print('=== SDSS DR17 PhotoPrimary in the Euclid VIS polygon ===')
    print(f'  rows                : {len(df):>8,}')
    n_star  = (df['type'] == 6).sum()
    n_gal   = (df['type'] == 3).sum()
    print(f'  classified star (6) : {n_star:>8,}')
    print(f'  classified gal  (3) : {n_gal:>8,}')
    print(f'  clean photometry    : {(df["clean"] == 1).sum():>8,}')
    print(f'  r < 22 (well-detect): {(df["psfMag_r"] < 22).sum():>8,}')
    print(f'  r < 20 (bright)     : {(df["psfMag_r"] < 20).sum():>8,}')
    print(f'  probPSF > 0.5       : {(df["probPSF"] > 0.5).sum():>8,}')


if __name__ == '__main__':
    main()
