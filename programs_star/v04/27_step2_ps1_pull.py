#!/usr/bin/env python
"""
27_step2_ps1_pull.py — Step 2.3: pull Pan-STARRS 1 DR2 MeanObject
over the v04 Euclid VIS polygon.

Sister of 20_/22_ for Gaia DR3 and SDSS DR17.

DR2 vs DR1:
  - DR2 astrometry is tied to Gaia DR2 (~10 mas) → matches JWST /
    HST_2005 frame; DR1 is 2MASS/USNO-B (20–50 mas).
  - DR2 ships per-detector zero-points, the ForcedMeanObject table
    for faint sources, and the per-epoch Detection table.

For v04 we pull the MeanObject view (deduplicated mean photometry across
all stack epochs).  Columns are g/r/i/z/y PSF and Kron magnitudes plus
position, per-band detection counts, and quality flags.

Spatial selection:
  1. Cone search at MAST Catalogs (radius covering the Euclid VIS bbox
     diagonal — bbox = [148.85, 151.30] RA × [1.10, 3.40] Dec, diagonal
     ≈ 1.7 deg).  MAST PS1 API doesn't accept box queries, only cones.
  2. Shapely point-in-polygon filter → exact Euclid VIS footprint
     (3.91 deg^2).

Outputs (to /Volumes/exdisk1/data/PanSTARRS/COSMOS/):
  ps1_cosmos.parquet         canonical (MeanObject in Euclid VIS polygon)
  ps1_cosmos.csv             human-readable sidecar
  ps1_cosmos.meta.json       timestamp + row counts + bbox + columns
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Euclid VIS extent (deg) — matches Gaia and SDSS pulls
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

# Cone search centre + radius covers the entire bbox
CENTRE_RA  = 0.5 * (RA_MIN + RA_MAX)
CENTRE_DEC = 0.5 * (DEC_MIN + DEC_MAX)
# half-diagonal of the bbox + 0.05 deg safety pad
import math
HALF_DIAG_DEG = 0.5 * math.hypot(RA_MAX - RA_MIN,
                                   DEC_MAX - DEC_MIN) + 0.05

OUT_DIR = Path('/Volumes/exdisk1/data/PanSTARRS/COSMOS')

# MAST PS1 DR2 MeanObject column subset.  Names follow PS1 schema docs:
# https://outerspace.stsci.edu/display/PANSTARRS/PS1+MeanObject+table+fields
COLUMNS = [
    # identifiers + flags
    'objID', 'objInfoFlag', 'qualityFlag',
    # position
    'raMean', 'decMean', 'raMeanErr', 'decMeanErr',
    # per-band detection counts
    'nDetections', 'ng', 'nr', 'ni', 'nz', 'ny',
    # PSF mean magnitudes + errors
    'gMeanPSFMag', 'gMeanPSFMagErr',
    'rMeanPSFMag', 'rMeanPSFMagErr',
    'iMeanPSFMag', 'iMeanPSFMagErr',
    'zMeanPSFMag', 'zMeanPSFMagErr',
    'yMeanPSFMag', 'yMeanPSFMagErr',
    # Kron mean magnitudes + errors (better for extended sources)
    'gMeanKronMag', 'gMeanKronMagErr',
    'rMeanKronMag', 'rMeanKronMagErr',
    'iMeanKronMag', 'iMeanKronMagErr',
    'zMeanKronMag', 'zMeanKronMagErr',
    'yMeanKronMag', 'yMeanKronMagErr',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Print params + exit; do not contact MAST.')
    p.add_argument('--table', default='mean',
                   choices=['mean', 'stack', 'forced_mean'],
                   help='Which PS1 DR2 catalog view (default mean).')
    p.add_argument('--ndet-min', type=int, default=1,
                   help='Server-side filter nDetections >= this (default 1).')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'ps1_cosmos'

    print('PS1 DR2 cone-search params:')
    print(f'  table        = {args.table}')
    print(f'  centre       = ({CENTRE_RA:.4f}, {CENTRE_DEC:.4f})  deg')
    print(f'  radius       = {HALF_DIAG_DEG:.4f} deg  (covers bbox diagonal + 0.05° pad)')
    print(f'  bbox post-filter via Shapely:')
    print(f'      RA   {RA_MIN}..{RA_MAX}')
    print(f'      Dec  {DEC_MIN}..{DEC_MAX}')
    print(f'  + Euclid VIS polygon filter (3.91 deg²)')
    print(f'  columns      = {len(COLUMNS)} ({COLUMNS[:5]} ...)')
    print()

    if args.dry_run:
        print('[dry-run] not contacting MAST.')
        return

    from astroquery.mast import Catalogs
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # MAST PS1 returns 504 / astroquery times out at 600s if a single
    # cone is too big.  Split the bbox into an 8x8 grid of sub-cones
    # (64 total), each radius ~0.22 deg covering one bbox cell;
    # concatenate + dedupe by objID.  Each sub-cone returns roughly
    # 40-60k rows in 10-20 s.
    N_RA, N_DEC = 8, 8
    import numpy as np
    ra_edges  = np.linspace(RA_MIN,  RA_MAX,  N_RA+1)
    dec_edges = np.linspace(DEC_MIN, DEC_MAX, N_DEC+1)
    sub_radius = 0.5 * math.hypot((RA_MAX-RA_MIN)/N_RA,
                                  (DEC_MAX-DEC_MIN)/N_DEC) + 0.02
    print(f'Splitting query into {N_RA}x{N_DEC}={N_RA*N_DEC} sub-cones '
          f'(radius {sub_radius:.3f} deg each) ...', flush=True)

    import pandas as pd
    parts = []
    t0 = time.time()
    for i in range(N_RA):
        for j in range(N_DEC):
            ra_c  = 0.5 * (ra_edges[i]  + ra_edges[i+1])
            dec_c = 0.5 * (dec_edges[j] + dec_edges[j+1])
            coord = SkyCoord(ra_c, dec_c, unit='deg', frame='icrs')
            try:
                t = Catalogs.query_region(
                    coord, radius=sub_radius * u.deg,
                    catalog='PANSTARRS',
                    data_release='dr2',
                    table=args.table,
                    columns=COLUMNS,
                )
            except Exception as e:
                print(f'  [{i},{j}] FAIL ({ra_c:.3f},{dec_c:.3f}): '
                      f'{e!r}', file=sys.stderr)
                sys.exit(1)
            n = 0 if t is None else len(t)
            print(f'  [{i},{j}] ({ra_c:.3f},{dec_c:.3f})  {n:,} rows '
                  f'({time.time()-t0:.1f}s elapsed)', flush=True)
            if n > 0:
                parts.append(t.to_pandas())
    dt = time.time() - t0
    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)
    print(f'  total raw: {n_raw:,} rows in {dt:.1f}s')

    # Dedupe by objID (sub-cones overlap at edges)
    df = df.drop_duplicates(subset=['objID']).reset_index(drop=True)
    n_cone = len(df)
    print(f'  after dedupe by objID: {n_cone:,}')

    # Bbox crop (cones may extend slightly past bbox)
    df = df[(df['raMean']  >= RA_MIN)  & (df['raMean']  <= RA_MAX) &
            (df['decMean'] >= DEC_MIN) & (df['decMean'] <= DEC_MAX)
           ].reset_index(drop=True)
    print(f'  after bbox crop: {len(df):,} / {n_cone:,}  '
          f'({100*len(df)/n_cone:.1f}%)')

    # ── Polygon filter to exact Euclid VIS footprint ───────────────────
    n_bbox = len(df)
    print(f'\nFiltering through the exact Euclid VIS polygon ...')
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
    print(f'  Euclid VIS polygon area : {vis_polygon.area:.4f} deg²')
    pts = points(df['raMean'].values, df['decMean'].values)
    df_in = df[vis_polygon.contains(pts)].reset_index(drop=True)
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
        'row_count_cone':           int(n_cone),
        'row_count_bbox':           int(n_bbox),
        'row_count_in_vis_polygon': int(len(df)),
        'cone_centre_deg':  [CENTRE_RA, CENTRE_DEC],
        'cone_radius_deg':  HALF_DIAG_DEG,
        'bbox_deg': {
            'ra_min':  RA_MIN,  'ra_max':  RA_MAX,
            'dec_min': DEC_MIN, 'dec_max': DEC_MAX,
        },
        'vis_polygon_deg2':       round(vis_polygon.area, 4),
        'spatial_selection':      'MAST cone + bbox crop + Shapely VIS polygon',
        'ps1_table':     f'DR2 {args.table}',
        'columns':       list(df.columns),
        'parquet_path':  str(parquet_path),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))
    print(f'[save] ps1_cosmos.meta.json')

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print('=== PS1 DR2 MeanObject in the Euclid VIS polygon ===')
    print(f'  rows                : {len(df):>8,}')
    for b in ('g', 'r', 'i', 'z', 'y'):
        col = f'{b}MeanPSFMag'
        if col in df.columns:
            n22 = (df[col] < 22).sum()
            n20 = (df[col] < 20).sum()
            print(f'  {b}<22 / {b}<20         : {n22:>8,} / {n20:>8,}')
    print(f'  nDetections >= 5    : {(df["nDetections"] >= 5).sum():>8,}')


if __name__ == '__main__':
    main()
