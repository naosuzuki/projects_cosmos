#!/usr/bin/env python
"""
34_step2_galex_pull.py — Step 2.5: pull GALEX GR6+7 catalog over the
v04 Euclid VIS polygon.

GALEX (Galaxy Evolution Explorer, 2003-2013) is the only UV catalog
covering COSMOS at moderate depth (NUV 5σ ~ 22.6, FUV 5σ ~ 22.3 in the
deepest tile).  Two bands FUV (~1540 Å) + NUV (~2300 Å) extend our
optical SEDs blueward of u-band.

Source: MAST Catalogs API (not NOIRLab Datalab — Datalab only carries
cross-match tables, not the GALEX primary catalog).  Sister to the
PS1 27_ cone-search script.

Spatial selection: cone search at MAST + bbox crop + Shapely VIS
polygon filter (3.91 deg²).  Splits the bbox into a small grid of
sub-cones to dodge the 504/timeout cluster MAST hits at large radii.

Outputs (to /Volumes/exdisk1/data/catalog/GALEX/COSMOS/):
  galex_cosmos.{parquet,csv,meta.json}
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Euclid VIS bbox — identical to Gaia/SDSS/PS1/LS DR10 pulls
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/catalog/GALEX/COSMOS')

COLUMNS = [
    # identifiers + classification
    'objID', 'survey', 'IAUName', 'band',
    # position
    'ra', 'dec',
    # field geometry
    'fov_radius', 'ra_cent', 'dec_cent',
    # exposure
    'nuv_exptime', 'fuv_exptime',
    # photometry (mag)
    'fuv_mag', 'fuv_magerr', 'nuv_mag', 'nuv_magerr',
    # photometry (flux, μJy)
    'fuv_flux', 'fuv_fluxerr', 'nuv_flux', 'nuv_fluxerr',
    'nuv_flux_auto', 'nuv_flux_aper_7',
    'fuv_flux_auto', 'fuv_flux_aper_7',
    # extinction + quality
    'e_bv', 'nuv_artifact', 'fuv_artifact',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--n-ra', type=int, default=4)
    p.add_argument('--n-dec', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'galex_cosmos'

    print('GALEX GR6+7 (MAST cone-search) parameters:')
    print(f'  bbox       : RA {RA_MIN}..{RA_MAX}, Dec {DEC_MIN}..{DEC_MAX}')
    print(f'  sub-cones  : {args.n_ra}x{args.n_dec} = {args.n_ra*args.n_dec}')
    print(f'  columns    : {len(COLUMNS)}')
    print(f'  + VIS polygon (3.91 deg²) Shapely filter')
    print()

    if args.dry_run:
        print('[dry-run]')
        return

    from astroquery.mast import Catalogs
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import numpy as np, pandas as pd

    ra_edges  = np.linspace(RA_MIN,  RA_MAX,  args.n_ra  + 1)
    dec_edges = np.linspace(DEC_MIN, DEC_MAX, args.n_dec + 1)
    sub_r = 0.5 * math.hypot((RA_MAX-RA_MIN)/args.n_ra,
                              (DEC_MAX-DEC_MIN)/args.n_dec) + 0.02

    print(f'  sub-radius : {sub_r:.3f} deg')
    print()
    parts = []
    t0 = time.time()
    for i in range(args.n_ra):
        for j in range(args.n_dec):
            ra_c  = 0.5*(ra_edges[i]  + ra_edges[i+1])
            dec_c = 0.5*(dec_edges[j] + dec_edges[j+1])
            coord = SkyCoord(ra_c, dec_c, unit='deg', frame='icrs')
            try:
                # MAST GALEX catalog doesn't accept a `columns` filter,
                # so we fetch the full row set and trim post-hoc.
                t = Catalogs.query_region(coord, radius=sub_r*u.deg,
                                          catalog='GALEX')
            except Exception as e:
                print(f'  [{i},{j}] FAIL: {e!r}', file=sys.stderr); sys.exit(1)
            n = 0 if t is None else len(t)
            print(f'  [{i},{j}] ({ra_c:.3f},{dec_c:.3f}) {n:,} rows '
                  f'({time.time()-t0:.1f}s)', flush=True)
            if n: parts.append(t.to_pandas())

    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset=['objID']).reset_index(drop=True)
    n_cone = len(df)
    print(f'  raw {n_raw:,} → dedup objID {n_cone:,}')

    # Bbox + polygon
    df = df[(df.ra.between(RA_MIN, RA_MAX)) &
            (df.dec.between(DEC_MIN, DEC_MAX))].reset_index(drop=True)
    n_bbox = len(df)
    print(f'  bbox crop  : {n_bbox:,} ({100*n_bbox/n_cone:.1f}%)')

    from shapely import points
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    LOOKUP = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/tile_lookup.parquet')
    tdf = pd.read_parquet(LOOKUP)
    vis = tdf[(tdf['mission'] == 'Euclid') & (tdf['filter'] == 'VIS')]
    polys = []
    for _, r in vis.iterrows():
        ring = list(zip(r['corners_ra'], r['corners_dec'])); ring.append(ring[0])
        p = Polygon(ring); polys.append(p if p.is_valid else p.buffer(0))
    vis_poly = unary_union(polys)
    pts = points(df['ra'].values, df['dec'].values)
    df_in = df[vis_poly.contains(pts)].reset_index(drop=True)
    print(f'  VIS polygon: {len(df_in):,} ({100*len(df_in)/n_bbox:.1f}%)')
    df = df_in

    df.to_parquet(base.with_suffix('.parquet'), index=False)
    df.to_csv(base.with_suffix('.csv'), index=False)
    dt = time.time() - t0
    print(f'\n[save] {base.with_suffix(".parquet").name}  '
          f'({base.with_suffix(".parquet").stat().st_size/1024/1024:.1f} MB, {len(df):,})')

    meta = {
        'query_utc_iso':           time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':            round(dt, 2),
        'method':                  f'MAST cone-search {args.n_ra}x{args.n_dec} sub-cones',
        'row_count_cone':          int(n_cone),
        'row_count_bbox':          int(n_bbox),
        'row_count_in_vis_polygon':int(len(df)),
        'bbox_deg': {'ra_min':RA_MIN,'ra_max':RA_MAX,'dec_min':DEC_MIN,'dec_max':DEC_MAX},
        'vis_polygon_deg2':        round(vis_poly.area, 4),
        'columns':                 list(df.columns),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))
    print(f'[save] {base.with_suffix(".meta.json").name}')

    print()
    print('=== GALEX GR6+7 in the Euclid VIS polygon ===')
    print(f'  rows                  : {len(df):>6,}')
    for b in ('nuv', 'fuv'):
        col = f'{b}_mag'
        if col in df.columns:
            d = df[col].dropna()
            n22 = (d < 22).sum(); n21 = (d < 21).sum()
            print(f'  {b.upper()}<21 / {b.upper()}<22         : {n21:>6,} / {n22:>6,}')


if __name__ == '__main__':
    main()
