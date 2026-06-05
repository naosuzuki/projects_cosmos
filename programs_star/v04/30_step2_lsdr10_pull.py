#!/usr/bin/env python
"""
30_step2_lsdr10_pull.py — Step 2.4: pull DESI Legacy Survey DR10
(Tractor catalog) over the v04 Euclid VIS polygon.

Sister of 20_/22_/27_ for Gaia DR3, SDSS DR17, PS1 DR2.

What is LS DR10?
  The DESI Legacy Imaging Surveys (Dey+ 2019) processed all available
  ground-based imaging into a unified grizW1W2W3W4 catalog using the
  Tractor forward-modeling pipeline.  DR10 (2023-04) is the latest
  release with full coverage in i band (newly added in DR10) for the
  COSMOS area via DECaPS/PS1 coadds.

  Each source is modeled as PSF / REX (round exp) / EXP (exponential)
  / DEV (de Vaucouleurs) / SER (Sérsic) and has Tractor-fit fluxes
  in 4 optical + 4 WISE bands.  For COSMOS at Dec ~ 2°, the optical
  imaging is dominated by DECaLS (DECam) and the WISE imaging by the
  unWISE coadds.

Spatial selection (two-stage like the other 2x scripts):
  1. ADQL bbox query at NOIRLab Datalab TAP:
       RA  [148.85, 151.30],  Dec [1.10, 3.40]  (~5.6 deg²)
  2. Shapely point-in-polygon to the exact 3.91 deg² Euclid VIS
     footprint.

Outputs (to /Volumes/exdisk1/data/catalog/DESI_Legacy/COSMOS/):
  lsdr10_cosmos.parquet         canonical (in VIS polygon)
  lsdr10_cosmos.csv             human-readable sidecar
  lsdr10_cosmos.query.adql      reproducible ADQL
  lsdr10_cosmos.meta.json       timestamp + row counts + bbox

Columns retrieved (~50):
  identifiers   : ls_id, brickid, brickname, objid
  position      : ra, dec, ra_ivar, dec_ivar
  morphology    : type (PSF/REX/EXP/DEV/SER), shape_r, shape_e1,
                  shape_e2, sersic
  photometry    : flux_{g,r,i,z,w1,w2,w3,w4} + flux_ivar
  extinction    : ebv, mw_transmission_{g,r,i,z,w1,w2,w3,w4}
  depth         : nobs_{g,r,i,z}, psfsize_{g,r,i,z},
                  psfdepth_{g,r,i,z}, galdepth_{g,r,i,z}
  quality       : maskbits, fitbits
  Gaia anchor   : gaia_phot_g_mean_mag, gaia_phot_bp_mean_mag,
                  gaia_phot_rp_mean_mag, ref_id, ref_cat,
                  pmra, pmdec  (Gaia DR3 PMs propagated to the
                  Tractor reference epoch when ref_cat matches)

Usage:
  python 30_step2_lsdr10_pull.py            # async pull (default)
  python 30_step2_lsdr10_pull.py --sync     # synchronous
  python 30_step2_lsdr10_pull.py --dry-run  # print ADQL only
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Spatial window — identical to Gaia/SDSS/PS1 pulls
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/catalog/DESI_Legacy/COSMOS')

TAP_URL = 'https://datalab.noirlab.edu/tap'
TABLE   = 'ls_dr10.tractor'

# Column list — kept narrow but covers all DR10 essentials.  See PSF /
# REX / EXP / DEV / SER taxonomy in Dey+2019 §4.5 and DR10 docs at
# https://www.legacysurvey.org/dr10/catalogs/
COLUMNS = [
    # identifiers
    'ls_id', 'brickid', 'brickname', 'objid',
    # position
    'ra', 'dec', 'ra_ivar', 'dec_ivar',
    # morphology
    'type', 'shape_r', 'shape_e1', 'shape_e2', 'sersic', 'sersic_ivar',
    # extinction
    'ebv',
    'mw_transmission_g', 'mw_transmission_r',
    'mw_transmission_i', 'mw_transmission_z',
    'mw_transmission_w1', 'mw_transmission_w2',
    'mw_transmission_w3', 'mw_transmission_w4',
    # optical fluxes (nanomaggies, AB) + ivar
    'flux_g',      'flux_r',      'flux_i',      'flux_z',
    'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_i', 'flux_ivar_z',
    # WISE fluxes + ivar
    'flux_w1',      'flux_w2',      'flux_w3',      'flux_w4',
    'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4',
    # depth / coverage
    'nobs_g',     'nobs_r',     'nobs_i',     'nobs_z',
    'psfsize_g',  'psfsize_r',  'psfsize_i',  'psfsize_z',
    'psfdepth_g', 'psfdepth_r', 'psfdepth_i', 'psfdepth_z',
    'galdepth_g', 'galdepth_r', 'galdepth_i', 'galdepth_z',
    # quality
    'maskbits', 'fitbits',
    # Gaia anchor / external reference
    'gaia_phot_g_mean_mag',
    'gaia_phot_bp_mean_mag',
    'gaia_phot_rp_mean_mag',
    'ref_id', 'ref_cat',
    'pmra', 'pmdec',
]

ADQL = f"""
SELECT
  {','.join(COLUMNS)}
FROM {TABLE}
WHERE ra  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND dec BETWEEN {DEC_MIN} AND {DEC_MAX}
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Print ADQL only; do not contact Datalab.')
    p.add_argument('--sync', action='store_true',
                   help='Use synchronous TAP query (faster for small '
                        'pulls; default is async because LS DR10 may '
                        'return >100k rows in COSMOS).')
    p.add_argument('--timeout', type=int, default=1800,
                   help='Maximum wait time for async job (s, default 1800).')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'lsdr10_cosmos'

    print('DESI Legacy Survey DR10 (NOIRLab Datalab TAP) query:')
    print(f'  service   : {TAP_URL}')
    print(f'  table     : {TABLE}')
    print(f'  bbox      : RA  {RA_MIN}..{RA_MAX}   Dec {DEC_MIN}..{DEC_MAX}')
    print(f'  columns   : {len(COLUMNS)}')
    print(f'  + Euclid VIS polygon post-filter (3.91 deg²)')
    print()
    print('-' * 60)
    print(ADQL)
    print('-' * 60)
    print()

    (base.with_suffix('.query.adql')).write_text(ADQL + '\n')

    if args.dry_run:
        print('[dry-run] ADQL saved, not contacting Datalab.')
        return

    import pyvo
    tap = pyvo.dal.TAPService(TAP_URL)
    print(f'TAP service capabilities found '
          f'({"sync" if args.sync else "async"} mode) ...', flush=True)

    t0 = time.time()
    try:
        if args.sync:
            result = tap.search(ADQL)
        else:
            job = tap.submit_job(ADQL)
            print(f'  submitted async job {job.job_id}', flush=True)
            job.run()
            job.wait(phases=['COMPLETED', 'ERROR', 'ABORTED'],
                     timeout=args.timeout)
            phase = job.phase
            print(f'  job phase: {phase}  (wall {time.time()-t0:.1f}s)',
                  flush=True)
            if phase != 'COMPLETED':
                # Try to fetch the error log
                try:
                    err = job.fetch_result()
                except Exception as e:
                    err = repr(e)
                sys.exit(f'TAP job ended in phase {phase}: {err}')
            result = job.fetch_result()
    except Exception as e:
        print(f'TAP query FAILED: {e!r}', file=sys.stderr)
        sys.exit(1)
    dt = time.time() - t0

    tbl = result.to_table()
    n_bbox = len(tbl)
    print(f'\nReturned {n_bbox:,} rows in {dt:.1f}s '
          f'({n_bbox/max(dt,1):.0f} rows/s)')

    import pandas as pd
    df = tbl.to_pandas()
    # ── Decode bytes columns (TAP may return bytes for strings) ─────────
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8', errors='replace')

    # ── Polygon filter to exact Euclid VIS footprint ───────────────────
    print(f'\nFiltering through the exact Euclid VIS polygon ...')
    from shapely import points
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    LOOKUP = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                  'tile_lookup.parquet')
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
    pts = points(df['ra'].values, df['dec'].values)
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
        'tap_service':   TAP_URL,
        'table':         TABLE,
        'mode':          'sync' if args.sync else 'async',
        'row_count_bbox':           int(n_bbox),
        'row_count_in_vis_polygon': int(len(df)),
        'bbox_deg': {
            'ra_min':  RA_MIN,  'ra_max':  RA_MAX,
            'dec_min': DEC_MIN, 'dec_max': DEC_MAX,
        },
        'vis_polygon_deg2':  round(vis_polygon.area, 4),
        'spatial_selection': 'TAP bbox + Shapely VIS polygon',
        'columns':           list(df.columns),
        'parquet_path':      str(parquet_path),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))
    print(f'[save] lsdr10_cosmos.meta.json')

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print('=== LS DR10 Tractor catalog in the Euclid VIS polygon ===')
    print(f'  rows                : {len(df):>8,}')
    if 'type' in df.columns:
        for ty in ('PSF', 'REX', 'EXP', 'DEV', 'SER'):
            n = (df['type'] == ty).sum()
            print(f'  type = {ty:<4s}         : {n:>8,}')
    # Magnitude proxy: nanomaggies → AB mag = 22.5 - 2.5*log10(flux)
    import numpy as np
    for b in ('g', 'r', 'i', 'z'):
        fcol = f'flux_{b}'
        if fcol in df.columns:
            flux = df[fcol].values
            ok = flux > 0
            mag = np.full(len(df), np.nan)
            mag[ok] = 22.5 - 2.5*np.log10(flux[ok])
            n22 = ((mag < 22) & np.isfinite(mag)).sum()
            n24 = ((mag < 24) & np.isfinite(mag)).sum()
            print(f'  {b}<22 / {b}<24         : {n22:>8,} / {n24:>8,}')
    if 'ref_cat' in df.columns:
        n_gaia = df['ref_cat'].isin(['G2', 'GE']).sum()  # Gaia DR2 / EDR3
        print(f'  Gaia-anchored        : {n_gaia:>8,}')


if __name__ == '__main__':
    main()
