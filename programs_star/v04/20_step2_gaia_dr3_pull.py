#!/usr/bin/env python
"""
20_step2_gaia_dr3_pull.py — Step 2.1: pull Gaia DR3 over the v04 area.

Step 2 collects external catalogs over the v04 common-area footprint.
This is the first sub-task: Gaia DR3 as the astrometric anchor
(positions, PM, parallax, basic photometry, point-source flags).

Spatial selection (two-stage):
  1. ADQL bbox query at the bbox of Euclid VIS (slightly padded) — the
     widest single rectangle that contains all 60 Euclid VIS MER tile
     polygons:
        bbox  RA  148.85 -> 151.30   (~2.45 deg wide)
              Dec   1.10 ->   3.40   (~2.30 deg tall)
        area  ~5.6 deg^2
  2. Shapely polygon filter at the EXACT Euclid VIS footprint (union
     of all 60 tile polygons, ~3.91 deg^2).  Sources that fall in the
     bbox corners outside the VIS polygon are dropped.  The output
     file therefore contains only sources within actual Euclid VIS
     coverage — the biggest of our four mission footprints, but no
     wider.

Outputs (to /Volumes/exdisk1/data/GAIA_DR3/COSMOS/):
  gaia_dr3_cosmos.parquet         canonical
  gaia_dr3_cosmos.csv             human-readable sidecar
  gaia_dr3_cosmos.query.adql      exact ADQL used (reproducibility)
  gaia_dr3_cosmos.meta.json       query timestamp + row count + bbox

Columns retrieved (32 of the gaia_source table's ~150):
  identifiers   : source_id, ref_epoch
  astrometry    : ra, dec, ra_error, dec_error,
                  parallax, parallax_error,
                  pmra, pmra_error, pmdec, pmdec_error
  photometry    : phot_{g,bp,rp}_mean_mag,
                  phot_{g,bp,rp}_mean_flux(+_error),
                  bp_rp, bp_g, g_rp
  quality       : ruwe, astrometric_excess_noise(+_sig),
                  astrometric_n_good_obs_al, visibility_periods_used,
                  phot_variable_flag
  classification: in_qso_candidates, in_galaxy_candidates, non_single_star
  GSP-Phot      : ag_gspphot, ebpminrp_gspphot, teff_gspphot, logg_gspphot

Usage:
  python 20_step2_gaia_dr3_pull.py            # async pull (recommended)
  python 20_step2_gaia_dr3_pull.py --sync     # synchronous
  python 20_step2_gaia_dr3_pull.py --dry-run  # print ADQL only, don't query
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Spatial window (deg).  See module docstring for rationale.
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/GAIA_DR3/COSMOS')

ADQL = f"""
SELECT
  source_id,
  ref_epoch,
  ra, dec, ra_error, dec_error,
  parallax, parallax_error,
  pmra, pmra_error,
  pmdec, pmdec_error,
  phot_g_mean_mag,  phot_g_mean_flux,  phot_g_mean_flux_error,
  phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error,
  phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error,
  bp_rp, bp_g, g_rp,
  ruwe,
  astrometric_excess_noise,
  astrometric_excess_noise_sig,
  astrometric_n_good_obs_al,
  visibility_periods_used,
  phot_variable_flag,
  in_qso_candidates,
  in_galaxy_candidates,
  non_single_star,
  ag_gspphot,
  ebpminrp_gspphot,
  teff_gspphot,
  logg_gspphot
FROM gaiadr3.gaia_source
WHERE ra  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND dec BETWEEN {DEC_MIN} AND {DEC_MAX}
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sync', action='store_true',
                   help='Use synchronous TAP+ query (default: async; safer for >2M rows but our area is small).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print the ADQL and exit; do not contact ESA.')
    return p.parse_args()


def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'gaia_dr3_cosmos'

    print('ADQL query:')
    print('-' * 60)
    print(ADQL)
    print('-' * 60)
    print()

    # Always persist the exact query for reproducibility
    (base.with_suffix('.query.adql')).write_text(ADQL + '\n')

    if args.dry_run:
        print('[dry-run] not contacting ESA.  ADQL saved to:',
              base.with_suffix('.query.adql'))
        return

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1   # unlimited

    print('Submitting query to ESA Gaia archive '
          f'({"sync" if args.sync else "async"}) ...', flush=True)
    t0 = time.time()
    try:
        if args.sync:
            job = Gaia.launch_job(ADQL, output_format='votable',
                                   verbose=False)
        else:
            job = Gaia.launch_job_async(ADQL, output_format='votable',
                                         verbose=False)
        tbl = job.get_results()
    except Exception as e:
        print(f'ESA query FAILED: {e!r}', file=sys.stderr)
        sys.exit(1)
    dt = time.time() - t0
    print(f'Returned {len(tbl):,} rows in {dt:.1f}s.')

    # ── Save ───────────────────────────────────────────────────────────
    import pandas as pd
    df = tbl.to_pandas()

    # phot_variable_flag is a fixed-length string in Gaia DR3
    if 'phot_variable_flag' in df.columns:
        df['phot_variable_flag'] = df['phot_variable_flag'].astype(str)

    # ── Polygon filter to exact Euclid VIS footprint ────────────────────
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
    df_in  = df[inside].reset_index(drop=True)
    print(f'  inside VIS polygon : {len(df_in):,} / {n_bbox:,}  '
          f'({100*len(df_in)/n_bbox:.1f}%)')
    df = df_in   # canonical = polygon-filtered only

    parquet_path = base.with_suffix('.parquet')
    csv_path     = base.with_suffix('.csv')
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f'[save] {parquet_path.name}  '
          f'({parquet_path.stat().st_size/1024/1024:.1f} MB,  {len(df):,} rows)')
    print(f'[save] {csv_path.name}      '
          f'({csv_path.stat().st_size/1024/1024:.1f} MB)')

    # Meta
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
        'spatial_selection':      'ADQL bbox at Euclid VIS extent, then '
                                  'Shapely point-in-polygon filter to the '
                                  'exact Euclid VIS footprint.',
        'gaia_table':    'gaiadr3.gaia_source',
        'mode':          'sync' if args.sync else 'async',
        'columns':       list(df.columns),
        'parquet_path':  str(parquet_path),
        'adql_path':     str(base.with_suffix('.query.adql')),
    }
    meta_path = base.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[save] {meta_path.name}')

    # ── Quick summary ──────────────────────────────────────────────────
    print()
    print('=== Gaia DR3 summary for the v04 query box ===')
    print(f'  rows                : {len(df):>8,}')
    print(f'  with parallax > 0   : {(df["parallax"] > 0).sum():>8,}')
    print(f'  with PM measured    : {df[["pmra","pmdec"]].notna().all(axis=1).sum():>8,}')
    print(f'  G < 18 (bright)     : {(df["phot_g_mean_mag"] < 18).sum():>8,}')
    print(f'  G < 14 (very bright): {(df["phot_g_mean_mag"] < 14).sum():>8,}')
    if 'ruwe' in df.columns:
        print(f'  RUWE < 1.4 (clean)  : {(df["ruwe"] < 1.4).sum():>8,}')
    print(f'  in_qso_candidates   : {df["in_qso_candidates"].sum():>8,}')
    print(f'  in_galaxy_candidates: {df["in_galaxy_candidates"].sum():>8,}')
    print(f'  non_single_star  >0 : {(df["non_single_star"] > 0).sum():>8,}')


if __name__ == '__main__':
    main()
