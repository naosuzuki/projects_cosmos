#!/usr/bin/env python
"""
36_step2_unwise_pull.py — Step 2.6: pull unWISE DR1 (Schlafly+ 2019)
forced-photometry catalog over the Euclid VIS polygon.

unWISE (Lang 2014; Schlafly+ 2019) re-coadds the WISE+NEOWISE imaging
to deeper limits than AllWISE.  DR1 W1 5σ depth is 0.5-1 mag deeper
than AllWISE (W1 ≈ 19.6 vs 18.5 AB).  unWISE is the cleanest standalone
mid-IR catalog when you want sources LS DR10 may have missed at the
bright end (saturation) or in mask-flagged regions.

Note: LS DR10 already carries unWISE forced fluxes (flux_w1..w4) at
LS DR10 source positions, so this standalone pull is primarily for
W1+W2 measurements at positions LS DR10 does not have.

Source: NOIRLab Datalab TAP table `unwise_dr1.object` (Schlafly+ 2019
unwise2019).  Async TAP query.

Outputs (to /Volumes/exdisk1/data/catalog/unWISE/COSMOS/):
  unwise_cosmos.{parquet,csv,query.adql,meta.json}
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Euclid VIS bbox
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/catalog/unWISE/COSMOS')

TAP_URL = 'https://datalab.noirlab.edu/tap'
TABLE   = 'unwise_dr1.object'

COLUMNS = [
    # identifiers
    'unwise_objid', 'coadd_id',
    # position
    'ra', 'dec',
    # W1 forced fluxes + uncertainties (nanomaggies, AB)
    'flux_w1', 'dflux_w1', 'fluxlbs_w1', 'dfluxlbs_w1',
    # W2 forced fluxes
    'flux_w2', 'dflux_w2', 'fluxlbs_w2', 'dfluxlbs_w2',
    # Quality + chi^2
    'qf_w1', 'qf_w2', 'rchi2_w1', 'rchi2_w2',
    'fracflux_w1', 'fracflux_w2',
    'fwhm_w1', 'fwhm_w2',
    'spread_model_w1', 'spread_model_w2',
    'sky_w1', 'sky_w2',
    # Vega magnitudes + W1-W2 colour
    'mag_w1_vg', 'mag_w2_vg', 'w1_w2_vg',
    # Number of measurements
    'nm_w1', 'nm_w2',
    # Flags  (note: "primary" is a SQL reserved word, must be double-quoted)
    '"primary"', 'flags_unwise_w1', 'flags_unwise_w2',
    'flags_info_w1', 'flags_info_w2',
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
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--sync', action='store_true')
    p.add_argument('--timeout', type=int, default=1800)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'unwise_cosmos'

    print('unWISE DR1 (NOIRLab Datalab TAP) query:')
    print(f'  table     : {TABLE}')
    print(f'  bbox      : RA {RA_MIN}..{RA_MAX}, Dec {DEC_MIN}..{DEC_MAX}')
    print(f'  columns   : {len(COLUMNS)}')
    print(f'  + VIS polygon (3.91 deg²) Shapely filter')
    print()
    (base.with_suffix('.query.adql')).write_text(ADQL + '\n')

    if args.dry_run:
        print('[dry-run]')
        return

    import pyvo
    tap = pyvo.dal.TAPService(TAP_URL)
    t0 = time.time()
    try:
        if args.sync:
            result = tap.search(ADQL)
        else:
            job = tap.submit_job(ADQL)
            print(f'  submitted job {job.job_id}', flush=True)
            job.run()
            job.wait(phases=['COMPLETED','ERROR','ABORTED'], timeout=args.timeout)
            phase = job.phase
            print(f'  phase: {phase} (wall {time.time()-t0:.1f}s)', flush=True)
            if phase != 'COMPLETED':
                sys.exit(f'TAP job ended in {phase}')
            result = job.fetch_result()
    except Exception as e:
        print(f'TAP query FAILED: {e!r}', file=sys.stderr); sys.exit(1)
    dt = time.time() - t0

    tbl = result.to_table()
    n_bbox = len(tbl)
    print(f'  returned {n_bbox:,} rows in {dt:.1f}s')

    import pandas as pd
    df = tbl.to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().head(1)
            if len(s) and isinstance(s.iloc[0], bytes):
                df[c] = df[c].str.decode('utf-8', errors='replace')

    from shapely import points
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    LOOKUP = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/tile_lookup.parquet')
    tdf = pd.read_parquet(LOOKUP)
    vis = tdf[(tdf['mission']=='Euclid')&(tdf['filter']=='VIS')]
    polys = []
    for _, r in vis.iterrows():
        ring = list(zip(r['corners_ra'], r['corners_dec'])); ring.append(ring[0])
        p = Polygon(ring); polys.append(p if p.is_valid else p.buffer(0))
    vis_poly = unary_union(polys)
    pts = points(df['ra'].values, df['dec'].values)
    df_in = df[vis_poly.contains(pts)].reset_index(drop=True)
    print(f'  in VIS polygon: {len(df_in):,} ({100*len(df_in)/n_bbox:.1f}%)')
    df = df_in

    df.to_parquet(base.with_suffix('.parquet'), index=False)
    df.to_csv(base.with_suffix('.csv'), index=False)
    print(f'[save] {base.with_suffix(".parquet").name}  '
          f'({base.with_suffix(".parquet").stat().st_size/1024/1024:.1f} MB, {len(df):,})')

    meta = {
        'query_utc_iso':           time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_seconds':            round(dt, 2),
        'tap_service':             TAP_URL,
        'table':                   TABLE,
        'mode':                    'sync' if args.sync else 'async',
        'row_count_bbox':          int(n_bbox),
        'row_count_in_vis_polygon':int(len(df)),
        'bbox_deg': {'ra_min':RA_MIN,'ra_max':RA_MAX,'dec_min':DEC_MIN,'dec_max':DEC_MAX},
        'vis_polygon_deg2':        round(vis_poly.area, 4),
        'columns':                 list(df.columns),
    }
    (base.with_suffix('.meta.json')).write_text(json.dumps(meta, indent=2))

    print()
    print('=== unWISE DR1 in the Euclid VIS polygon ===')
    print(f'  rows                : {len(df):>8,}')
    print(f'  primary == 1        : {(df["primary"]==1).sum():>8,}')
    print(f'  mag_w1_vg < 17 (Vega): {((df["mag_w1_vg"]<17)&df["mag_w1_vg"].notna()).sum():>8,}')
    print(f'  mag_w1_vg < 19      : {((df["mag_w1_vg"]<19)&df["mag_w1_vg"].notna()).sum():>8,}')
    print(f'  flux_w1 / dflux_w1 > 5 (5σ W1) : {((df["flux_w1"]/df["dflux_w1"])>5).sum():>8,}')


if __name__ == '__main__':
    main()
