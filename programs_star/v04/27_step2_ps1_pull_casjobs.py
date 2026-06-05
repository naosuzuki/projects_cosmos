#!/usr/bin/env python
"""
27_step2_ps1_pull_casjobs.py — Step 2.3 (CasJobs flavour): pull
Pan-STARRS 1 DR2 MeanObject over the v04 Euclid VIS bbox via one
SQL job at the PS1 CasJobs SOAP service.

Why CasJobs and not the MAST Catalogs cone-search API?
  The MAST Catalogs API only accepts cone queries.  For our 6 deg² bbox
  a single cone returns ~2 M rows and the API returns 504 / astroquery
  times out at 600 s.  Splitting into 64 sub-cones works but takes
  35-60 min wall because individual sub-cones get queued unpredictably.
  CasJobs accepts a SQL box query, runs it server-side in one job,
  and returns the full result table in ~1-2 min.

Credentials:
  reads $CASJOBS_WSID and $CASJOBS_PW (set them in ~/.zshenv).  Get
  the WSID from https://mastweb.stsci.edu/ps1casjobs/ → Profile.

Output is identical to the cone-search version so the rest of the
pipeline (Step 2.4 3-way subset, Step 2.5 image download) doesn't care
which puller produced it.

Outputs (to /Volumes/exdisk1/data/catalog/PanSTARRS/COSMOS/):
  ps1_cosmos.parquet         in Euclid VIS polygon
  ps1_cosmos.csv             human-readable sidecar
  ps1_cosmos.query.sql       the SQL we ran
  ps1_cosmos.meta.json       timestamp + row counts + bbox + columns
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── v04 spatial parameters (must match 20_/22_/27_cone) ─────────────
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/catalog/PanSTARRS/COSMOS')

# ── PS1 DR2 MeanObjectView columns (subset matching cone-search) ────
COLUMNS = [
    # identifiers + flags
    'm.objID', 'm.objInfoFlag', 'm.qualityFlag',
    # position
    'm.raMean', 'm.decMean', 'm.raMeanErr', 'm.decMeanErr',
    # per-band detection counts
    'm.nDetections', 'm.ng', 'm.nr', 'm.ni', 'm.nz', 'm.ny',
    # PSF mean magnitudes + errors
    'm.gMeanPSFMag', 'm.gMeanPSFMagErr',
    'm.rMeanPSFMag', 'm.rMeanPSFMagErr',
    'm.iMeanPSFMag', 'm.iMeanPSFMagErr',
    'm.zMeanPSFMag', 'm.zMeanPSFMagErr',
    'm.yMeanPSFMag', 'm.yMeanPSFMagErr',
    # Kron mean magnitudes + errors (better for extended sources)
    'm.gMeanKronMag', 'm.gMeanKronMagErr',
    'm.rMeanKronMag', 'm.rMeanKronMagErr',
    'm.iMeanKronMag', 'm.iMeanKronMagErr',
    'm.zMeanKronMag', 'm.zMeanKronMagErr',
    'm.yMeanKronMag', 'm.yMeanKronMagErr',
]

SQL = f"""
SELECT
  {','.join(COLUMNS)}
FROM MeanObjectView m
WHERE m.raMean  BETWEEN {RA_MIN}  AND {RA_MAX}
  AND m.decMean BETWEEN {DEC_MIN} AND {DEC_MAX}
  AND m.nDetections >= 1
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true',
                   help='Print params + SQL, do not contact CasJobs.')
    p.add_argument('--context', default='PanSTARRS_DR2',
                   help='CasJobs context (default PanSTARRS_DR2).')
    p.add_argument('--task-name', default='ps1_cosmos_vis_bbox',
                   help='Job name shown in the CasJobs queue.')
    p.add_argument('--poll-seconds', type=int, default=5,
                   help='Poll interval while waiting for the job (s).')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / 'ps1_cosmos'

    print('PS1 DR2 CasJobs query:')
    print(f'  context      : {args.context}')
    print(f'  bbox         : RA  {RA_MIN}..{RA_MAX}   Dec {DEC_MIN}..{DEC_MAX}')
    print(f'  columns      : {len(COLUMNS)}')
    print(f'  + Euclid VIS polygon post-filter (3.91 deg²)')
    print()
    print('-' * 60)
    print(SQL)
    print('-' * 60)
    print()

    (base.with_suffix('.query.sql')).write_text(SQL + '\n')

    if args.dry_run:
        print('[dry-run] SQL saved, not contacting CasJobs.')
        return

    # ── Credentials check (env vars from ~/.zshenv) ────────────────────
    wsid = os.environ.get('CASJOBS_WSID')
    pw   = os.environ.get('CASJOBS_PW')
    if not wsid or not pw:
        sys.exit('ERROR: $CASJOBS_WSID and/or $CASJOBS_PW not in env.\n'
                 '       Add exports to ~/.zshenv (see script docstring).')
    print(f'CasJobs WSID    : {wsid}  (password {len(pw)} chars)')

    import mastcasjobs
    jobs = mastcasjobs.MastCasJobs(userid=wsid, password=pw,
                                   context=args.context)

    # ── Submit job + poll ──────────────────────────────────────────────
    print(f'\nSubmitting job "{args.task_name}" ...', flush=True)
    t0 = time.time()
    jobid = jobs.submit(SQL, task_name=args.task_name)
    print(f'  job_id = {jobid}', flush=True)

    print(f'Polling every {args.poll_seconds}s ...', flush=True)
    status_names = ['ready', 'started', 'cancelling', 'cancelled',
                    'failed', 'finished']
    while True:
        status_code = jobs.status(jobid)
        # mastcasjobs.status returns (int_code, label) on recent versions
        if isinstance(status_code, tuple):
            code, label = status_code
        else:
            code, label = status_code, status_names[status_code] \
                if 0 <= status_code < len(status_names) else str(status_code)
        elapsed = time.time() - t0
        print(f'  [{elapsed:6.1f}s] status = {label} (code {code})',
              flush=True)
        if label in ('finished', 'failed', 'cancelled'):
            break
        time.sleep(args.poll_seconds)

    if label != 'finished':
        sys.exit(f'CasJobs job ended with status {label}.')

    # ── Pull the results table ─────────────────────────────────────────
    print('\nDownloading result table ...', flush=True)
    # quick_table() returns the table from MyDB after a finished job.
    # The output table name defaults to the task_name.
    try:
        tbl = jobs.fast_table(args.task_name)
    except Exception:
        # fast_table needs a CSV-friendly output; fall back to get_table
        tbl = jobs.get_table(args.task_name)
    n_raw = len(tbl)
    print(f'  retrieved {n_raw:,} rows')

    # Drop the table from MyDB so it doesn't accumulate
    try:
        jobs.drop_table_if_exists(args.task_name)
        print(f'  dropped MyDB.{args.task_name}')
    except Exception as e:
        print(f'  (drop_table failed, ignoring: {e!r})')

    dt = time.time() - t0
    print(f'\nWall time: {dt:.1f}s  ({dt/60:.1f} min)')

    import pandas as pd
    df = tbl.to_pandas() if hasattr(tbl, 'to_pandas') else pd.DataFrame(tbl)

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
        'method':        'CasJobs PanSTARRS_DR2 box query (MeanObjectView)',
        'casjobs_jobid': int(jobid) if str(jobid).isdigit() else str(jobid),
        'row_count_bbox':           int(n_bbox),
        'row_count_in_vis_polygon': int(len(df)),
        'bbox_deg': {
            'ra_min':  RA_MIN,  'ra_max':  RA_MAX,
            'dec_min': DEC_MIN, 'dec_max': DEC_MAX,
        },
        'vis_polygon_deg2':  round(vis_polygon.area, 4),
        'spatial_selection': 'CasJobs SQL bbox + Shapely VIS polygon',
        'ps1_table':         'DR2 MeanObjectView',
        'columns':           list(df.columns),
        'parquet_path':      str(parquet_path),
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
            n22 = ((df[col] > 0) & (df[col] < 22)).sum()
            n20 = ((df[col] > 0) & (df[col] < 20)).sum()
            print(f'  {b}<22 / {b}<20         : {n22:>8,} / {n20:>8,}')
    if 'nDetections' in df.columns:
        print(f'  nDetections >= 5    : {(df["nDetections"] >= 5).sum():>8,}')


if __name__ == '__main__':
    main()
