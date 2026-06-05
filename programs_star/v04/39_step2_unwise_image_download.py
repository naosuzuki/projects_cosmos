#!/usr/bin/env python
"""
39_step2_unwise_image_download.py — Step 2.6c: download unWISE DR1
coadd images (W1 + W2 intensity + inverse-variance + frame-count +
std-dev) for tiles covering the Euclid VIS polygon.

unWISE coadds are 2048×2048 px @ 2.75 arcsec/pix = 1.564° × 1.564°
per tile.  Five tiles cover the COSMOS Euclid VIS area; four cover
the 3-way subset.

Tile discovery: harvest unique `coadd_id` values from the catalog
36_step2_unwise_pull.py landed.  Each tile has 4 file types per band:
  unwise-<TILE>-w<BAND>-img-m.fits     intensity (counts/s, AB-zp 22.5)
  unwise-<TILE>-w<BAND>-invvar-m.fits  inverse-variance
  unwise-<TILE>-w<BAND>-n-m.fits       number of contributing frames
  unwise-<TILE>-w<BAND>-std-m.fits     std-dev across frames

URL pattern (Schlafly+ 2019 NEO7 coadds, hosted at unwise.me):
  https://unwise.me/data/neo7/unwise-coadds/fulldepth/<TILE[:3]>/<TILE>/unwise-<TILE>-w<BAND>-<TYPE>-m.fits

Defaults: scope=vis (5 tiles × 2 bands × 4 types = 40 files, ~1.5 GB).

Storage:
  /Volumes/exdisk1/data/unWISE/COSMOS/neo7/<TILE[:3]>/<TILE>/
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

CATALOG_DIR = Path('/Volumes/exdisk1/data/catalog/unWISE/COSMOS')
UW_DIR      = Path('/Volumes/exdisk1/data/unWISE/COSMOS')
TILES_ROOT  = UW_DIR / 'neo7'

ARCHIVE_BASE = 'https://unwise.me/data/neo7/unwise-coadds/fulldepth'
BANDS_DEFAULT = ['w1', 'w2']
TYPES_DEFAULT = ['img', 'invvar', 'n', 'std']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scope', choices=['vis', '3way'], default='vis',
                   help='VIS (default, 5 tiles ~1.5 GB) or 3way (4 tiles ~1.2 GB).')
    p.add_argument('--bands', nargs='+', default=BANDS_DEFAULT,
                   choices=BANDS_DEFAULT)
    p.add_argument('--types', nargs='+', default=TYPES_DEFAULT,
                   choices=TYPES_DEFAULT)
    p.add_argument('--jobs', '-j', type=int, default=4)
    p.add_argument('--conn', '-x', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if shutil.which('aria2c') is None:
        sys.exit('aria2c not in PATH.  brew install aria2')

    cat_path = CATALOG_DIR / (f'unwise_{args.scope}.parquet'
                              if args.scope == '3way'
                              else 'unwise_cosmos.parquet')
    if not cat_path.exists():
        sys.exit(f'{cat_path} not found — run 36_/37_ first.')

    print(f'Scope         : {args.scope}')
    print(f'Bands         : {" ".join(args.bands)}')
    print(f'Types         : {" ".join(args.types)}')
    print(f'Catalog       : {cat_path}')
    print(f'Image root    : {TILES_ROOT}')
    print()

    df = pd.read_parquet(cat_path)
    tiles = sorted(df['coadd_id'].astype(str).str.strip().unique())
    print(f'Unique tiles  : {len(tiles)}  {tiles}')
    print()

    UW_DIR.mkdir(parents=True, exist_ok=True)
    TILES_ROOT.mkdir(parents=True, exist_ok=True)

    # NEO7 hosts:  img-m.fits (uncompressed); invvar/n/std are .fits.gz
    have = miss = 0; file_records = []
    for tile in tiles:
        td = TILES_ROOT / tile[:3] / tile
        td.mkdir(parents=True, exist_ok=True)
        for b in args.bands:
            for t in args.types:
                ext = '.fits' if t == 'img' else '.fits.gz'
                fn  = f'unwise-{tile}-{b}-{t}-m{ext}'
                url = f'{ARCHIVE_BASE}/{tile[:3]}/{tile}/{fn}'
                rel = f'{tile[:3]}/{tile}/{fn}'
                dest = TILES_ROOT / rel
                file_records.append({'tile': tile, 'url': url, 'rel': rel})
                if dest.exists() and dest.stat().st_size > 0:
                    have += 1
                else:
                    miss += 1

    expected = len(file_records)
    print(f'Files expected   : {expected:,}  '
          f'({len(tiles)} tiles × {len(args.bands)} bands × {len(args.types)} types)')
    print(f'Already on disk  : {have:,}')
    print(f'To download      : {miss:,}')
    # img ~50 MB, invvar ~50 MB, n ~5 MB, std ~50 MB.  Avg ~40 MB.
    est_gb = miss * 0.04
    print(f'Estimated size   : ~{est_gb:.1f} GB')
    print()

    if args.dry_run or miss == 0:
        print(f'[{"dry-run" if args.dry_run else "all-present"}] not invoking aria2c.')
        return

    url_file = TILES_ROOT / f'_aria2_urls_{args.scope}.txt'
    lines = []
    for rec in file_records:
        if (TILES_ROOT / rec['rel']).exists():
            continue
        lines.append(rec['url'])
        lines.append(f'  out={rec["rel"]}')
    url_file.write_text('\n'.join(lines) + '\n')
    log_file = TILES_ROOT / f'_aria2_{args.scope}.log'

    cmd = ['aria2c',
           f'--input-file={url_file}',
           f'--dir={TILES_ROOT}',
           f'--max-concurrent-downloads={args.jobs}',
           f'--max-connection-per-server={args.conn}',
           f'--split={args.conn}',
           '--min-split-size=1M',
           '--continue=true',
           '--auto-file-renaming=false',
           '--allow-overwrite=false',
           '--check-integrity=false',
           '--conditional-get=true',
           '--remote-time=true',
           '--file-allocation=none',
           '--summary-interval=15',
           '--console-log-level=warn',
           f'--log={log_file}',
           '--log-level=notice',
           '--retry-wait=5',
           '--max-tries=10',
           '--timeout=60',
           '--connect-timeout=30',
           '--lowest-speed-limit=0']
    print('Launching aria2c ...')
    print(' '.join(cmd[:6]) + ' ...')
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        sys.exit(130)
    dt = time.time() - t0

    n_on_disk = 0; total = 0
    for rec in file_records:
        d = TILES_ROOT / rec['rel']
        if d.exists():
            n_on_disk += 1; total += d.stat().st_size
    print(f'\naria2c exit code: {rc}   wall: {dt/60:.1f} min')
    print(f'files on disk : {n_on_disk}/{len(file_records)}')
    print(f'total size     : {total/1e9:.2f} GB')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'image_release':   'unWISE NEO7 fulldepth coadds (Schlafly+ 2019)',
        'scope':           args.scope,
        'bands':           args.bands,
        'types':           args.types,
        'catalog_source':  str(cat_path),
        'unique_tiles':    int(len(tiles)),
        'files_expected':  int(len(file_records)),
        'files_on_disk':   int(n_on_disk),
        'total_bytes':     int(total),
        'aria2_exit_code': int(rc),
        'wall_seconds':    round(dt, 2),
    }
    (UW_DIR / f'unwise_images_{args.scope}.meta.json').write_text(
        json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
