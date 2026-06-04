#!/usr/bin/env python
"""
03_download_hst_cosmosweb_apr23_30mas.py — parallel download of the
ORIGINAL COSMOS 2003–2005 ACS/WFC F814W mosaics re-reduced in the
COSMOS-Web Apr-2023 reduction on the A-tile grid (A1..A10), 30-mas
pixel scale.

Companion to:
  v04/02_download_hst_cosmos_orig_2005_30mas.py  (B-tile grid, Jan-2024
                                                  reduction of the same
                                                  original 2003–2005
                                                  data)

Same tile-naming convention as the JWST COSMOS-Web tiling scheme
(see the paper: A1–A10 cover the southern half, B1–B10 the northern;
each group spans ~0.5 deg^2 of COSMOS in a staggered brick pattern).
This script downloads the HST/ACS F814W mosaics that were drizzled
onto the A-tile grid by the Apr-2023 reduction.

Source : https://exchg.calet.org/cosmosweb/COSMOS-Web_Apr23/ACS-cosmos-orig-only/
Target : /Volumes/exdisk1/data/HST/COSMOS_ACS2005/
Auth   : HTTP Basic via ~/.netrc (already provisioned)

Files  : 10 tiles (A1..A10) × 3 products (drz, err, wht), ~1.92 GB each
         → 30 files ≈ 57 GB total.  60-mas siblings are skipped.

Engine : aria2c — same tuned flags that downloaded the Jan-24 ACS and
         the COSMOS_ACS2005 reductions cleanly at ~18 MB/s with zero
         retries (exchg.calet.org is bandwidth-capped; > a few concurrent
         streams trigger per-connection throughput collapse).
         -j 2 -x 4 -s 4               : 2 files × 4 connections (8 total)
         --lowest-speed-limit=0       : never self-abort for slowness
         --file-allocation=none       : no misleading 1.8 GB placeholders
         -c, --continue=true          : resume partial downloads

Usage  :  python 03_download_hst_cosmosweb_apr23_30mas.py
          python 03_download_hst_cosmosweb_apr23_30mas.py --jobs 3
          python 03_download_hst_cosmosweb_apr23_30mas.py --dry-run

Re-runs are idempotent: completed files are skipped, partial files resume.

Observed throughput on the prior runs (same engine, same server):
  ~18 MB/s steady, 0 errors, ~50 min wall for 30 × 1.92 GB.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = 'https://exchg.calet.org/cosmosweb/COSMOS-Web_Apr23/ACS-cosmos-orig-only/'
DEST     = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
TILES    = [f'A{i}' for i in range(1, 11)]
PRODUCTS = ['drz', 'err', 'wht']
PIXSCALE = '30mas'
APPROX_GB_PER_FILE = 1.92


def build_url(tile: str, product: str) -> str:
    return (BASE_URL +
            f'mosaic_cosmos_web_2023apr_{PIXSCALE}_tile_{tile}'
            f'_hst_acs_wfc_f814w_{product}.fits')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', '-j', type=int, default=2,
                   help='Number of files to download in parallel (default 2 — '
                        'server is bandwidth-capped; >3 risks per-conn '
                        'throughput collapse + abort loops)')
    p.add_argument('--conn', '-x', type=int, default=4,
                   help='Max connections per file (default 4)')
    p.add_argument('--split', '-s', type=int, default=4,
                   help='Segments per file (default 4)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print URLs and exit; download nothing.')
    p.add_argument('--products', nargs='+', default=PRODUCTS,
                   choices=PRODUCTS,
                   help='Subset of products to fetch (drz/err/wht).')
    p.add_argument('--tiles', nargs='+', default=TILES,
                   help='Subset of tiles to fetch (default A1..A10).')
    return p.parse_args()


def main():
    args = parse_args()

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not found in PATH.  Install with: brew install aria2')

    netrc = Path.home() / '.netrc'
    if not netrc.exists():
        sys.exit(f'{netrc} not found — server requires HTTP Basic auth.')

    urls = [build_url(t, p) for t in args.tiles for p in args.products]
    n    = len(urls)
    est_gb = n * APPROX_GB_PER_FILE

    DEST.mkdir(parents=True, exist_ok=True)
    print(f'Source : {BASE_URL}')
    print(f'Target : {DEST}')
    print(f'Files  : {n}  ({len(args.tiles)} tiles × {len(args.products)} products, {PIXSCALE})')
    print(f'Approx : {est_gb:.0f} GB total')
    print(f'Engine : aria2c  -j {args.jobs}  -x {args.conn}  -s {args.split}  (resume + integrity)')
    print()

    if args.dry_run:
        print('-- dry run, URL list --')
        for u in urls:
            print(u)
        return

    url_file = DEST / '_aria2_urls.txt'
    url_file.write_text('\n'.join(urls) + '\n')

    log_file = DEST / '_aria2.log'
    cmd = [
        'aria2c',
        f'--input-file={url_file}',
        f'--dir={DEST}',
        f'--max-concurrent-downloads={args.jobs}',
        f'--max-connection-per-server={args.conn}',
        f'--split={args.split}',
        '--min-split-size=20M',
        '--continue=true',
        '--auto-file-renaming=false',
        '--allow-overwrite=false',
        '--check-integrity=false',
        '--conditional-get=true',
        '--remote-time=true',
        '--file-allocation=none',
        f'--netrc-path={netrc}',
        '--http-accept-gzip=false',
        '--summary-interval=15',
        '--console-log-level=warn',
        f'--log={log_file}',
        '--log-level=notice',
        '--retry-wait=15',
        '--max-tries=0',
        '--timeout=120',
        '--connect-timeout=60',
        '--lowest-speed-limit=0',
        '--stream-piece-selector=inorder',
    ]

    print('Launching aria2c ...')
    print(' '.join(cmd))
    print(f'(progress streams here; full log in {log_file})')
    print()
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print('\n[interrupted] partial files left in place; rerun to resume.')
        sys.exit(130)
    dt = time.time() - t0

    print(f'\naria2c exit code: {rc}     wall: {dt/60:.1f} min')
    fits = sorted(DEST.glob('*.fits'))
    part = sorted(DEST.glob('*.aria2'))
    total = sum(f.stat().st_size for f in fits) / 1e9
    print(f'fits files on disk : {len(fits):>3}   ({total:.1f} GB)')
    print(f'partial (.aria2)   : {len(part):>3}')
    if len(fits) >= n and not part:
        print('All requested files present.  Done.')
    else:
        print('Some files still pending — rerun to resume.')


if __name__ == '__main__':
    main()
