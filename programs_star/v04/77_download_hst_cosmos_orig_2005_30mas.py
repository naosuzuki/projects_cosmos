#!/usr/bin/env python
"""
77_download_hst_cosmos_orig_2005_30mas.py — parallel download of the
ORIGINAL COSMOS 2003–2005 ACS/WFC F814W mosaics, re-reduced in the
COSMOS-Web Jan-2024 reduction, 30-mas pixel scale.

These are the 2005-epoch HST imaging used as a LONG-BASELINE reference
for proper motion against the contemporaneous (Jan-2024-reduced)
COSMOS-Web ACS/WFC mosaics already in COSMOS_ACS2024/.  Tile grid
(B1..B10) is identical between the two reductions, which makes
pairwise PM measurement per-tile-trivial: identical pixel grids,
identical WCS, ~19-year baseline.

Source : https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/ACS-cosmos-orig-only/
Target : /Volumes/exdisk1/data/HST/COSMOS_ACS2005/
Auth   : HTTP Basic via ~/.netrc (already provisioned)

Files  : 10 tiles (B1..B10) × 3 products (drz, err, wht), ~1.92 GB each
         → 30 files ≈ 57 GB total.  60-mas siblings are skipped.

Engine : aria2c — TUNED FOR THIS HOST (exchg.calet.org is bandwidth-capped;
         hammering with 64 concurrent streams triggers per-connection
         throughput < 100 KB/s, which aria2's lowest-speed-limit then
         aborts in an infinite retry loop.  Conservative settings here.)
         -j 2 -x 4 -s 4               : 2 files × 4 connections (8 total)
         --lowest-speed-limit=0       : never self-abort for slowness
         --file-allocation=none       : no misleading 1.8 GB placeholders
         -c, --continue=true          : resume partial downloads

Usage  :  python 77_download_hst_cosmos_orig_2005_30mas.py
          python 77_download_hst_cosmos_orig_2005_30mas.py --jobs 3
          python 77_download_hst_cosmos_orig_2005_30mas.py --dry-run

Re-runs are idempotent: completed files are skipped, partial files resume.

Observed throughput on the COSMOS_ACS2024 sister run (same engine,
same server): 18 MB/s steady, 0 errors, ~50 minutes wall.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = 'https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/ACS-cosmos-orig-only/'
DEST     = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
TILES    = [f'B{i}' for i in range(1, 11)]
PRODUCTS = ['drz', 'err', 'wht']
PIXSCALE = '30mas'
APPROX_GB_PER_FILE = 1.92


def build_url(tile: str, product: str) -> str:
    return (BASE_URL +
            f'mosaic_cosmos_web_2024jan_{PIXSCALE}_tile_{tile}'
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
                   help='Subset of tiles to fetch (default B1..B10).')
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

    # Write URL list to a temp file inside the dest dir (visible / restartable).
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
        '--check-integrity=false',     # server doesn't publish checksums
        '--conditional-get=true',      # Last-Modified-based skip
        '--remote-time=true',          # preserve mtimes
        '--file-allocation=none',      # no misleading 1.8 GB placeholders
        f'--netrc-path={netrc}',
        '--http-accept-gzip=false',
        '--summary-interval=15',
        '--console-log-level=warn',    # less spam on the terminal
        f'--log={log_file}',
        '--log-level=notice',
        '--retry-wait=15',             # polite back-off
        '--max-tries=0',               # 0 = infinite; only stop on terminal err
        '--timeout=120',
        '--connect-timeout=60',
        '--lowest-speed-limit=0',      # never self-abort on slow throughput
        '--stream-piece-selector=inorder',  # sequential = nicer to server
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

    # Report final state
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
