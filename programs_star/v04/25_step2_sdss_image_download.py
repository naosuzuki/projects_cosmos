#!/usr/bin/env python
"""
25_step2_sdss_image_download.py — Step 2.2c: download SDSS DR17
imaging (frame fits.bz2) over the Euclid VIS polygon (or the 3-way
subset).

Per the locked Q4 decision (catalog + imaging hybrid, augmented with
PSFEx/DAOPHOT in Step 3) we want our own photometry pass on SDSS
imaging, not just the published catalog magnitudes.  The unique
(run, rerun, camcol, field) tuples per source are already in our
sdss_cosmos.parquet — this script harvests the URLs from that table
and downloads each frame via aria2c.

Each frame:
  https://data.sdss.org/sas/dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{filter}-{run:06d}-{camcol}-{field:04d}.fits.bz2

Storage (mirrors SDSS DAS hierarchy):
  /Volumes/exdisk1/data/SDSS/COSMOS/frames/{rerun}/{run}/{camcol}/frame-{filter}-...fits.bz2

Defaults:
  --scope vis  (default)  158 fields × 5 bands = 790 frames, ~3.2 GB bz2
  --scope 3way            32 fields × 5 bands = 160 frames, ~640 MB bz2

Files remain bz2-compressed for now — a separate Step 3 helper script
will decompress on demand when SExtractor/PSFEx needs the FITS.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SDSS_DIR    = Path('/Volumes/exdisk1/data/SDSS/COSMOS')
FRAMES_ROOT = SDSS_DIR / 'frames'
BASE_URL    = 'https://data.sdss.org/sas/dr17/eboss/photoObj/frames'
BANDS       = ['u', 'g', 'r', 'i', 'z']


def frame_url(rerun: int, run: int, camcol: int, field: int, band: str) -> str:
    return (f'{BASE_URL}/{rerun}/{run}/{camcol}/'
            f'frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2')


def frame_dest(rerun: int, run: int, camcol: int, field: int, band: str) -> Path:
    return (FRAMES_ROOT / str(rerun) / str(run) / str(camcol) /
            f'frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scope', choices=['vis', '3way'], default='vis',
                   help='Which catalog to harvest fields from.')
    p.add_argument('--bands', nargs='+', default=BANDS, choices=BANDS,
                   help='Which SDSS bands to fetch (default ugriz).')
    p.add_argument('--jobs', '-j', type=int, default=4,
                   help='aria2c concurrent downloads (default 4).')
    p.add_argument('--conn', '-x', type=int, default=4,
                   help='aria2c connections per server (default 4).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print URL list, do not download.')
    return p.parse_args()


def main():
    args = parse_args()

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not in PATH.  brew install aria2')

    cat = SDSS_DIR / ('sdss_cosmos.parquet' if args.scope == 'vis'
                      else 'sdss_3way.parquet')
    if not cat.exists():
        sys.exit(f'{cat} not found — run 22_step2_sdss_pull.py first.')

    print(f'Source catalog : {cat}')
    df = pd.read_parquet(cat)
    fields = (df[['rerun', 'run', 'camcol', 'field']]
                .astype(int)
                .drop_duplicates()
                .sort_values(['rerun', 'run', 'camcol', 'field'])
                .reset_index(drop=True))
    print(f'  unique imaging fields: {len(fields):,}')

    urls = []
    for _, r in fields.iterrows():
        for band in args.bands:
            urls.append(frame_url(int(r.rerun), int(r.run),
                                   int(r.camcol), int(r.field), band))
    n = len(urls)
    approx_gb = n * 4.0 / 1024
    print(f'  total frames to fetch: {n:,}  ({len(args.bands)} bands)')
    print(f'  approx size (4 MB/frame avg): {approx_gb:.2f} GB compressed')
    print(f'  target dir: {FRAMES_ROOT}')
    print()

    if args.dry_run:
        print('-- dry-run, first 10 URLs --')
        for u in urls[:10]:
            print(u)
        print('...' if n > 10 else '')
        return

    # ensure all per-(rerun/run/camcol) destination dirs exist BEFORE aria2c
    # writes to them
    for _, r in fields.iterrows():
        (FRAMES_ROOT / str(int(r.rerun)) / str(int(r.run))
                     / str(int(r.camcol))).mkdir(parents=True, exist_ok=True)

    # aria2c needs a "dir" + an "out" per URL.  Use --input-file with the
    # `url\n  out=relative/path/to/file` syntax.
    url_file = FRAMES_ROOT / f'_aria2_urls_{args.scope}.txt'
    lines = []
    for _, r in fields.iterrows():
        for band in args.bands:
            url = frame_url(int(r.rerun), int(r.run),
                             int(r.camcol), int(r.field), band)
            dest = frame_dest(int(r.rerun), int(r.run),
                               int(r.camcol), int(r.field), band)
            rel = dest.relative_to(FRAMES_ROOT)
            lines.append(url)
            lines.append(f'  out={rel}')
    url_file.write_text('\n'.join(lines) + '\n')
    log_file = FRAMES_ROOT / f'_aria2_{args.scope}.log'

    cmd = [
        'aria2c',
        f'--input-file={url_file}',
        f'--dir={FRAMES_ROOT}',
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
        '--lowest-speed-limit=0',
    ]
    print('Launching aria2c ...')
    print(' '.join(cmd[:6]) + '  ...')
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print('\n[interrupted] re-run to resume.')
        sys.exit(130)
    dt = time.time() - t0

    # Summary
    n_have = sum(1 for _, r in fields.iterrows()
                 for band in args.bands
                 if frame_dest(int(r.rerun), int(r.run),
                                int(r.camcol), int(r.field), band).exists())
    total = sum(frame_dest(int(r.rerun), int(r.run),
                            int(r.camcol), int(r.field), band).stat().st_size
                for _, r in fields.iterrows() for band in args.bands
                if frame_dest(int(r.rerun), int(r.run),
                               int(r.camcol), int(r.field), band).exists())
    print(f'\naria2c exit code: {rc}   wall: {dt/60:.1f} min')
    print(f'frames on disk : {n_have:>4}/{n}')
    print(f'total size     : {total/1e9:.2f} GB compressed')
    if n_have == n:
        print('All requested frames present.  Done.')
    else:
        print('Some frames missing — re-run to resume.')


if __name__ == '__main__':
    main()
