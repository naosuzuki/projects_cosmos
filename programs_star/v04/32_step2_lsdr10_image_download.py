#!/usr/bin/env python
"""
32_step2_lsdr10_image_download.py — Step 2.4c: download LS DR10
DECaLS coadd imaging (image + invvar + maskbits) over the Euclid VIS
polygon or the 3-way subset.

Why brick-level coadds and not per-source cutouts:
  LS DR10 packages all imaging at the brick level — each brick is
  ~0.27° × 0.27° (3600 × 3600 pixels at 0.262"/pix).  Tractor needs
  the full brick to re-derive any photometry, and DECaLS PSF models
  are also brick-level products.  Per-source cutouts of 1M+ sources
  would be ~250 GB of tiny files; ~92 bricks for the full VIS scope
  is ~6-7 GB — clearly the right granularity.

Brick discovery:
  Every row in our LS DR10 catalog (27_/30_) carries a `brickname`
  column.  We harvest the unique values for the requested scope and
  download whatever's missing.

URL pattern (LS DR10 south = DECaLS):
  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/
    {brickname[:3]}/{brickname}/legacysurvey-{brickname}-{type}.fits.fz

Per-brick file types we fetch by default:
  image-g, image-r, image-i, image-z   — coadd science images
  invvar-g, invvar-r, invvar-i, invvar-z — inverse-variance maps
  maskbits                              — bit-packed mask

Defaults:
  --scope 3way   (default — 18 bricks × 9 files = 162 files, ~1.5 GB)
  --scope vis    (full Euclid VIS coverage; 92 bricks × 9 = 828 files,
                  ~7-8 GB)

Storage (mirrors the DR10 directory tree):
  /Volumes/exdisk1/data/DESI_Legacy/COSMOS/dr10/south/coadd/{brick[:3]}/{brickname}/
    legacysurvey-{brickname}-{type}.fits.fz
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

# ── Paths ──────────────────────────────────────────────────────────
# Catalog stays at /LSDR10/COSMOS where the 27_ scripts put it.
CATALOG_DIR = Path('/Volumes/exdisk1/data/catalog/DESI_Legacy/COSMOS')
# User-requested location for imaging:
DESI_DIR    = Path('/Volumes/exdisk1/data/DESI_Legacy/COSMOS')
COADD_ROOT  = DESI_DIR / 'dr10' / 'south' / 'coadd'

ARCHIVE_BASE = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10'

BANDS_DEFAULT = ['g', 'r', 'i', 'z']
TYPES_DEFAULT = ['image', 'invvar', 'maskbits']
# Available per-band types: image, invvar, chi2, nexp, psfsize, depth,
# galdepth, model, blobs. 'maskbits' has no band suffix.
PER_BAND_TYPES = {'image', 'invvar', 'chi2', 'nexp', 'psfsize',
                  'depth', 'galdepth', 'model'}
NO_BAND_TYPES  = {'maskbits', 'blobs'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scope', choices=['vis', '3way'], default='3way',
                   help='3way (default ~1.5 GB) or vis (~7 GB).')
    p.add_argument('--bands', nargs='+', default=BANDS_DEFAULT,
                   choices=BANDS_DEFAULT,
                   help='LS DR10 bands to fetch (default grizi).')
    p.add_argument('--types', nargs='+', default=TYPES_DEFAULT,
                   choices=sorted(PER_BAND_TYPES | NO_BAND_TYPES),
                   help='Image types per brick (default image+invvar+maskbits).')
    p.add_argument('--jobs', '-j', type=int, default=4,
                   help='aria2c concurrent downloads (default 4).')
    p.add_argument('--conn', '-x', type=int, default=4,
                   help='aria2c connections per server (default 4).')
    p.add_argument('--dry-run', action='store_true',
                   help='List unique bricks + size estimate; do not download.')
    return p.parse_args()


def brick_dir(brickname: str) -> Path:
    """Local destination directory for a brick (mirrors DR10 layout)."""
    return COADD_ROOT / brickname[:3] / brickname


def brick_files(brickname: str, bands: list[str],
                types: list[str]) -> list[tuple[str, str]]:
    """Return list of (url, rel_path) for all requested per-brick files."""
    out = []
    sub = f'south/coadd/{brickname[:3]}/{brickname}'
    for t in types:
        if t in PER_BAND_TYPES:
            for b in bands:
                fn = f'legacysurvey-{brickname}-{t}-{b}.fits.fz'
                out.append((f'{ARCHIVE_BASE}/{sub}/{fn}',
                            f'{sub}/{fn}'))
        elif t in NO_BAND_TYPES:
            fn = f'legacysurvey-{brickname}-{t}.fits.fz'
            out.append((f'{ARCHIVE_BASE}/{sub}/{fn}',
                        f'{sub}/{fn}'))
        else:
            print(f'  WARNING: unknown image type {t!r}; skipping')
    return out


def main():
    args = parse_args()

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not in PATH.  brew install aria2')

    cat_path = CATALOG_DIR / (f'lsdr10_{args.scope}.parquet'
                              if args.scope == '3way'
                              else 'lsdr10_cosmos.parquet')
    if not cat_path.exists():
        sys.exit(f'{cat_path} not found — run 30_/31_ first.')

    print(f'Scope         : {args.scope}')
    print(f'Bands         : {" ".join(args.bands)}')
    print(f'Types         : {" ".join(args.types)}')
    print(f'Catalog       : {cat_path}')
    print(f'Image root    : {COADD_ROOT}')
    print()

    df = pd.read_parquet(cat_path)
    bricks = sorted(df['brickname'].astype(str).str.strip().unique())
    print(f'Unique bricks : {len(bricks):,}')
    print(f'  (e.g. {bricks[:3]} … {bricks[-3:]})')
    print()

    # ── Build URL list ────────────────────────────────────────────────
    DESI_DIR.mkdir(parents=True, exist_ok=True)
    COADD_ROOT.mkdir(parents=True, exist_ok=True)
    have = miss = 0
    file_records = []
    for brick in bricks:
        brick_dir(brick).mkdir(parents=True, exist_ok=True)
        for url, rel in brick_files(brick, args.bands, args.types):
            dest = DESI_DIR / 'dr10' / rel
            file_records.append({'brick': brick, 'url': url, 'rel': rel})
            if dest.exists() and dest.stat().st_size > 0:
                have += 1
            else:
                miss += 1
    expected = len(file_records)
    print(f'Files expected   : {expected:,}  '
          f'({len(bricks)} bricks × {expected//max(len(bricks),1)}/brick)')
    print(f'Already on disk  : {have:,}')
    print(f'To download      : {miss:,}')
    # Rough size estimates from the brick we probed (1496p020 image-r = 9.7 MB)
    # image ~10 MB, invvar ~6 MB, maskbits ~3 MB.  Weighted average ~7 MB.
    est_gb = miss * 0.008
    print(f'Estimated size   : ~{est_gb:.1f} GB')
    print()

    if args.dry_run or miss == 0:
        print(f'[{"dry-run" if args.dry_run else "all-present"}] '
              f'not invoking aria2c.')
        return

    url_file = COADD_ROOT.parent / f'_aria2_urls_{args.scope}.txt'
    lines = []
    for rec in file_records:
        dest = DESI_DIR / 'dr10' / rec['rel']
        if dest.exists() and dest.stat().st_size > 0:
            continue
        lines.append(rec['url'])
        lines.append(f'  out=south/coadd/{rec["rel"].split("south/coadd/", 1)[1]}')
    url_file.write_text('\n'.join(lines) + '\n')
    log_file = COADD_ROOT.parent / f'_aria2_{args.scope}.log'

    cmd = [
        'aria2c',
        f'--input-file={url_file}',
        f'--dir={DESI_DIR / "dr10"}',   # not COADD_ROOT.parent — that would prepend an extra "south/" since each out= line already starts with "south/coadd/..."
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
    n_on_disk = 0; total = 0
    for rec in file_records:
        dest = DESI_DIR / 'dr10' / rec['rel']
        if dest.exists():
            n_on_disk += 1
            total += dest.stat().st_size
    print(f'\naria2c exit code: {rc}   wall: {dt/60:.1f} min')
    print(f'files on disk : {n_on_disk:>5}/{len(file_records)}')
    print(f'total size     : {total/1e9:.2f} GB')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'image_release':   'LS DR10 south (DECaLS) coadd bricks',
        'scope':           args.scope,
        'bands':           args.bands,
        'types':           args.types,
        'catalog_source':  str(cat_path),
        'unique_bricks':   int(len(bricks)),
        'files_expected':  int(len(file_records)),
        'files_on_disk':   int(n_on_disk),
        'total_bytes':     int(total),
        'aria2_exit_code': int(rc),
        'wall_seconds':    round(dt, 2),
    }
    meta_path = DESI_DIR / f'lsdr10_images_{args.scope}.meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[save] {meta_path.name}')

    if n_on_disk == len(file_records):
        print('All requested brick files present.  Done.')
    else:
        print('Some files missing — re-run to resume.')


if __name__ == '__main__':
    main()
