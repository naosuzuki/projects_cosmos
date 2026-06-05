#!/usr/bin/env python
"""
26_step2_sdss_image_uncompress.py — decompress the SDSS frame*.fits.bz2
files downloaded by 25_step2_sdss_image_download.py into plain FITS
in-place (alongside the .bz2 originals).

Parallel decompression across CPU cores (default 8 workers).  Skips
files whose .fits already exists so the script is idempotent / safe
to re-run.  Originals are kept by default; pass --delete-bz2 to remove
them after verifying the uncompressed file is non-empty.

Inputs:
  /Volumes/exdisk1/data/SDSS/COSMOS/frames/**/frame-*.fits.bz2

Outputs (alongside each input):
  frame-{band}-{run:06d}-{camcol}-{field:04d}.fits

Expected after run:
  790 .fits.bz2 originals (2.74 GB) + 790 .fits uncompressed (~9.3 GB).
"""
from __future__ import annotations
import argparse
import bz2
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path('/Volumes/exdisk1/data/SDSS/COSMOS/frames')


def decompress_one(src_str: str, delete_bz2: bool = False
                   ) -> tuple[str, str, int | str]:
    """One worker — decompress src.fits.bz2 → src.fits; optionally
    delete the original.  Returns (status, basename, bytes_or_error)."""
    src = Path(src_str)
    dst = src.with_suffix('')             # strip .bz2 suffix
    if dst.exists() and dst.stat().st_size > 0:
        return ('skip', src.name, dst.stat().st_size)
    try:
        with bz2.open(src, 'rb') as fin, open(dst, 'wb') as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)   # 1 MiB chunks
        size = dst.stat().st_size
        if size == 0:
            raise RuntimeError('decompressed file is empty')
        if delete_bz2:
            src.unlink()
        return ('ok', src.name, size)
    except Exception as e:
        # leave any partial dst behind for inspection
        return ('err', src.name, repr(e))


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', '-j', type=int, default=8,
                   help='Parallel worker processes (default 8).')
    p.add_argument('--delete-bz2', action='store_true',
                   help='Remove the .bz2 original after successful '
                        'decompression (default: keep both).')
    args = p.parse_args()

    files = sorted(ROOT.rglob('frame-*.fits.bz2'))
    print(f'Scan       : {ROOT}')
    print(f'Found      : {len(files):,} .fits.bz2 files')
    print(f'Workers    : {args.workers}')
    print(f'Delete bz2 : {args.delete_bz2}')
    print()

    if not files:
        print('Nothing to do.')
        return

    n_ok = n_skip = n_err = 0
    bytes_total = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decompress_one, str(f), args.delete_bz2): f
                   for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            status, name, info = fut.result()
            if status == 'ok':
                n_ok += 1
                bytes_total += int(info)
            elif status == 'skip':
                n_skip += 1
                bytes_total += int(info)
            else:
                n_err += 1
                print(f'  err {name}: {info}', file=sys.stderr)
            if i % 100 == 0 or i == len(files):
                dt = time.time() - t0
                rate = bytes_total / 1e9 / max(dt, 0.01)
                print(f'  [{i:>4}/{len(files)}]  ok={n_ok:>4} '
                      f'skip={n_skip:>4} err={n_err}  '
                      f'{bytes_total/1e9:.2f} GB  ({rate:.1f} GB/s)',
                      flush=True)

    dt = time.time() - t0
    print()
    print(f'Wall time    : {dt:.1f} s')
    print(f'Decompressed : {n_ok} new, {n_skip} already present, {n_err} errors')
    print(f'Total size   : {bytes_total/1e9:.2f} GB uncompressed')


if __name__ == '__main__':
    main()
