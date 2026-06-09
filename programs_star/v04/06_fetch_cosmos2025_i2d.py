#!/usr/bin/env python
"""
06_fetch_cosmos2025_i2d.py — download COMPLETE v1.0 F115W **i2d** mosaics for the
4 patched tiles (A2/A10/B4/B6) from the COSMOS2025 / COSMOS-Web DR1 release at
IAP, decompress to raw FITS, and verify all four extensions.

    page : https://cosmos2025.iap.fr/nircam.html
    base : https://cosmos2025.iap.fr/data/nircam/
           mosaic_nircam_f115w_COSMOS-Web_30mas_<tile>_v1.0_i2d.fits.gz   (~10 GB gz)
    dest : /Volumes/exdisk1/data/JWST/COSMOS_v0.8/   (same dir as the 76 good i2d)

WHY i2d (not the separate extensions): the other 76 tiles are built straight from
their multi-extension i2d (SExtractor reads SCI=ext1 + WHT=ext4 in place).  Using
the complete IAP i2d for these 4 keeps them IDENTICAL to the rest.  The local i2d
for A2/B4/B6 are truncated and A10's is absent; the IAP copies are complete.

Flow (per tile), idempotent / resumable:
  1. raw '..._i2d.fits' already complete (opens with SCI+ERR+CON+WHT)  -> keep.
  2. else download '..._i2d.fits.gz' (aria2c --continue, resumable) and
     stream-decompress to raw (gzip CRC guarantees a complete file).
  3. verify the raw i2d exposes SCI/ERR/CON/WHT, each with full NAXIS data.

Usage:
  python 06_fetch_cosmos2025_i2d.py --tiles A2            # TEST one tile first
  python 06_fetch_cosmos2025_i2d.py                       # all four
  python 06_fetch_cosmos2025_i2d.py --verify-only
  python 06_fetch_cosmos2025_i2d.py --keep-gz             # don't delete the .gz
"""
from __future__ import annotations
import argparse, os, shutil, ssl, subprocess, sys, warnings
from pathlib import Path

BASE = 'https://cosmos2025.iap.fr/data/nircam/'
DEST = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
PIX  = '30mas'
EXTS = ('SCI', 'ERR', 'CON', 'WHT')
DEFAULT_TILES = ['A2', 'A10', 'B4', 'B6']
_CTX = ssl.create_default_context()


def base_name(tile, gz=False):
    return f'mosaic_nircam_f115w_COSMOS-Web_{PIX}_{tile}_v1.0_i2d.fits' + ('.gz' if gz else '')


def verify_i2d(path: Path) -> str:
    """'ok' if the raw i2d opens with all four extensions, each holding data."""
    if not path.exists():
        return 'absent'
    with open(path, 'rb') as f:
        if f.read(6) != b'SIMPLE':
            return 'not-raw-FITS(gzip?)'
    warnings.simplefilter('ignore')
    try:
        from astropy.io import fits
        with fits.open(path) as h:
            have = {(hd.name or '').upper() for hd in h}
            for ext in EXTS:
                if ext not in have:
                    return f'missing-{ext}-ext'
                hd = h[ext]
                if hd.header.get('NAXIS', 0) < 2:
                    return f'{ext}-no-image'
                _ = hd.data[-1, :4]          # touch last row -> raises if short
        return 'ok'
    except Exception as e:
        return f'{type(e).__name__}'


def download_gz(url: str, dest_gz: Path) -> bool:
    if shutil.which('aria2c'):
        cmd = ['aria2c', '-x4', '-s4', '--continue=true', '--auto-file-renaming=false',
               '--allow-overwrite=false', '--summary-interval=30', '--console-log-level=warn',
               f'--dir={dest_gz.parent}', '-o', dest_gz.name, url]
        return subprocess.call(cmd) == 0 and dest_gz.exists()
    import urllib.request
    try:
        req = urllib.request.Request(url); req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=180, context=_CTX) as r, open(dest_gz, 'wb') as f:
            shutil.copyfileobj(r, f, length=64 * 1024 * 1024)
        return True
    except Exception as e:
        print(f'    download FAILED: {type(e).__name__}: {e}')
        return False


def decompress(gz: Path, raw: Path) -> bool:
    """gunzip gz -> raw (atomic).  gzip CRC fails the whole thing if incomplete."""
    tmp = raw.parent / (raw.name + '.tmp')
    print(f'    decompressing -> {raw.name}')
    with open(tmp, 'wb') as out:
        rc = subprocess.call(['gzip', '-dc', str(gz)], stdout=out)
    if rc != 0:
        print(f'    gunzip FAILED (rc={rc}; likely incomplete .gz)')
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, raw)
    return True


def ensure(tile: str) -> str:
    raw = DEST / base_name(tile)
    if verify_i2d(raw) == 'ok':
        return 'already-complete'
    gz = DEST / base_name(tile, gz=True)
    url = BASE + base_name(tile, gz=True)
    print(f'    {url}')
    if not download_gz(url, gz):
        return 'download-fail'
    if not decompress(gz, raw):
        return 'decompress-fail'
    return 'downloaded+decompressed'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tiles', nargs='+', default=DEFAULT_TILES)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--keep-gz', action='store_true', help='keep the .gz after decompress')
    return p.parse_args()


def main():
    a = parse_args()
    print(f'Source : {BASE}')
    print(f'Target : {DEST}\n')
    if a.verify_only:
        for t in a.tiles:
            print(f'  {t:3} i2d: {verify_i2d(DEST / base_name(t))}')
        return
    results = {}
    for t in a.tiles:
        print(f'  {t} i2d:')
        results[t] = ensure(t)
        print(f'    -> {results[t]}')
        if results[t] in ('downloaded+decompressed', 'already-complete') and not a.keep_gz:
            (DEST / base_name(t, gz=True)).unlink(missing_ok=True)
    print('\n── verification (SCI/ERR/CON/WHT each readable) ──')
    allok = True
    for t in a.tiles:
        v = verify_i2d(DEST / base_name(t)); allok &= (v == 'ok')
        sz = (DEST / base_name(t)).stat().st_size / 1e9 if (DEST / base_name(t)).exists() else 0
        print(f'  {t:3} i2d: {v:18} {sz:5.1f}G')
    print('\nAll patched-tile i2d are complete + 4-extension.' if allok else
          '\nSome i2d not ready — rerun (resumes the .gz download).')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
