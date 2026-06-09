#!/usr/bin/env python
"""
04_download_jwst_nircam_i2d.py — (re)download JWST/NIRCam COSMOS-Web 30-mas
i2d mosaics from the v0.8 reduction, with the same tuned aria2c engine as the
HST download scripts (01/02/03_download_*.py).

Motivation: a few f115w i2d downloads arrived TRUNCATED — short on their WHT
extension, so the PSF builder couldn't weight on them.  A full-disk scan of all
80 i2d (A1..A10, B1..B10 × 4 bands) found exactly 4 bad files, all f115w:

    A2  f115w   truncated (24.6 of 26.8 GB; WHT extension missing)
    A10 f115w   absent entirely
    B4  f115w   truncated (WHT missing)
    B6  f115w   truncated (WHT missing)

These 4 are the DEFAULT target set.  (scidir holds a complete single-HDU SCI
for each, which the pipeline uses as a weightless fallback — but we want the
proper weighted i2d.)

Source : https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8/
Target : /Volumes/exdisk1/data/JWST/COSMOS_v0.8/
Auth   : HTTP Basic via ~/.netrc (same as the ACS downloads)
File   : mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v1.0_i2d.fits  (~25-27 GB)

Engine : aria2c, same flags that pulled the ACS mosaics cleanly
         (exchg.calet.org is bandwidth-capped; keep concurrency low).
         --continue=true resumes the truncated partials byte-for-byte; the
         already-correct leading bytes are kept and only the missing tail is
         fetched (so A2/B4/B6 only pull their last ~2 GB, A10 the full file).

After downloading, every requested file is re-opened with astropy and its
SCI + WHT extensions verified — the exact integrity test that flagged the
truncation in the first place.

Usage  :  python 04_download_jwst_nircam_i2d.py                 # the 4 bad ones
          python 04_download_jwst_nircam_i2d.py --dry-run
          python 04_download_jwst_nircam_i2d.py --tiles A2 --bands f115w
          python 04_download_jwst_nircam_i2d.py --verify-only   # just re-check
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time, warnings
from pathlib import Path

BASE_URL = 'https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8/'
DEST     = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
PIXSCALE = '30mas'
# default = the files the integrity scan flagged as incomplete
DEFAULT_TARGETS = [('A2', 'f115w'), ('A10', 'f115w'),
                   ('B4', 'f115w'), ('B6', 'f115w')]
ALL_TILES = [f'{r}{i}' for r in 'AB' for i in range(1, 11)]
ALL_BANDS = ['f115w', 'f150w', 'f277w', 'f444w']


def fname(tile: str, band: str) -> str:
    return f'mosaic_nircam_{band}_COSMOS-Web_{PIXSCALE}_{tile}_v1.0_i2d.fits'


def build_url(tile: str, band: str) -> str:
    return BASE_URL + fname(tile, band)


def verify(path: Path) -> str:
    """Return 'ok' if SCI+WHT both readable, else a short reason."""
    if not path.exists():
        return 'absent'
    warnings.simplefilter('ignore')
    try:
        from astropy.io import fits
        with fits.open(path) as h:
            _ = h['SCI'].header['NAXIS1']
            _ = h['WHT'].header['NAXIS1']
        return 'ok'
    except Exception as e:
        return f'{type(e).__name__}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tiles', nargs='+', default=None,
                   help='tiles to fetch (default: the 4 flagged-incomplete files)')
    p.add_argument('--bands', nargs='+', default=None, choices=ALL_BANDS,
                   help='bands to fetch (used with --tiles; default f115w)')
    p.add_argument('--jobs', '-j', type=int, default=2,
                   help='files in parallel (default 2 — server is bw-capped)')
    p.add_argument('--conn', '-x', type=int, default=4, help='connections/file')
    p.add_argument('--split', '-s', type=int, default=4, help='segments/file')
    p.add_argument('--dry-run', action='store_true', help='print URLs, exit')
    p.add_argument('--verify-only', action='store_true',
                   help='only re-check SCI+WHT integrity of the targets')
    return p.parse_args()


def targets(args):
    if args.tiles:
        bands = args.bands or ['f115w']
        return [(t, b) for t in args.tiles for b in bands]
    return list(DEFAULT_TARGETS)


def main():
    args = parse_args()
    tgts = targets(args)

    print(f'Source : {BASE_URL}')
    print(f'Target : {DEST}')
    print(f'Files  : {len(tgts)}  -> ' +
          ', '.join(f'{t}/{b}' for t, b in tgts))
    print()

    # integrity check (before / verify-only)
    print('── current integrity ──')
    need = []
    for t, b in tgts:
        st = verify(DEST / fname(t, b))
        print(f'  {t:3} {b:6}  {st}')
        if st != 'ok':
            need.append((t, b))
    if args.verify_only:
        return
    if not need:
        print('\nAll targets already complete (SCI+WHT OK).  Nothing to do.')
        return

    urls = [build_url(t, b) for t, b in need]
    print(f'\n{len(need)} file(s) to (re)download: ' +
          ', '.join(f'{t}/{b}' for t, b in need))
    if args.dry_run:
        print('-- dry run, URL list --')
        for u in urls:
            print(u)
        return

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not found.  brew install aria2')
    netrc = Path.home() / '.netrc'
    if not netrc.exists():
        sys.exit(f'{netrc} not found — server requires HTTP Basic auth.')

    DEST.mkdir(parents=True, exist_ok=True)
    url_file = DEST / '_aria2_jwst_urls.txt'
    url_file.write_text('\n'.join(urls) + '\n')
    log_file = DEST / '_aria2_jwst.log'
    cmd = [
        'aria2c',
        f'--input-file={url_file}',
        f'--dir={DEST}',
        f'--max-concurrent-downloads={args.jobs}',
        f'--max-connection-per-server={args.conn}',
        f'--split={args.split}',
        '--min-split-size=20M',
        '--continue=true',                 # resume the truncated partials
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
    print('\nLaunching aria2c ...')
    print(' '.join(cmd))
    print(f'(full log in {log_file})\n')
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print('\n[interrupted] partial files left in place; rerun to resume.')
        sys.exit(130)
    print(f'\naria2c exit code: {rc}     wall: {(time.time()-t0)/60:.1f} min')

    # post-download integrity check
    print('\n── post-download integrity ──')
    allok = True
    for t, b in need:
        st = verify(DEST / fname(t, b))
        print(f'  {t:3} {b:6}  {st}')
        allok &= (st == 'ok')
    print('\nAll targets complete (SCI+WHT OK).' if allok else
          '\nSome targets still incomplete — rerun to resume.')


if __name__ == '__main__':
    main()
