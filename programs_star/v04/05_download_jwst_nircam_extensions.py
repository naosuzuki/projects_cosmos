#!/usr/bin/env python
"""
05_download_jwst_nircam_extensions.py — download the INDIVIDUAL extension
mosaics (sci / wht / err) for the JWST/NIRCam tiles whose combined i2d is
broken at the source.

Why: the v0.8 combined i2d for A2, B4, B6 (f115w) are truncated IN THE ARCHIVE
(headers declare ~26.8 GB but the file stops at ~24.6 GB, so the WHT extension
is missing — aria2 reports them "already complete" because local size == the
truncated server size).  The separate per-extension mosaics in
  .../v0.8/extension_mosaics/
sidestep that: each is a clean single-HDU ~1.91 GB file.  A10 was already
fetched this way (its sci+wht+err are in scidir), so the PSF builder can use
scidir SCI + WHT (weighted) instead of the weightless fallback.

Filename convention (matched to the A10 files already in scidir):
    sci :  mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v0_8_sci.fits
    wht :  mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v1.0_wht.fits
    err :  mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v1.0_err.fits

Source : https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8/extension_mosaics/
Target : /Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir/      (same dir as the SCI)
Auth   : HTTP Basic via ~/.netrc

Default targets: A2, B4, B6 (f115w) × {sci, wht, err}.  Already-present files
(e.g. the SCI, which scidir has for all tiles) are skipped — so this typically
only pulls the 6 missing wht+err files (~11.5 GB).  A10 is already complete.

Usage  :  python 05_download_jwst_nircam_extensions.py
          python 05_download_jwst_nircam_extensions.py --dry-run
          python 05_download_jwst_nircam_extensions.py --verify-only
          python 05_download_jwst_nircam_extensions.py --tiles A2 --products wht err
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time, warnings
from pathlib import Path

BASE_URL = ('https://exchg.calet.org/cosmosweb/COSMOS-Web_Jan24/'
            'NIRCam/v0.8/extension_mosaics/')
DEST     = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
PIXSCALE = '30mas'
PROD_VER = {'sci': 'v1.0', 'wht': 'v1.0', 'err': 'v1.0'}   # v1.0 throughout (consistent)
DEFAULT_TILES    = ['A2', 'B4', 'A10', 'B6']
DEFAULT_PRODUCTS = ['sci', 'wht', 'err']


def fname(tile: str, band: str, prod: str) -> str:
    return f'mosaic_nircam_{band}_COSMOS-Web_{PIXSCALE}_{tile}_{PROD_VER[prod]}_{prod}.fits'


def build_url(tile: str, band: str, prod: str) -> str:
    return BASE_URL + fname(tile, band, prod)


def verify(path: Path) -> str:
    """'ok' only if the file holds at least header+data bytes (not truncated).

    A truncated FITS still opens with a valid NAXIS header — the data block is
    just short — so 'opens ok' is NOT enough (that bug let interrupted downloads
    pass as complete and got skipped on resume).  Compare the on-disk size to the
    bytes the header DECLARES (NAXIS1*NAXIS2*|BITPIX|/8) WITHOUT loading data."""
    if not path.exists():
        return 'absent'
    warnings.simplefilter('ignore')
    try:
        import os
        from astropy.io import fits
        with fits.open(path) as h:
            for hd in h:
                hh = hd.header
                if hh.get('NAXIS', 0) >= 2:
                    need = hh['NAXIS1'] * hh['NAXIS2'] * (abs(hh['BITPIX']) // 8)
                    if os.path.getsize(path) < need:
                        return (f'truncated({os.path.getsize(path)/1e9:.2f}/'
                                f'{need/1e9:.2f}G)')
                    return 'ok'
        return 'no-image-hdu'
    except Exception as e:
        return f'{type(e).__name__}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tiles', nargs='+', default=DEFAULT_TILES)
    p.add_argument('--band', default='f115w')
    p.add_argument('--products', nargs='+', default=DEFAULT_PRODUCTS,
                   choices=DEFAULT_PRODUCTS)
    p.add_argument('--jobs', '-j', type=int, default=2)
    p.add_argument('--conn', '-x', type=int, default=4)
    p.add_argument('--split', '-s', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    items = [(t, args.band, pr) for t in args.tiles for pr in args.products]

    print(f'Source : {BASE_URL}')
    print(f'Target : {DEST}')
    print(f'Files  : {len(items)}  ({len(args.tiles)} tiles × {len(args.products)} products)')
    print('\n── current state in scidir ──')
    need = []
    for t, b, pr in items:
        st = verify(DEST / fname(t, b, pr))
        print(f'  {t:3} {pr}  {st}')
        if st != 'ok':
            need.append((t, b, pr))
    if args.verify_only:
        return
    if not need:
        print('\nAll requested extensions already present + valid.  Nothing to do.')
        return

    urls = [build_url(*it) for it in need]
    print(f'\n{len(need)} file(s) to download:')
    for it in need:
        print(f'  {fname(*it)}')
    if args.dry_run:
        print('\n-- dry run, URL list --')
        for u in urls:
            print(u)
        return

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not found.  brew install aria2')
    netrc = Path.home() / '.netrc'
    if not netrc.exists():
        sys.exit(f'{netrc} not found — server requires HTTP Basic auth.')

    DEST.mkdir(parents=True, exist_ok=True)
    url_file = DEST / '_aria2_ext_urls.txt'
    url_file.write_text('\n'.join(urls) + '\n')
    log_file = DEST / '_aria2_ext.log'
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
    print('\nLaunching aria2c ...')
    print(f'(full log in {log_file})\n')
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print('\n[interrupted] partial files left in place; rerun to resume.')
        sys.exit(130)
    print(f'\naria2c exit code: {rc}     wall: {(time.time()-t0)/60:.1f} min')

    print('\n── post-download verification ──')
    allok = True
    for t, b, pr in need:
        st = verify(DEST / fname(t, b, pr))
        print(f'  {t:3} {pr}  {st}')
        allok &= (st == 'ok')
    print('\nAll requested extensions present + valid.' if allok else
          '\nSome still missing/invalid — rerun to resume.')


if __name__ == '__main__':
    main()
