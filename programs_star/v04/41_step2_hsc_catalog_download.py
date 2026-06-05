#!/usr/bin/env python
"""
41_step2_hsc_catalog_download.py — Step 2.7b: download HSC SSP s23b
coadd catalogs (deepCoadd_meas + deepCoadd_forced_src) for tract 9813
(COSMOS) via the NAOJ DAS API.

Two catalog products:
  deepCoadd_meas       — per-patch detection + morphology / model-fit
                         FITS binary table.  ~1 file/patch  → ~81 files.
  deepCoadd_forced_src — per-patch-per-band forced photometry FITS
                         binary table.  ~5 files/patch  → ~405 files.

Total ~486 files, ~3-5 GB.

Same DAS query-then-URL-list workflow as 40_step2_hsc_image_download.py.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

DAS_BASE = 'https://hscdata.mtk.nao.ac.jp/das_console/s23b'

HSC_DIR = Path('/Volumes/exdisk1/data/catalog/HSC/COSMOS')

PRODUCTS_DEFAULT = ['deepCoadd_meas', 'deepCoadd_forced_src']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rerun', default='s23b_deep2',
                   choices=['s23b_deep2', 's23b_deep', 's23b_wide'])
    p.add_argument('--tract', type=int, default=9813)
    p.add_argument('--products', nargs='+', default=PRODUCTS_DEFAULT,
                   choices=['deepCoadd_meas',
                            'deepCoadd_forced_src',
                            'deepCoadd_measMatchFull'])
    p.add_argument('--jobs', '-j', type=int, default=4)
    p.add_argument('--conn', '-x', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def _curl_post(url: str, data: dict, netrc: Path) -> bytes:
    body = urllib.parse.urlencode(data).encode()
    cmd = ['curl', '-s', '--netrc-file', str(netrc), '-X', 'POST',
           '--data-binary', '@-', url]
    r = subprocess.run(cmd, input=body, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'curl POST failed: {r.stderr.decode()}')
    return r.stdout


def _curl_get(url: str, netrc: Path) -> bytes:
    cmd = ['curl', '-sL', '--netrc-file', str(netrc), url]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'curl GET failed: {r.stderr.decode()}')
    return r.stdout


def main():
    args = parse_args()
    if shutil.which('aria2c') is None:
        sys.exit('aria2c not in PATH.  brew install aria2')
    netrc = Path.home() / '.netrc'
    if not netrc.exists() or 'hscdata.mtk.nao.ac.jp' not in netrc.read_text():
        sys.exit('~/.netrc missing or no hscdata entry')

    print(f'HSC SSP s23b DAS catalog query:')
    print(f'  rerun     : {args.rerun}')
    print(f'  tract     : {args.tract}')
    print(f'  products  : {", ".join(args.products)}')
    print(f'  out dir   : {HSC_DIR}')
    print()

    # ── Submit one query per product (DAS doesn't combine across coadd types) ──
    all_urls = []
    file_records = []
    HSC_DIR.mkdir(parents=True, exist_ok=True)
    for product in args.products:
        sql = {'type': 'and', 'operands': [
            {'type': 'tract', 'tracts': [args.tract], 'patches': []}
        ]}
        query = {
            'sqlCondition': sql,
            'frames': [],
            'coadds': [product],
            'reruns': [args.rerun],
        }
        print(f'── submitting query for {product} ──')
        t0 = time.time()
        resp = _curl_post(f'{DAS_BASE}/cgi-bin/dasQuery',
                          {'query': json.dumps(query)}, netrc)
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            sys.exit(f'Non-JSON reply for {product}:\n'
                     f'{resp[:300].decode(errors="replace")}')
        if data is None:
            print(f'  {product}: server returned null (no matches) — skipping')
            continue
        qid = data['id']
        n_coadd = len(data.get('coadds', []))
        url_resp = _curl_get(f'{DAS_BASE}/cgi-bin/urls?id={qid}&type=text', netrc)
        urls = [u for u in url_resp.decode().splitlines() if u.strip()]
        print(f'  query ID = {qid}, coadd rows = {n_coadd}, URLs = {len(urls)}  '
              f'({time.time()-t0:.1f}s)')
        all_urls.extend(urls)
        # Save the URL list per-product for the record
        (HSC_DIR / f'_url_list_{product}_{args.tract}_{args.rerun}.txt'
         ).write_text('\n'.join(urls) + '\n')

    print(f'\n  total URLs across products : {len(all_urls)}')

    # ── Build local destinations ──
    have = miss = 0
    lines = []
    for url in all_urls:
        # URL ends with: .../<tract>/<patch_id>/<band>/<fn>  OR
        #                .../<tract>/<patch_id>/<fn>          (no band for meas)
        parts = url.split('/')
        fn = parts[-1]
        # Look up tract index in path
        try:
            tract_idx = next(i for i, p in enumerate(parts)
                             if p == str(args.tract))
            sub = '/'.join(parts[tract_idx:-1])
        except StopIteration:
            sub = '/'.join(parts[-4:-1])   # fallback
        rel = f's23b/{args.rerun}/{sub}/{fn}'
        dest = HSC_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        file_records.append({'url': url, 'rel': rel})
        if dest.exists() and dest.stat().st_size > 0:
            have += 1
        else:
            miss += 1
            lines.append(url)
            lines.append(f'  out={rel}')
    print(f'Files expected   : {len(file_records)}')
    print(f'Already on disk  : {have}')
    print(f'To download      : {miss}')
    est_gb = miss * 0.01   # rough ~10 MB per catalog file
    print(f'Estimated size   : ~{est_gb:.1f} GB')
    print()

    if args.dry_run or miss == 0:
        print(f'[{"dry-run" if args.dry_run else "all-present"}]')
        return

    url_file = HSC_DIR / f'_aria2_urls_{args.tract}_{args.rerun}.txt'
    url_file.write_text('\n'.join(lines) + '\n')
    log_file = HSC_DIR / f'_aria2_{args.tract}_{args.rerun}.log'

    cmd = ['aria2c',
           f'--input-file={url_file}',
           f'--dir={HSC_DIR}',
           f'--max-concurrent-downloads={args.jobs}',
           f'--max-connection-per-server={args.conn}',
           f'--split={args.conn}',
           f'--netrc-path={netrc}',
           '--min-split-size=1M',
           '--continue=true',
           '--auto-file-renaming=false',
           '--allow-overwrite=false',
           '--check-integrity=false',
           '--conditional-get=true',
           '--remote-time=true',
           '--file-allocation=none',
           '--summary-interval=30',
           '--console-log-level=warn',
           f'--log={log_file}',
           '--log-level=notice',
           '--retry-wait=5',
           '--max-tries=10',
           '--timeout=300',
           '--connect-timeout=30',
           '--lowest-speed-limit=0']
    print('Launching aria2c ...')
    t1 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        sys.exit(130)
    dt = time.time() - t1

    n_ok = 0; total = 0
    for rec in file_records:
        d = HSC_DIR / rec['rel']
        if d.exists() and d.stat().st_size > 0:
            n_ok += 1; total += d.stat().st_size
    print(f'\naria2c exit code: {rc}   wall: {dt/60:.1f} min')
    print(f'files on disk : {n_ok}/{len(file_records)}')
    print(f'total size     : {total/1e9:.2f} GB')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source':          'HSC SSP s23b NAOJ DAS catalogs',
        'rerun':           args.rerun,
        'tract':           args.tract,
        'products':        args.products,
        'files_expected':  int(len(file_records)),
        'files_on_disk':   int(n_ok),
        'total_bytes':     int(total),
        'aria2_exit_code': int(rc),
        'wall_seconds':    round(dt, 2),
    }
    (HSC_DIR / f'hsc_catalog_{args.rerun}_{args.tract}.meta.json').write_text(
        json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
