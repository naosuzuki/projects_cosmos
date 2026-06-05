#!/usr/bin/env python
"""
40_step2_hsc_image_download.py — Step 2.7: download HSC SSP s23b
deepCoadd_calexp images over a tract (default 9813 = COSMOS), in
grizY broadband, via the NAOJ DAS request-builder API.

The DAS does NOT expose flat file URLs by rerun/tract; instead:
  1. POST a JSON query to cgi-bin/dasQuery → get a query ID
  2. GET  cgi-bin/urls?id=<ID>&type=text   → get the file URL list
  3. wget/aria2c each URL with HTTP Basic auth (~/.netrc)

Default rerun: s23b_deep2 (member-only HSC SSP DR4 Deep2 reduction).

Auth: HTTP Basic via ~/.netrc (machine hscdata.mtk.nao.ac.jp).

Storage:
  /Volumes/exdisk1/data/HSC/COSMOS/s23b/<rerun>/<tract>/<patch>/<band>/<filename>.fits
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DAS_BASE = 'https://hscdata.mtk.nao.ac.jp/das_console/s23b'

HSC_DIR = Path('/Volumes/exdisk1/data/HSC/COSMOS')

BANDS_DEFAULT = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rerun', default='s23b_deep2',
                   choices=['s23b_deep2', 's23b_deep', 's23b_wide'])
    p.add_argument('--tract', type=int, default=9813,
                   help='Tract number (default 9813 = COSMOS).')
    p.add_argument('--patches', nargs='+', default=None,
                   help='Patch list as comma-tuples "0,0" "0,1" ... '
                        '(default: all patches in the tract).')
    p.add_argument('--bands', nargs='+', default=BANDS_DEFAULT,
                   choices=BANDS_DEFAULT)
    p.add_argument('--coadd-type', default='deepCoadd_calexp',
                   choices=['deepCoadd_calexp', 'deepCoadd',
                            'deepCoadd_calexp_background',
                            'deepCoadd_directWarp'])
    p.add_argument('--jobs', '-j', type=int, default=4)
    p.add_argument('--conn', '-x', type=int, default=4)
    p.add_argument('--dry-run', action='store_true',
                   help='Submit the query + print URL list + size; do not download.')
    return p.parse_args()


def _curl_post(url: str, data: dict, netrc: Path) -> bytes:
    """POST form data with HTTP basic auth from netrc, return raw bytes."""
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
        sys.exit('~/.netrc missing or has no entry for hscdata.mtk.nao.ac.jp')

    # ── 1. Build query JSON ──
    if args.patches:
        patches_arr = []
        for p in args.patches:
            # Convert "x,y" → patch_id integer (x*100 + y), matching DAS convention
            x, y = (int(t) for t in p.split(','))
            patches_arr.append(100*x + y)
    else:
        patches_arr = []   # empty = all patches
    sql = {
        'type': 'and',
        'operands': [
            {'type': 'tract', 'tracts': [args.tract], 'patches': patches_arr},
        ],
    }
    # If the user picked a subset of bands, we'll filter the URL list
    # client-side after submission (server-side {type: "filters"} block
    # combined under "and" returns null in our testing).
    query = {
        'sqlCondition': sql,
        'frames': [],
        'coadds': [args.coadd_type],
        'reruns': [args.rerun],
    }

    print('HSC SSP s23b DAS query:')
    print(f'  rerun       : {args.rerun}')
    print(f'  tract       : {args.tract}')
    print(f'  patches     : {patches_arr if patches_arr else "all"}')
    print(f'  bands       : {" ".join(args.bands)}')
    print(f'  coadd type  : {args.coadd_type}')
    print(f'  out dir     : {HSC_DIR}')
    print()
    print('── submitting query ──')
    t0 = time.time()
    resp = _curl_post(f'{DAS_BASE}/cgi-bin/dasQuery',
                      {'query': json.dumps(query)}, netrc)
    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        sys.exit(f'Non-JSON reply (first 500 bytes):\n{resp[:500].decode(errors="replace")}')
    if data is None:
        sys.exit('Server replied null — no matching files. '
                 'Check rerun/tract/coadd-type combination.')
    qid = data['id']
    n_coadd = len(data.get('coadds', []))
    print(f'  query ID    : {qid}')
    print(f'  coadd rows  : {n_coadd}  ({time.time()-t0:.1f}s)')

    # ── 2. Fetch URL list ──
    url_resp = _curl_get(
        f'{DAS_BASE}/cgi-bin/urls?id={qid}&type=text', netrc)
    all_urls = [u for u in url_resp.decode().splitlines() if u.strip()]
    print(f'  URLs in list: {len(all_urls)}')

    # Client-side band filter: URL path contains /<band_lower>/
    wanted_lower = {b.replace('HSC-', '').lower() for b in args.bands}
    urls = [u for u in all_urls
            if any(f'/{b}/' in u for b in wanted_lower)]
    print(f'  after band filter ({sorted(wanted_lower)}): {len(urls)}')
    print()

    # ── 3. Build local destinations ──
    # URL pattern: .../<tract>/<patch_id>/<band>/<filename>.fits
    # Map to:      HSC_DIR/s23b/<rerun>/<tract>/<patch_id>/<band>/<filename>.fits
    have = miss = 0
    lines = []
    file_records = []
    for url in urls:
        # Extract tract / patch / band / filename from URL
        parts = url.split('/')
        try:
            band = parts[-2]
            patch_id = parts[-3]
            tract = parts[-4]
            fn = parts[-1]
        except IndexError:
            print(f'  WARN unusual URL: {url}')
            continue
        rel = f's23b/{args.rerun}/{tract}/{patch_id}/{band}/{fn}'
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
    # Each deepCoadd_calexp is ~150 MB
    est_gb = miss * 0.15
    print(f'Estimated size   : ~{est_gb:.1f} GB')
    print()

    if args.dry_run or miss == 0:
        print(f'[{"dry-run" if args.dry_run else "all-present"}]')
        # Save URL list as record
        ul_path = HSC_DIR / f'_url_list_{args.tract}_{args.rerun}.txt'
        HSC_DIR.mkdir(parents=True, exist_ok=True)
        ul_path.write_text('\n'.join(urls) + '\n')
        print(f'[save] {ul_path}')
        return

    # ── 4. Download via aria2c ──
    HSC_DIR.mkdir(parents=True, exist_ok=True)
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
           '--min-split-size=4M',
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
    print(f'Launching aria2c ({args.jobs} parallel × {args.conn} conn) ...')
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
        'source':          'HSC SSP s23b NAOJ DAS',
        'das_query_id':    qid,
        'rerun':           args.rerun,
        'tract':           args.tract,
        'patches':         args.patches if args.patches else 'all',
        'bands':           args.bands,
        'coadd_type':      args.coadd_type,
        'files_expected':  int(len(file_records)),
        'files_on_disk':   int(n_ok),
        'total_bytes':     int(total),
        'aria2_exit_code': int(rc),
        'wall_seconds':    round(dt, 2),
    }
    (HSC_DIR / f'hsc_{args.rerun}_{args.tract}.meta.json').write_text(
        json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
