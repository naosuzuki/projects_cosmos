#!/usr/bin/env python
"""
29_step2_ps1_image_download.py — Step 2.3c: download PS1 stack
imaging (unconvolved warps + masks + weights) over the Euclid VIS
polygon or the 3-way subset.

Image-release vs catalog-release nuance:
  PS1 only ever produced one set of stack images — the rings.v3
  projection built for DR1 in 2016.  DR2 (2019) reprocessed the
  CATALOGS (Gaia-DR2-tied astrometry, per-detector zeropoints, the
  ForcedMeanObject table) on the SAME pixels.  So strictly:
    - 27_ pulled DR2 catalog (MeanObjectView)
    - 29_ pulls DR1 rings.v3 stacks — the images the DR2 catalog
      was measured from.
  Nobody has "DR2 stacks" because they don't exist as a separate
  product.

Why stacks and not per-source cutouts:
  185 796 PS1 sources × 5 bands × 240" cutouts would be ~46 GB of
  tiny files — much harder to work with than the ~25 native PS1
  skycell stacks that already tile the same area (each 0.4° × 0.4°,
  ~250 MB fz-compressed).  PSF photometry in Step 3 wants the full
  skycell anyway (PSF + sky model).

PS1 stack image discovery:
  https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=...&dec=...&type=stack
  returns the (projcell, subcell, filter, type, filename) tuples
  covering that RA/Dec.  We sample a grid of points across the scope
  polygon, dedupe by filename, then bulk-download with aria2c.

Stack file URL pattern (PS1 DR2):
  https://ps1images.stsci.edu/rings.v3.skycell/{projcell}/{subcell}/rings.v3.skycell.{projcell}.{subcell}.stk.{band}.{type}.fits

Image types we fetch by default:
  unconv          — unconvolved stack image (best for PSF photometry)
  wt              — inverse-variance weight map (per pixel)
  mask            — bitmask (saturated, cosmic-ray, etc.)
Add 'exp', 'num', 'expwt' via --types if needed (each adds ~150 MB / cell).

Defaults:
  --scope 3way    (default — smaller, faster, ~6-10 GB)
  --scope vis     ~30-40 GB across the full Euclid VIS coverage
  --bands g r i z y

Storage (mirrors PS1 DAS hierarchy under /Volumes/exdisk1):
  /Volumes/exdisk1/data/PanSTARRS/COSMOS/skycells/
      rings.v3.skycell/{projcell}/{subcell}/
          rings.v3.skycell.{projcell}.{subcell}.stk.{band}.{type}.fits
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PS1_DIR   = Path('/Volumes/exdisk1/data/PanSTARRS/COSMOS')
CELL_ROOT = PS1_DIR / 'skycells'

PS1_LOOKUP  = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
PS1_ARCHIVE = 'https://ps1images.stsci.edu'

BANDS_DEFAULT = ['g', 'r', 'i', 'z', 'y']
# Image auxiliary types — the lookup CGI only returns the canonical
# .stk.{band}.unconv.fits filename; the wt/mask/exp/num companions are
# named by tacking these suffixes onto the .fits stem (PS1 convention).
AUX_TYPES_DEFAULT = ['wt', 'mask']
# Available aux suffixes: 'wt' (inverse-variance), 'mask' (bitmask),
# 'exp' (exposure-time map), 'num' (count of contributing exposures).
# The main image (unconv.fits) is always downloaded.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scope', choices=['vis', '3way'], default='3way',
                   help='Spatial coverage: 3way (default, ~6 GB) or vis (~30 GB).')
    p.add_argument('--bands', nargs='+', default=BANDS_DEFAULT,
                   choices=BANDS_DEFAULT,
                   help='PS1 bands to fetch (default grizy).')
    p.add_argument('--aux', nargs='*', default=AUX_TYPES_DEFAULT,
                   choices=['wt', 'mask', 'exp', 'num'],
                   help='Aux maps in addition to the main unconv image '
                        '(default wt + mask).  Pass --aux with no values '
                        'to download images only.')
    p.add_argument('--grid-step', type=float, default=0.08,
                   help='Discovery-grid step in degrees (default 0.08 — '
                        'finer than the 0.4 deg skycell size to guarantee '
                        'every cell is sampled).')
    p.add_argument('--jobs', '-j', type=int, default=4,
                   help='aria2c concurrent downloads (default 4).')
    p.add_argument('--conn', '-x', type=int, default=4,
                   help='aria2c connections per server (default 4).')
    p.add_argument('--dry-run', action='store_true',
                   help='Discover skycells + print list+size; do not download.')
    p.add_argument('--no-download', action='store_true',
                   help='Same as --dry-run but also writes the URL list file.')
    return p.parse_args()


def load_scope_polygon(scope: str):
    """Return a shapely polygon for the requested scope."""
    from shapely import wkt as shp_wkt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if scope == '3way':
        WKT_3WAY = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                        'common_area_3way.wkt')
        return shp_wkt.loads(WKT_3WAY.read_text())
    # scope == 'vis' — build VIS polygon from tile_lookup
    LOOKUP = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04/'
                  'tile_lookup.parquet')
    tdf = pd.read_parquet(LOOKUP)
    vis = tdf[(tdf['mission'] == 'Euclid') & (tdf['filter'] == 'VIS')]
    polys = []
    for _, r in vis.iterrows():
        ring = list(zip(r['corners_ra'], r['corners_dec']))
        ring.append(ring[0])
        p = Polygon(ring); p = p if p.is_valid else p.buffer(0)
        polys.append(p)
    return unary_union(polys)


def grid_inside(poly, step_deg: float):
    """Return list of (ra, dec) inside the polygon on a regular grid."""
    minx, miny, maxx, maxy = poly.bounds
    ras  = np.arange(minx, maxx + step_deg/2, step_deg)
    decs = np.arange(miny, maxy + step_deg/2, step_deg)
    pts = [(ra, dec)
           for dec in decs for ra in ras
           if poly.contains(__import__('shapely.geometry', fromlist=['Point']).Point(ra, dec))]
    return pts


def lookup_skycells(grid_pts, bands, timeout=60):
    """Hit PS1 filenames.py for the discovery grid and collect skycell rows.

    Uses POST with a ra/dec text file upload — the canonical batched form,
    and avoids the URL-length limit a GET with hundreds of points would hit.
    Returns one DataFrame whose columns include projcell, subcell, filter,
    type, filename (whitespace-separated CSV).
    """
    import io
    f_str = ''.join(bands)              # e.g. 'grizy'
    # Build the radec.txt payload (one "ra dec" line per point, with header).
    radec_buf = io.StringIO()
    radec_buf.write('ra dec\n')
    for ra, dec in grid_pts:
        radec_buf.write(f'{ra:.5f} {dec:.5f}\n')
    radec_bytes = radec_buf.getvalue().encode()

    # Even at 5000 points the endpoint is fine — we send one POST.
    files = {'file': ('radec.txt', radec_bytes, 'text/plain')}
    data  = {'filters': f_str, 'type': 'stack', 'format': 'csv'}
    t0 = time.time()
    r = requests.post(PS1_LOOKUP, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    # Response is whitespace-separated, even with format=csv (PS1 quirk)
    sub = pd.read_csv(io.StringIO(r.text), sep=r'\s+')
    print(f'  POST {len(grid_pts)} pts → {len(sub):,} filename rows '
          f'({time.time()-t0:.1f}s)', flush=True)
    # Dedupe by filename — overlapping query points return the same cell
    sub = sub.drop_duplicates(subset=['filename']).reset_index(drop=True)
    return sub


def aux_filename(main_path: str, aux: str) -> str:
    """Derive a PS1 auxiliary-map filename from the main .unconv.fits path.

    Input  : /rings.v3.skycell/1360/059/rings.v3.skycell.1360.059.stk.g.unconv.fits
    Output : /rings.v3.skycell/1360/059/rings.v3.skycell.1360.059.stk.g.unconv.wt.fits
             (or .mask.fits, .exp.fits, .num.fits depending on `aux`)
    """
    # Replace trailing '.unconv.fits' with '.unconv.{aux}.fits'
    assert main_path.endswith('.unconv.fits'), \
        f'Unexpected stack filename: {main_path}'
    return main_path[:-len('.fits')] + f'.{aux}.fits'


def main():
    args = parse_args()
    CELL_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = CELL_ROOT.parent

    if shutil.which('aria2c') is None:
        sys.exit('aria2c not in PATH.  brew install aria2')

    print(f'Scope         : {args.scope}')
    print(f'Bands         : {" ".join(args.bands)}')
    print(f'Aux maps      : {" ".join(args.aux) if args.aux else "(images only)"}')
    print(f'Grid step     : {args.grid_step}°')
    print(f'Target dir    : {CELL_ROOT}')
    print()

    # ── 1. Build the discovery grid ────────────────────────────────────
    poly = load_scope_polygon(args.scope)
    n_pieces = len(poly.geoms) if poly.geom_type == 'MultiPolygon' else 1
    print(f'Scope polygon : {n_pieces} pieces, area = {poly.area:.4f} deg²')
    grid_pts = grid_inside(poly, args.grid_step)
    print(f'Grid points   : {len(grid_pts):,}')
    if not grid_pts:
        sys.exit('Empty grid — polygon probably degenerate.')
    print()

    # ── 2. Discover unique skycells via ps1filenames.py ────────────────
    print(f'Querying PS1 filenames.py ({len(args.bands)} bands × type=stack) ...')
    t0 = time.time()
    sk = lookup_skycells(grid_pts, args.bands)
    print(f'  unique unconv-image rows discovered: {len(sk):,}  '
          f'(wall {time.time()-t0:.1f}s)')
    if len(sk) == 0:
        sys.exit('PS1 returned no skycells — check connectivity / scope.')

    # PS1 ps1filenames.py returns columns:
    #   projcell, subcell, ra, dec, filter, mjd, type, filename, shortname
    print(f'  columns      : {list(sk.columns)}')
    # Quick distribution print
    if 'filter' in sk.columns:
        print('  by filter    :',
              dict(sk['filter'].value_counts().to_dict()))
    n_cells = sk[['projcell', 'subcell']].drop_duplicates().shape[0]
    print(f'  unique skycells: {n_cells:,}')

    # ── 3. Save the discovery table for the record ────────────────────
    sk.to_parquet(PS1_DIR / f'ps1_skycells_{args.scope}.parquet', index=False)
    sk.to_csv(    PS1_DIR / f'ps1_skycells_{args.scope}.csv',     index=False)
    print(f'[save] ps1_skycells_{args.scope}.parquet / .csv  '
          f'({len(sk):,} rows)')

    # ── 4. Expand each unconv row into main + aux files; build URL list ─
    url_file = CELL_ROOT / f'_aria2_urls_{args.scope}.txt'
    lines = []
    have = miss = 0
    file_records = []   # for the meta summary
    for _, r in sk.iterrows():
        main_fn = r['filename']
        # filename starts with '/rings.v3.skycell/...'
        for suffix in [None] + list(args.aux):
            fn = main_fn if suffix is None else aux_filename(main_fn, suffix)
            url = PS1_ARCHIVE + fn
            rel = fn.lstrip('/')
            dest = CELL_ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            file_records.append({
                'projcell': r['projcell'], 'subcell': r['subcell'],
                'filter':   r['filter'],   'suffix':  suffix or 'image',
                'url':      url,           'rel':     rel,
            })
            if dest.exists() and dest.stat().st_size > 0:
                have += 1
                continue
            lines.append(url)
            lines.append(f'  out={rel}')
            miss += 1
    print(f'  files (main + aux): {len(file_records):,}  '
          f'(= {len(sk):,} cells × {1+len(args.aux)} types)')
    print(f'  already on disk   : {have:,}')
    print(f'  to download       : {miss:,}')
    # PS1 stack files are fits.fz compressed: unconv ~150 MB, wt ~75 MB,
    # mask ~30 MB, exp ~75 MB, num ~30 MB. Conservative average ~80 MB.
    est_gb = miss * 0.08
    print(f'  estimated size    : ~{est_gb:.1f} GB ({miss * 80 / 1024:.1f} GB)')
    print()

    if args.dry_run or args.no_download or miss == 0:
        if miss > 0 and args.no_download:
            url_file.write_text('\n'.join(lines) + '\n')
            print(f'[save] URL list at {url_file}')
        print(f'[{"dry-run" if args.dry_run else "no-download" if args.no_download else "all-present"}] not invoking aria2c.')
        return

    # ── 5. Write URL list + run aria2c ─────────────────────────────────
    url_file.write_text('\n'.join(lines) + '\n')
    log_file = CELL_ROOT / f'_aria2_{args.scope}.log'

    cmd = [
        'aria2c',
        f'--input-file={url_file}',
        f'--dir={CELL_ROOT}',
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

    # ── 6. Summary ────────────────────────────────────────────────────
    n_on_disk = 0; total = 0
    for rec in file_records:
        dest = CELL_ROOT / rec['rel']
        if dest.exists():
            n_on_disk += 1
            total += dest.stat().st_size
    print(f'\naria2c exit code: {rc}   wall: {dt/60:.1f} min')
    print(f'files on disk : {n_on_disk:>5}/{len(file_records)}')
    print(f'total size     : {total/1e9:.2f} GB')
    print()

    # Persist a summary meta.json
    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'image_release':   'PS1 DR1 rings.v3 stacks (same pixels DR2 catalog was measured from)',
        'scope':           args.scope,
        'bands':           args.bands,
        'aux_types':       args.aux,
        'scope_polygon_area_deg2': round(poly.area, 4),
        'grid_step_deg':   args.grid_step,
        'grid_points':     len(grid_pts),
        'unique_skycell_images': int(len(sk)),
        'files_expected':        int(len(file_records)),
        'files_on_disk':         int(n_on_disk),
        'total_bytes':           int(total),
        'aria2_exit_code':       int(rc),
        'wall_seconds':          round(dt, 2),
    }
    meta_path = PS1_DIR / f'ps1_skycells_{args.scope}.meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[save] {meta_path.name}')

    if n_on_disk == len(file_records):
        print('All requested skycell files present.  Done.')
    else:
        print('Some files missing — re-run to resume.')


if __name__ == '__main__':
    main()
