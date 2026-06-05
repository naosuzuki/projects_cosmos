#!/usr/bin/env python
"""
38_step2_galex_image_download.py — Step 2.5c: download GALEX FUV+NUV
intensity maps (+ rrhr exposure-time + mask) for observations
covering the Euclid VIS polygon.

GALEX coverage in COSMOS comes from four surveys with different
depths:
  AIS — All-Sky Imaging Survey  (FUV/NUV ~ 20.5/20.5, t ~ 100 s)
  MIS — Medium Imaging Survey   (~ 22.7/22.7,           ~ 1500 s)
  DIS — Deep Imaging Survey     (~ 24.8/24.4,           ~ 30 ks)
  GII — Guest Investigator      (varies)
For COSMOS the DIS coverage is the canonical deep dataset.

Discovery: astroquery.mast.Observations.query_region over the bbox,
filter to GALEX, then for each unique obs_id pull the product list
and download the per-band intensity (-int.fits.gz) + relative-response
(-rrhr.fits.gz) + flag (-flag.fits.gz) files.

Outputs (mirroring MAST's directory layout):
  /Volumes/exdisk1/data/GALEX/COSMOS/mastDownload/GALEX/<obs_id>/*.fits.gz
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path

# Euclid VIS bbox
RA_MIN,  RA_MAX  = 148.85, 151.30
DEC_MIN, DEC_MAX =   1.10,   3.40

OUT_DIR = Path('/Volumes/exdisk1/data/GALEX/COSMOS')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--surveys', nargs='+',
                   default=['DIS', 'MIS'],
                   choices=['DIS', 'MIS', 'AIS', 'GII'],
                   help='Which GALEX surveys to include (default DIS+MIS '
                        '— skip shallow AIS and heterogeneous GII).')
    p.add_argument('--product-types', nargs='+',
                   default=['intensity', 'rrhr', 'flag'],
                   choices=['intensity', 'rrhr', 'flag', 'skybg', 'cnt',
                            'objmask', 'response'],
                   help='Which per-observation product types to fetch.')
    p.add_argument('--dry-run', action='store_true',
                   help='List products + size; do not download.')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('GALEX (MAST Observations) imaging download:')
    print(f'  bbox      : RA {RA_MIN}..{RA_MAX}, Dec {DEC_MIN}..{DEC_MAX}')
    print(f'  surveys   : {", ".join(args.surveys)}')
    print(f'  products  : {", ".join(args.product_types)}')
    print(f'  out dir   : {OUT_DIR}')
    print()

    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    # ── Discovery ──
    centre = SkyCoord(0.5*(RA_MIN+RA_MAX), 0.5*(DEC_MIN+DEC_MAX), unit='deg')
    # Bbox diagonal radius + small pad
    import math
    r = 0.5 * math.hypot(RA_MAX-RA_MIN, DEC_MAX-DEC_MIN) + 0.2
    print(f'Discovery cone : centre={centre.to_string("hmsdms")}, radius={r:.3f}°')
    t0 = time.time()
    obs = Observations.query_region(centre, radius=r*u.deg)
    print(f'  total observations          : {len(obs):,}  '
          f'({time.time()-t0:.1f}s)')
    g = obs[obs['obs_collection'] == 'GALEX']
    print(f'  GALEX observations          : {len(g):,}')

    # GALEX target_name encodes the survey (DIS/MIS/AIS or GII prefix);
    # apply requested-survey filter
    def survey_of(name):
        s = str(name)
        if s.startswith('PS_'):       return 'DIS'   # PhotoSci COSMOS DIS pointings
        if s.startswith('MISDR'):     return 'MIS'
        if s.startswith('AIS'):       return 'AIS'
        return 'GII'                                  # everything else / GII
    if 'survey' in g.colnames:
        s_arr = [survey_of(n) for n in g['target_name']]
    else:
        s_arr = [survey_of(n) for n in g['target_name']]
    keep_mask = [s in args.surveys for s in s_arr]
    g = g[keep_mask]
    print(f'  after survey filter         : {len(g):,}')
    if len(g) == 0:
        sys.exit('No GALEX observations matched the requested surveys.')

    # Save the discovery table for record
    g_pd = g.to_pandas() if hasattr(g, 'to_pandas') else g
    n_obs = len(g)
    disc_path = OUT_DIR / f'galex_obs_discovery.parquet'
    try:
        # Pandas can choke on object types; cast to str
        for c in list(g_pd.columns):
            if g_pd[c].dtype.kind in 'OS':
                g_pd[c] = g_pd[c].astype(str)
        g_pd.to_parquet(disc_path, index=False)
        print(f'  saved discovery table       : {disc_path.name} '
              f'({len(g_pd):,} rows)')
    except Exception as e:
        print(f'  could not save discovery table: {e!r}')

    # ── Product enumeration (each obs has many product types) ──
    print(f'\nEnumerating products for {n_obs} observations ...', flush=True)
    t1 = time.time()
    products = Observations.get_product_list(g)
    print(f'  total products              : {len(products):,}  '
          f'({time.time()-t1:.1f}s)')

    # ── Filter to wanted product types via filename suffix ──
    # GALEX product filenames look like <obs>-<band>-<type>.fits.gz where
    # band in {fd=FUV, nd=NUV, xg=composite} and type in {int, intbgsub,
    # cnt, exp, rrhr, flagstar, ncat, ...}.  We grep on the suffix.
    import re
    # Map user-facing names to filename-suffix regex pieces
    suffix_map = {
        'intensity': 'int',          # raw intensity (counts/s)
        'rrhr':      'rrhr',         # high-res relative-response (flat)
        'flag':      'flagstar',     # bright-star flag map
        'skybg':     'intbgsub',     # actually background-subtracted int
        'cnt':       'cnt',
        'objmask':   'objmask',
        'response':  'rr',
    }
    wanted_suffixes = [suffix_map[t] for t in args.product_types]
    pat = re.compile(
        r'-(fd|nd)-(' + '|'.join(re.escape(s) for s in wanted_suffixes)
        + r')\.fits(\.gz)?$')
    sub = products[[bool(pat.search(str(fn)))
                    for fn in products['productFilename']]]
    print(f'  matching requested types    : {len(sub):,}')
    if len(sub) == 0:
        from collections import Counter
        suff = Counter()
        for fn in products['productFilename']:
            s = str(fn)
            parts = s.split('-')
            if len(parts) >= 2:
                last = parts[-1].replace('.fits.gz', '').replace('.fits', '')
                band = parts[-2]
                suff[f'{band}-{last}'] += 1
        print('  filename-suffix breakdown (top 15):')
        for k, v in sorted(suff.items(), key=lambda x: -x[1])[:15]:
            print(f'    {v:4d}  {k}')
        sys.exit(1)
    # Show breakdown by suffix
    from collections import Counter
    cb = Counter()
    for fn in sub['productFilename']:
        m = pat.search(str(fn))
        if m:
            cb[f'{m.group(1)}-{m.group(2)}'] += 1
    print('  breakdown by band/type:')
    for k, v in sorted(cb.items()):
        print(f'    {v:4d}  {k}')

    # Size estimate from `size` column (bytes)
    if 'size' in sub.colnames:
        total_gb = sub['size'].sum() / 1e9
        print(f'  total payload (uncompressed estimate)  : {total_gb:.1f} GB')

    if args.dry_run:
        print('\n[dry-run] not downloading.')
        return

    # ── Download ──
    print(f'\nDownloading to {OUT_DIR} via Observations.download_products() ...',
          flush=True)
    t2 = time.time()
    manifest = Observations.download_products(sub, download_dir=str(OUT_DIR),
                                              extension='fits.gz')
    dt = time.time() - t2
    print(f'  wall: {dt/60:.1f} min')
    print(f'  manifest rows: {len(manifest)}')

    # Persist meta
    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'image_release':   'GALEX GR6+7 (MAST Observations API)',
        'surveys':         args.surveys,
        'product_types':   args.product_types,
        'discovery_radius_deg': round(r, 3),
        'n_observations':  int(n_obs),
        'n_products':      int(len(sub)),
        'wall_seconds':    round(dt, 2),
    }
    (OUT_DIR / 'galex_images.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[save] galex_images.meta.json')


if __name__ == '__main__':
    main()
