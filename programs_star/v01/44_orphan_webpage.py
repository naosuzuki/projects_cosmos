#!/usr/bin/env python
"""
44_orphan_webpage.py — paginated HTML for orphans_v5, in the style of
                       htmls/orphan_star/v01/euclid_only_index0001.html

Layout per candidate row:
  [JWST F115W] [JWST F277W] [HST F814W] [Euclid VIS] [Euclid NISP-J]

Per `found_in` (HST-only, JWST-only, Euclid-only) it writes one
paginated index (100 rows/page).

Outputs (in htmls/star_orphans_v01/):
  index.html                       summary / navigation
  hst_only_index<NNNN>.html        paginated
  jwst_only_index<NNNN>.html
  euclid_only_index<NNNN>.html
  thumbs/<found>_<seq>_<band>.png  5″ cutouts
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astropy import units as u

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML_ROOT = ROOT / 'htmls' / 'star_orphans_v01'
THUMB_DIR = HTML_ROOT / 'thumbs'
THUMB_DIR.mkdir(parents=True, exist_ok=True)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

CUTOUT_AS = 5.0
PER_PAGE  = 100
TOP_N_PER_CAT = 600   # 6 pages × 100 per category

BANDS_SHOW = [
    ('JWST',   'F115W'),
    ('JWST',   'F277W'),
    ('HST',    'F814W'),
    ('Euclid', 'VIS'),
    ('Euclid', 'NIR-J'),
]

CSS = """
<style>
body { font-family: serif; margin: 0; padding: 0; background: #fafafa; }
.header { position: sticky; top: 0; background: white;
          border-bottom: 1px solid #ddd; padding: 8px; z-index: 10; }
.pagination { display: inline-block; }
.pagination a {
  color: black; float: left; padding: 6px 12px; text-decoration: none;
  transition: background-color .2s; border: 1px solid #ccc; margin: 1px;
}
.pagination a.active { background-color: #4CAF50; color: white; }
.pagination a:hover:not(.active) { background-color: #ddd; }
.row { padding: 4px 8px; border-bottom: 1px solid #eee; }
.row h3 { margin: 6px 0 4px 0; font-size: 14px; color: #222; }
.imgs img { width: 18%; margin: 0 0.5%; vertical-align: top; }
.imgs span { display: inline-block; width: 18%; margin: 0 0.5%;
             text-align: center; vertical-align: top;
             padding: 60px 0; color: #888; font-style: italic;
             background: #f0f0f0; }
</style>
"""


def _path_for(mission, band, tile, fname=''):
    if mission == 'HST':
        return str(HST_DIR / f'acs_I_030mas_{int(tile):03d}_sci.fits')
    if mission == 'JWST':
        if tile == 'A10' and band == 'F115W':
            return str(JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits')
        return str(JWST_DIR / f'mosaic_nircam_{band.lower()}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits')
    if mission == 'Euclid':
        return str(EUCLID_DIR / fname)
    return None


def assign_tiles(ra, dec, tile_df, mission, band):
    sel = tile_df[(tile_df['mission'] == mission) & (tile_df['band'] == band)]
    out_tile = np.full(len(ra), '', dtype=object)
    out_file = np.full(len(ra), '', dtype=object)
    for r in sel.itertuples():
        ra_min  = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        ra_max  = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        in_box = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max) & (out_tile == '')
        out_tile[in_box] = r.tile
        out_file[in_box] = r.file
    return out_tile, out_file


def render_cutout(data, wcs, ra, dec, out_png, title):
    try:
        center = SkyCoord(ra * u.deg, dec * u.deg)
        cu = Cutout2D(data, center, size=CUTOUT_AS * u.arcsec, wcs=wcs,
                      mode='partial', fill_value=np.nan)
    except Exception:
        return False
    img = cu.data
    if not np.isfinite(img).any():
        return False
    z = ZScaleInterval()
    try:
        v1, v2 = z.get_limits(img[np.isfinite(img)])
    except Exception:
        return False
    norm = ImageNormalize(vmin=v1, vmax=v2, stretch=AsinhStretch())
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(img, origin='lower', cmap='gray', norm=norm)
    ax.set_xticks([]); ax.set_yticks([])
    n = img.shape[0] / 2
    ax.plot([n, n], [n - 9, n + 9], 'r-', lw=0.8, alpha=0.85)
    ax.plot([n - 9, n + 9], [n, n], 'r-', lw=0.8, alpha=0.85)
    ax.set_title(title, fontsize=8)
    plt.tight_layout(pad=0.05)
    plt.savefig(out_png, dpi=120)
    plt.close()
    return True


def process_band(targets, mission, band, tile_df, slug_prefix):
    """Per-tile-grouped cutout generation.  Returns dict {idx: png_slug}."""
    import time
    paths = {}
    ra  = targets['ra'].values
    dec = targets['dec'].values
    tile, fname = assign_tiles(ra, dec, tile_df, mission, band)
    targets = targets.assign(_tile=tile, _file=fname)
    groups = targets.groupby('_tile')
    for tname, grp in groups:
        ti0 = time.time()
        if not tname:
            for idx in grp.index: paths[idx] = None
            continue
        p = _path_for(mission, band, tname, grp['_file'].iloc[0])
        if p is None or not Path(p).exists():
            for idx in grp.index: paths[idx] = None
            continue
        try:
            with fits.open(p, memmap=True) as h:
                d = None; hdr = None
                for hdu in h:
                    if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                        d = hdu.data; hdr = hdu.header
                        break
                if d is None:
                    for idx in grp.index: paths[idx] = None
                    continue
                wcs = WCS(hdr)
                for r in grp.itertuples():
                    slug = f'{slug_prefix}_{r.Index:04d}_{mission}_{band.replace("-","_")}.png'
                    out_png = THUMB_DIR / slug
                    if out_png.exists():
                        paths[r.Index] = slug
                        continue
                    title = f'{mission} {band}'
                    ok = render_cutout(d, wcs, r.ra, r.dec, out_png, title)
                    paths[r.Index] = slug if ok else None
        except Exception as e:
            print(f'  [tile fail] {mission} {band} {tname}: {e}', flush=True)
            for idx in grp.index: paths[idx] = None
        print(f'  {mission:6s} {band:6s} tile {tname:>5s}: {len(grp)} cutouts in {time.time()-ti0:.1f}s',
              flush=True)
    return paths


def make_html(rows, page_idx, n_pages, kind, all_band_paths, html_path, title_kind):
    rows_html = []
    for r in rows:
        snr_s = f'  SNR={r["snr"]:.1f}' if pd.notna(r.get('snr')) else ''
        miss = ''.join([
            'J' if r.get('missing_jwst')   else '_',
            'H' if r.get('missing_hst')    else '_',
            'E' if r.get('missing_euclid') else '_',
        ])
        head = (f'#{r["seq"]}  ID={r["star_id"]}  RA={r["ra"]:.6f}  Dec={r["dec"]:.6f}'
                f'  mag={r["mag"]:+.2f}{snr_s}  tile={r["tile"]}  miss={miss}')
        cells = []
        for mi, ba in BANDS_SHOW:
            slug = all_band_paths[(mi, ba)].get(r['_pd_idx'])
            if slug:
                cells.append(f'<img src="thumbs/{slug}" title="{mi} {ba}">')
            else:
                cells.append(f'<span>no {mi} {ba}</span>')
        rows_html.append(f"<div class='row'>\n  <h3>{head}</h3>\n  <div class='imgs'>\n    "
                          + '\n    '.join(cells) + '\n  </div>\n</div>')

    nav_links = ['<a href=\"{0:s}_index{1:04d}.html\">&laquo;</a>'.format(kind, max(1, page_idx - 1))]
    for p in range(1, n_pages + 1):
        cls = ' class="active"' if p == page_idx else ''
        nav_links.append(f'<a href="{kind}_index{p:04d}.html"{cls}>{p}</a>')
    nav_links.append(f'<a href="{kind}_index{min(n_pages, page_idx+1):04d}.html">&raquo;</a>')

    html = f"""<!doctype html>
<html><head>
<title>Star orphans: {title_kind} (page {page_idx}/{n_pages})</title>
{CSS}
</head><body>
<div class='header'>
<h2 style='margin:4px 0;'>Star orphans: <code>{title_kind}</code> &mdash; page {page_idx} / {n_pages}</h2>
<div style='font-size:11px;color:#666'>orphans_v5 (HST∩JWST common region, sat_stars_v3 + Gaia + peak + local vetos applied) &middot; sorted by DAO SNR descending</div>
<div class='pagination'>
{chr(10).join(nav_links)}
</div>
</div>
{chr(10).join(rows_html)}
</body></html>"""
    html_path.write_text(html)


def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    orph = pd.read_parquet(OUT / 'orphans_v5.parquet')
    print(f'orphans_v5: {len(orph):,}')

    # 3 candidate kinds — top N per found_in by DAO SNR
    for kind_key, found_in, title_kind in [
        ('hst_only',    'HST F814W',  'hst_only (HST F814W detections, no JWST or Euclid match)'),
        ('jwst_only',   'JWST F115W', 'jwst_only (JWST F115W detections, no HST or Euclid match)'),
        ('euclid_only', 'euclid_vis', 'euclid_only (Euclid VIS detections, no HST or JWST match)'),
    ]:
        # We use missing_X explicitly to ensure no cross-mission match exists
        if kind_key == 'hst_only':
            sub = orph[(orph['found_in'] == 'HST F814W') &
                       (orph['missing_jwst']) & (orph['missing_euclid'])].copy()
        elif kind_key == 'jwst_only':
            sub = orph[(orph['found_in'] == 'JWST F115W') &
                       (orph['missing_hst']) & (orph['missing_euclid'])].copy()
        else:
            sub = orph[(orph['found_in'] == 'Euclid VIS') &
                       (orph['missing_jwst']) & (orph['missing_hst'])].copy()
        if 'snr' not in sub.columns:
            sub['snr'] = np.nan
        sub = sub.sort_values('snr', ascending=False).head(TOP_N_PER_CAT).reset_index(drop=True)
        sub['seq'] = np.arange(1, len(sub) + 1)
        sub['_pd_idx'] = sub.index.values
        print(f'\n=== {kind_key}  n={len(sub):,} ===')

        # Generate thumbnails per band (tile-grouped for speed)
        all_band_paths = {}
        for mi, ba in BANDS_SHOW:
            print(f'  --- {mi} {ba} ---')
            all_band_paths[(mi, ba)] = process_band(sub, mi, ba, tile_df, kind_key)

        # Pages
        n_pages = max(1, int(np.ceil(len(sub) / PER_PAGE)))
        for page_idx in range(1, n_pages + 1):
            page_rows = sub.iloc[(page_idx - 1) * PER_PAGE : page_idx * PER_PAGE].to_dict('records')
            for r in page_rows:
                r.setdefault('snr', np.nan)
            html_path = HTML_ROOT / f'{kind_key}_index{page_idx:04d}.html'
            make_html(page_rows, page_idx, n_pages, kind_key, all_band_paths, html_path, title_kind)
        print(f'  → {n_pages} HTML pages')

    # Build top-level index
    idx_html = HTML_ROOT / 'index.html'
    idx_html.write_text(f"""<!doctype html>
<html><head><title>Star orphans v01 — index</title>{CSS}</head><body>
<div class='header'><h2>Star orphans v01 — categories</h2></div>
<p style='padding:14px'>Each list is sorted by DAO SNR (descending) and limited to {TOP_N_PER_CAT:,} entries.
Sources are restricted to the HST F814W ∩ JWST F115W common pixel coverage (0.54 deg²).
The Gaia bright-star + DAO peak + local-bright veto was applied (saturated_stars_v3).</p>
<ul style='padding:14px'>
  <li><a href='hst_only_index0001.html'>HST-only point sources (no JWST or Euclid match)</a></li>
  <li><a href='jwst_only_index0001.html'>JWST-only point sources (no HST or Euclid match)</a></li>
  <li><a href='euclid_only_index0001.html'>Euclid-only point sources (no HST or JWST match)</a></li>
</ul>
</body></html>""")
    print(f'\nWrote {idx_html}')


if __name__ == '__main__':
    main()
