#!/usr/bin/env python
"""
47_orphan_webpage_v03.py — paginated orphan webpage matching the v01
                            color recipe + galaxy veto applied.

Color recipe (carbon-copy of programs_jwst/03_make_sn_cutouts_jwst.py
              and programs_euclid/03_make_sn_cutouts_euclid.py):

  JWST  RGB (jwst1, jwst2):
      sqrt stretch, vmin = 1e-4, vmax = max of central 10×10 × 0.85
      (with floor at 1.0 if center is dim).
      make_rgb(R, G, B, interval=ManualInterval(vmin, vmax)).
      jwst1: B=F115W G=F150W R=F277W
      jwst2: B=F150W G=F277W R=F444W
  Euclid NISP RGB:
      sqrt stretch, vmin = 1e-5, vmax = max central × 1.2 (or 1.0).
      R=NIR-H G=NIR-J B=NIR-Y
  HST F814W single-band grayscale:
      sqrt stretch, vmin = 1e-4, vmax = max central × 0.85
  Euclid VIS single-band grayscale:
      same recipe as HST.

Source: orphans_v6.parquet (galaxy-vetoed via catalog FWHM/ellipticity +
        survey star/galaxy flags).
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
from astropy.visualization import make_rgb, ManualInterval
from astropy import units as u

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

VERSION = 'v03'
HTML_ROOT = ROOT / 'htmls' / 'orphan_star' / VERSION
HTML_ROOT.mkdir(parents=True, exist_ok=True)

PNG_BASE = Path('/Volumes/exdisk1/data')
PNG_DIRS = {
    'jwst':   PNG_BASE / 'JWST'    / f'COSMOS_v0.8_png_star_orphans_{VERSION}',
    'hst':    PNG_BASE / 'HST'     / f'COSMOS_v2.0_png_star_orphans_{VERSION}',
    'euclid': PNG_BASE / 'Euclid'  / f'COSMOS_DR1_png_star_orphans_{VERSION}',
}
for d in PNG_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
for name, target in PNG_DIRS.items():
    link = HTML_ROOT / name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

CUTOUT_AS = 6.0       # same as v01 (200 px @ 30 mas)
PER_PAGE  = 100
TOP_N_PER_CAT = 600

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


# ---------- v01 rendering recipes ----------
def render_rgb_jwst(b_data, g_data, r_data, png_path, label_lines):
    """Carbon copy of programs_jwst/03_make_sn_cutouts_jwst.py:render_rgb."""
    if b_data is None or g_data is None or r_data is None:
        return False
    minimum = 1e-4
    b = np.sqrt(np.where(b_data > minimum, b_data, minimum))
    g = np.sqrt(np.where(g_data > minimum, g_data, minimum))
    r = np.sqrt(np.where(r_data > minimum, r_data, minimum))
    if not (np.isfinite(b).any() and np.isfinite(g).any() and np.isfinite(r).any()):
        return False

    maximum = 0.001
    for img in (b, g, r):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        val = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if val > maximum:
            maximum = val
    if maximum < 1.0:
        maximum = 1.0
    else:
        maximum *= 0.85

    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111)
    ax.imshow(rgb, origin='lower')
    ax.axis('off')
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color='white',
                fontsize=11, transform=ax.transAxes,
                verticalalignment='top', family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig)
    return True


def render_rgb_euclid_nisp(b_data, g_data, r_data, png_path, label_lines):
    """Carbon copy of programs_euclid/03_make_sn_cutouts_euclid.py:render_rgb
    (B=Y, G=J, R=H).  vmin = 1e-5, vmax = max central × 1.2."""
    if b_data is None or g_data is None or r_data is None:
        return False
    minimum = 1e-5
    b = np.sqrt(np.where(b_data > minimum, b_data, minimum))
    g = np.sqrt(np.where(g_data > minimum, g_data, minimum))
    r = np.sqrt(np.where(r_data > minimum, r_data, minimum))
    if not (np.isfinite(b).any() and np.isfinite(g).any() and np.isfinite(r).any()):
        return False
    maximum = 0.001
    for img in (b, g, r):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        val = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if val > maximum:
            maximum = val
    if maximum < 1.0:
        maximum = 1.0
    else:
        maximum *= 1.2

    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111)
    ax.imshow(rgb, origin='lower')
    ax.axis('off')
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color='white',
                fontsize=11, transform=ax.transAxes,
                verticalalignment='top', family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig)
    return True


def render_gray(data, png_path, label_lines):
    """Single-band greyscale (HST F814W, Euclid VIS)."""
    if data is None:
        return False
    minimum = 1e-4
    bw = np.sqrt(np.where(data > minimum, data, minimum))
    if not np.isfinite(bw).any():
        return False
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if val < 1.0 else val * 0.85

    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111)
    ax.imshow(bw, origin='lower', cmap='gray', vmin=minimum, vmax=maximum)
    ax.axis('off')
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color='white',
                fontsize=11, transform=ax.transAxes,
                verticalalignment='top', family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig)
    return True


# ---------- helpers ----------
def _hst_path(tile_int): return HST_DIR / f'acs_I_030mas_{int(tile_int):03d}_sci.fits'
def _jwst_path(tile, band):
    if tile == 'A10' and band == 'F115W':
        return JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    return JWST_DIR / f'mosaic_nircam_{band.lower()}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'


def assign_tiles(ra, dec, tile_df, mission, band):
    sel = tile_df[(tile_df['mission'] == mission) & (tile_df['band'] == band)]
    out_tile = np.full(len(ra), '', dtype=object)
    out_file = np.full(len(ra), '', dtype=object)
    for r in sel.itertuples():
        ra_min  = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        ra_max  = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        m = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max) & (out_tile == '')
        out_tile[m] = r.tile
        out_file[m] = r.file
    return out_tile, out_file


def _open_2d(path):
    if not Path(path).exists(): return None, None
    with fits.open(path, memmap=True) as h:
        for hdu in h:
            if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                return np.asarray(hdu.data), hdu.header.copy()
    return None, None


def cut_array(data, wcs, ra, dec, size_as=CUTOUT_AS):
    try:
        c = SkyCoord(ra * u.deg, dec * u.deg)
        cu = Cutout2D(data, c, size=size_as * u.arcsec, wcs=wcs, mode='partial', fill_value=np.nan)
        return cu.data
    except Exception:
        return None


# ---------- per-band tile-grouped builders ----------
def process_jwst_rgb(targets, tile_df, kind_key,
                     bands_b, bands_g, bands_r, png_dir, suffix):
    import time
    paths = {}
    ra = targets['ra'].values; dec = targets['dec'].values
    tile, _ = assign_tiles(ra, dec, tile_df, 'JWST', bands_g)
    targets = targets.assign(_tile=tile)
    for tname, grp in targets.groupby('_tile'):
        if not tname:
            for idx in grp.index: paths[idx] = None
            continue
        t0 = time.time()
        d = {}; w = {}
        for band in (bands_b, bands_g, bands_r):
            data, hdr = _open_2d(_jwst_path(tname, band))
            d[band] = data; w[band] = WCS(hdr) if hdr is not None else None
        for r in grp.itertuples():
            slug = f'star_{kind_key}_{r.Index:04d}_{suffix}.png'
            out_png = png_dir / slug
            if out_png.exists():
                paths[r.Index] = slug; continue
            cuts = {}
            for band in (bands_b, bands_g, bands_r):
                cuts[band] = cut_array(d[band], w[band], r.ra, r.dec) if d[band] is not None and w[band] is not None else None
            label = [f'ID={r.star_id}', f'{bands_b}/{bands_g}/{bands_r}']
            ok = render_rgb_jwst(cuts[bands_b], cuts[bands_g], cuts[bands_r],
                                  out_png, label)
            paths[r.Index] = slug if ok else None
        print(f'  jwst_{suffix:<6} tile {tname:>5s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
              flush=True)
    return paths


def process_hst(targets, tile_df, kind_key, png_dir):
    import time
    paths = {}
    ra = targets['ra'].values; dec = targets['dec'].values
    tile, _ = assign_tiles(ra, dec, tile_df, 'HST', 'F814W')
    targets = targets.assign(_tile=tile)
    for tname, grp in targets.groupby('_tile'):
        if not tname:
            for idx in grp.index: paths[idx] = None; continue
        t0 = time.time()
        data, hdr = _open_2d(_hst_path(tname))
        wcs = WCS(hdr) if hdr is not None else None
        for r in grp.itertuples():
            slug = f'star_{kind_key}_{r.Index:04d}_hst.png'
            out_png = png_dir / slug
            if out_png.exists(): paths[r.Index] = slug; continue
            img = cut_array(data, wcs, r.ra, r.dec) if data is not None else None
            label = [f'ID={r.star_id}', 'HST F814W']
            ok = render_gray(img, out_png, label)
            paths[r.Index] = slug if ok else None
        print(f'  hst   tile {tname:>5s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
              flush=True)
    return paths


def process_euclid(targets, tile_df, kind_key, png_dir):
    import time
    paths_vis  = {}
    paths_nisp = {}
    ra = targets['ra'].values; dec = targets['dec'].values
    tile, _ = assign_tiles(ra, dec, tile_df, 'Euclid', 'VIS')
    targets = targets.assign(_tile=tile)
    vis_files  = {r.tile: r.file for r in tile_df[(tile_df['mission']=='Euclid') & (tile_df['band']=='VIS')].itertuples()}
    nisp_files = {b: {r.tile: r.file for r in tile_df[(tile_df['mission']=='Euclid') & (tile_df['band']==b)].itertuples()}
                  for b in ('NIR-Y', 'NIR-J', 'NIR-H')}
    for tname, grp in targets.groupby('_tile'):
        if not tname:
            for idx in grp.index:
                paths_vis[idx] = None; paths_nisp[idx] = None
            continue
        t0 = time.time()
        d_vis, h_vis = _open_2d(EUCLID_DIR / vis_files[tname])
        w_vis = WCS(h_vis) if h_vis is not None else None
        d_y, h_y = _open_2d(EUCLID_DIR / nisp_files['NIR-Y'].get(tname, ''))
        d_j, h_j = _open_2d(EUCLID_DIR / nisp_files['NIR-J'].get(tname, ''))
        d_h, h_h = _open_2d(EUCLID_DIR / nisp_files['NIR-H'].get(tname, ''))
        w_y = WCS(h_y) if h_y is not None else None
        w_j = WCS(h_j) if h_j is not None else None
        w_h = WCS(h_h) if h_h is not None else None
        for r in grp.itertuples():
            v_slug = f'star_{kind_key}_{r.Index:04d}_euclid_vis.png'
            n_slug = f'star_{kind_key}_{r.Index:04d}_euclid_nisp.png'
            v_png  = png_dir / v_slug
            n_png  = png_dir / n_slug
            if not v_png.exists():
                vis_img = cut_array(d_vis, w_vis, r.ra, r.dec) if d_vis is not None else None
                ok = render_gray(vis_img, v_png, [f'ID={r.star_id}', 'Euclid VIS'])
                paths_vis[r.Index] = v_slug if ok else None
            else:
                paths_vis[r.Index] = v_slug
            if not n_png.exists():
                Y = cut_array(d_y, w_y, r.ra, r.dec) if d_y is not None else None
                J = cut_array(d_j, w_j, r.ra, r.dec) if d_j is not None else None
                H = cut_array(d_h, w_h, r.ra, r.dec) if d_h is not None else None
                ok = render_rgb_euclid_nisp(Y, J, H, n_png, [f'ID={r.star_id}', 'Y/J/H'])
                paths_nisp[r.Index] = n_slug if ok else None
            else:
                paths_nisp[r.Index] = n_slug
        print(f'  euclid tile {tname:>10s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
              flush=True)
    return paths_vis, paths_nisp


# ---------- HTML ----------
def write_page(rows, page_idx, n_pages, kind, title, paths):
    rows_html = []
    for r in rows:
        idx = r['_pd_idx']
        snr_s = f"  SNR={r['snr']:.1f}" if pd.notna(r.get('snr')) else ''
        miss = ''.join([
            'J' if r.get('missing_jwst')   else '_',
            'H' if r.get('missing_hst')    else '_',
            'E' if r.get('missing_euclid') else '_',
        ])
        head = (f"#{r['seq']}  ID={r['star_id']}  RA={r['ra']:.6f}  Dec={r['dec']:.6f}"
                f"  mag_DAO={r['mag']:+.2f}{snr_s}  tile={r['tile']}  miss[J/H/E]={miss}")

        def img(rel, t):
            return f"<img src='{rel}' title='{t}'>" if rel else f"<span>no {t}</span>"
        cells = [
            img(paths['jwst1'].get(idx)  and f"./jwst/{paths['jwst1'][idx]}",   'JWST F115/F150/F277'),
            img(paths['jwst2'].get(idx)  and f"./jwst/{paths['jwst2'][idx]}",   'JWST F150/F277/F444'),
            img(paths['hst'].get(idx)    and f"./hst/{paths['hst'][idx]}",       'HST F814W'),
            img(paths['vis'].get(idx)    and f"./euclid/{paths['vis'][idx]}",    'Euclid VIS'),
            img(paths['nisp'].get(idx)   and f"./euclid/{paths['nisp'][idx]}",   'Euclid NISP YJH'),
        ]
        rows_html.append(
            "<div class='row'>\n"
            f"  <h3>{head}</h3>\n"
            "  <div class='imgs'>\n    "
            + "\n    ".join(cells) +
            "\n  </div>\n</div>"
        )

    pag = [f"<a href='{kind}_index{max(1,page_idx-1):04d}.html'>&laquo;</a>"]
    for p in range(1, n_pages + 1):
        cls = " class='active'" if p == page_idx else ""
        pag.append(f"<a href='{kind}_index{p:04d}.html'{cls}>{p}</a>")
    pag.append(f"<a href='{kind}_index{min(n_pages,page_idx+1):04d}.html'>&raquo;</a>")

    html = f"""<!doctype html>
<html><head>
<title>Star orphans v03: {kind} (page {page_idx}/{n_pages})</title>
{CSS}
</head><body>
<div class='header'>
<h2 style='margin:4px 0;'>Star orphans v03: <code>{kind}</code> &mdash; page {page_idx} / {n_pages}</h2>
<div style='font-size:11px;color:#666'>
{title}<br>
HST F814W ∩ JWST F115W common region (0.54 deg²); saturated_stars_v3 + catalog galaxy veto (FWHM, ellipticity, survey star/galaxy flags) applied. DAO mag is instrumental.</div>
<div class='pagination'>
{chr(10).join(pag)}
</div>
</div>
{chr(10).join(rows_html)}
</body></html>"""
    (HTML_ROOT / f'{kind}_index{page_idx:04d}.html').write_text(html)


def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    orph = pd.read_parquet(OUT / 'orphans_v6.parquet')
    print(f'orphans_v6 (galaxy-vetoed): {len(orph):,}')

    for kind_key, found_in, title_long in [
        ('hst_only',    'HST F814W',  'HST F814W detections with no JWST and no Euclid match (and no catalog galaxy match)'),
        ('jwst_only',   'JWST F115W', 'JWST F115W detections with no HST and no Euclid match (galaxy-vetoed)'),
        ('euclid_only', 'Euclid VIS', 'Euclid VIS detections with no HST and no JWST match (galaxy-vetoed)'),
    ]:
        if kind_key == 'hst_only':
            sub = orph[(orph['found_in'] == 'HST F814W') &
                       (orph['missing_jwst']) & (orph['missing_euclid'])].copy()
        elif kind_key == 'jwst_only':
            sub = orph[(orph['found_in'] == 'JWST F115W') &
                       (orph['missing_hst']) & (orph['missing_euclid'])].copy()
        else:
            sub = orph[(orph['found_in'] == 'Euclid VIS') &
                       (orph['missing_jwst']) & (orph['missing_hst'])].copy()
        sub = sub.sort_values('snr', ascending=False).head(TOP_N_PER_CAT).reset_index(drop=True)
        sub['seq'] = np.arange(1, len(sub) + 1)
        sub['_pd_idx'] = sub.index.values
        print(f'\n=== {kind_key} (n={len(sub):,}) ===')

        paths = {}
        print(' building jwst1 (B=F115, G=F150, R=F277)')
        paths['jwst1'] = process_jwst_rgb(sub, tile_df, kind_key,
                                          'F115W', 'F150W', 'F277W',
                                          PNG_DIRS['jwst'], 'jwst1')
        print(' building jwst2 (B=F150, G=F277, R=F444)')
        paths['jwst2'] = process_jwst_rgb(sub, tile_df, kind_key,
                                          'F150W', 'F277W', 'F444W',
                                          PNG_DIRS['jwst'], 'jwst2')
        print(' building hst F814W')
        paths['hst']  = process_hst(sub, tile_df, kind_key, PNG_DIRS['hst'])
        print(' building euclid vis + nisp')
        paths['vis'], paths['nisp'] = process_euclid(sub, tile_df, kind_key, PNG_DIRS['euclid'])

        n_pages = max(1, int(np.ceil(len(sub) / PER_PAGE)))
        for page_idx in range(1, n_pages + 1):
            page_rows = sub.iloc[(page_idx-1)*PER_PAGE: page_idx*PER_PAGE].to_dict('records')
            write_page(page_rows, page_idx, n_pages, kind_key, title_long, paths)
        print(f' → {n_pages} HTML pages')

    idx = HTML_ROOT / 'index.html'
    idx.write_text(f"""<!doctype html>
<html><head><title>Star orphans — v03 index</title>{CSS}</head><body>
<div class='header'><h2>Star orphans v03 — categories</h2></div>
<p style='padding:14px'>Source = <code>orphans_v6.parquet</code>.
HST F814W ∩ JWST F115W common pixel coverage; saturated_stars_v3 vetoed;
<b>catalog galaxy veto</b> (FWHM + ellipticity + Euclid PHZ + ACS mu_class +
COSMOS2020 lp_type + CW flag_star) applied.
Sorted by DAO SNR; top {TOP_N_PER_CAT:,} per category.
Cutouts use the v01 sqrt+ManualInterval recipe.</p>
<ul style='padding:14px'>
  <li><a href='hst_only_index0001.html'>HST-only point sources</a></li>
  <li><a href='jwst_only_index0001.html'>JWST-only point sources</a></li>
  <li><a href='euclid_only_index0001.html'>Euclid-only point sources</a></li>
</ul>
</body></html>""")
    print(f'\nWrote {idx}')


if __name__ == '__main__':
    main()
