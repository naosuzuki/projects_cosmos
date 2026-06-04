#!/usr/bin/env python
"""
49_orphan_v04.py — orphan STAR list under the user's strict rule:

    A valid orphan STAR candidate must have HST F814W AND Euclid VIS
    cross-match (i.e. missing_hst == False AND missing_euclid == False).
    The "orphan" character: missing JWST F115W match.

Pipeline:
  1. Read orphans_v6 (HST∩JWST common region + saturated_stars_v3 vetoed
                       + catalog galaxy veto applied).
  2. Strict filter: keep only rows with missing_hst==False AND
                     missing_euclid==False  (=> must include HST and
                     Euclid data).  These are necessarily missing JWST
                     (otherwise they wouldn't be orphans).
  3. Image-extent galaxy/artifact veto using HST F814W second moments
     (data_properties on a 3″ cutout): drop rows with fit FWHM > 1.6×PSF
     OR ellipticity > 0.30.
  4. Dedup by position (sources reach v6 twice — once via found_in=
     Euclid VIS and once via found_in=HST F814W).
  5. Generate v01-style paginated webpage at htmls/orphan_star/v04/
     using the original sqrt+ManualInterval color recipe.

Outputs:
  csvfiles_star/orphans_v8.parquet          strict rule applied
  csvfiles_star/orphans_v9.parquet          + image extent veto
  htmls/orphan_star/v04/<kind>_indexNNNN.html
  symlink-managed thumbnail dirs under /Volumes/exdisk1/data/.../...v04/
"""
from __future__ import annotations
import warnings, time
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
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_rgb, ManualInterval
from astropy import units as u
from photutils.morphology import data_properties

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
VERSION = 'v04'
HTML_ROOT = ROOT / 'htmls' / 'orphan_star' / VERSION
HTML_ROOT.mkdir(parents=True, exist_ok=True)

PNG_BASE = Path('/Volumes/exdisk1/data')
PNG_DIRS = {
    'jwst':   PNG_BASE / 'JWST'   / f'COSMOS_v0.8_png_star_orphans_{VERSION}',
    'hst':    PNG_BASE / 'HST'    / f'COSMOS_v2.0_png_star_orphans_{VERSION}',
    'euclid': PNG_BASE / 'Euclid' / f'COSMOS_DR1_png_star_orphans_{VERSION}',
}
for d in PNG_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
for name, target in PNG_DIRS.items():
    link = HTML_ROOT / name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target)

HST_DIR  = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

CUTOUT_AS = 6.0
PER_PAGE  = 100
TOP_N     = 600

HST_PSF_AS = 0.134
HST_FWHM_GAL_AS = 1.6 * HST_PSF_AS
HST_ELLIP_GAL   = 0.30
HST_PIX_AS = 0.030

CSS = """<style>
body{font-family:serif;margin:0;padding:0;background:#fafafa}
.header{position:sticky;top:0;background:white;border-bottom:1px solid #ddd;padding:8px;z-index:10}
.pagination{display:inline-block}
.pagination a{color:black;float:left;padding:6px 12px;text-decoration:none;
  transition:background-color .2s;border:1px solid #ccc;margin:1px}
.pagination a.active{background-color:#4CAF50;color:white}
.pagination a:hover:not(.active){background-color:#ddd}
.row{padding:4px 8px;border-bottom:1px solid #eee}
.row h3{margin:6px 0 4px 0;font-size:14px;color:#222}
.imgs img{width:18%;margin:0 0.5%;vertical-align:top}
.imgs span{display:inline-block;width:18%;margin:0 0.5%;text-align:center;
  vertical-align:top;padding:60px 0;color:#888;font-style:italic;background:#f0f0f0}
</style>"""


# ---- v01 render recipes ----
def render_rgb_jwst(b, g, r, png_path, labels):
    if b is None or g is None or r is None: return False
    minimum = 1e-4
    B = np.sqrt(np.where(b > minimum, b, minimum))
    G = np.sqrt(np.where(g > minimum, g, minimum))
    R = np.sqrt(np.where(r > minimum, r, minimum))
    if not (np.isfinite(B).any() and np.isfinite(G).any() and np.isfinite(R).any()):
        return False
    maximum = 0.001
    for img in (B, G, R):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        v = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if v > maximum: maximum = v
    maximum = 1.0 if maximum < 1.0 else maximum * 0.85
    rgb = make_rgb(R, G, B, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin='lower'); ax.axis('off')
    for k, t in enumerate(labels):
        ax.text(0.03, 0.95 - 0.07*k, t, color='white', fontsize=11,
                transform=ax.transAxes, verticalalignment='top',
                family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig); return True


def render_rgb_euclid(b, g, r, png_path, labels):
    if b is None or g is None or r is None: return False
    minimum = 1e-5
    B = np.sqrt(np.where(b > minimum, b, minimum))
    G = np.sqrt(np.where(g > minimum, g, minimum))
    R = np.sqrt(np.where(r > minimum, r, minimum))
    if not (np.isfinite(B).any() and np.isfinite(G).any() and np.isfinite(R).any()):
        return False
    maximum = 0.001
    for img in (B, G, R):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        v = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if v > maximum: maximum = v
    maximum = 1.0 if maximum < 1.0 else maximum * 1.2
    rgb = make_rgb(R, G, B, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin='lower'); ax.axis('off')
    for k, t in enumerate(labels):
        ax.text(0.03, 0.95 - 0.07*k, t, color='white', fontsize=11,
                transform=ax.transAxes, verticalalignment='top',
                family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig); return True


def render_gray(data, png_path, labels):
    if data is None: return False
    minimum = 1e-4
    bw = np.sqrt(np.where(data > minimum, data, minimum))
    if not np.isfinite(bw).any(): return False
    ny, nx = bw.shape; cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    v = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if v < 1.0 else v * 0.85
    fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(111)
    ax.imshow(bw, origin='lower', cmap='gray', vmin=minimum, vmax=maximum)
    ax.axis('off')
    for k, t in enumerate(labels):
        ax.text(0.03, 0.95 - 0.07*k, t, color='white', fontsize=11,
                transform=ax.transAxes, verticalalignment='top',
                family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig); return True


# ---- helpers ----
def _hst_path(tile): return HST_DIR / f'acs_I_030mas_{int(tile):03d}_sci.fits'
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
        out_tile[m] = r.tile; out_file[m] = r.file
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
        return Cutout2D(data, c, size=size_as*u.arcsec, wcs=wcs, mode='partial', fill_value=np.nan).data
    except Exception:
        return None


def measure_hst_extent(data, wcs, ra, dec):
    """Return (fwhm_as, ellip) from HST F814W cutout."""
    img = cut_array(data, wcs, ra, dec, size_as=3.0)
    if img is None or not np.isfinite(img).any():
        return np.nan, np.nan
    finite = img[np.isfinite(img)]
    _, med, std = sigma_clipped_stats(finite, sigma=3.0, maxiters=3)
    sub = img - med
    ny, nx = sub.shape; cy, cx = ny/2.0, nx/2.0
    r_pix = 0.6 / HST_PIX_AS
    y, x = np.indices(sub.shape)
    in_aper = (x - cx)**2 + (y - cy)**2 <= r_pix**2
    thr = 2.0 * std
    mask = in_aper & np.isfinite(sub) & (sub > thr)
    if mask.sum() < 5:
        return np.nan, np.nan
    try:
        pos = np.nan_to_num(sub)
        cat = data_properties(pos, mask=mask)
        a = float(cat.semimajor_sigma.value) * HST_PIX_AS
        b = float(cat.semiminor_sigma.value) * HST_PIX_AS
        ellip = 1.0 - b / max(a, 1e-9)
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(max(a*b, 1e-12))
        return fwhm, ellip
    except Exception:
        return np.nan, np.nan


# ---- main ----
def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    df = pd.read_parquet(OUT / 'orphans_v6.parquet')
    print(f'orphans_v6: {len(df):,}')

    # ---- Step 1: strict rule ----
    # Valid orphan STAR: missing_hst==False AND missing_euclid==False (must have HST and Euclid).
    # By geometry this means the source has HST + Euclid cross-matches; the
    # "orphan" character is that JWST F115W cross-match is missing.
    strict = df[(~df['missing_hst']) & (~df['missing_euclid'])].copy()
    print(f'\nStrict rule (must have HST AND Euclid): {len(strict):,}')

    # Dedup by RA/Dec (sources reach v6 via both Euclid-VIS and HST-F814W
    # found_in lines).  Round to 0.05" precision.
    strict['_ra_r']  = (strict['ra']  * 72000).round() / 72000
    strict['_dec_r'] = (strict['dec'] * 72000).round() / 72000
    strict = strict.sort_values('snr', ascending=False).drop_duplicates(
        ['_ra_r', '_dec_r'], keep='first').drop(columns=['_ra_r', '_dec_r'])
    print(f'After dedup (0.05″): {len(strict):,}')

    strict.to_parquet(OUT / 'orphans_v8.parquet', index=False)

    # ---- Step 2: image-extent veto (HST F814W) ----
    print('\nMeasuring HST F814W image extent for galaxy/artifact veto...')
    strict['_hst_tile'], _ = assign_tiles(strict['ra'].values, strict['dec'].values,
                                          tile_df, 'HST', 'F814W')
    fwhm_arr = np.full(len(strict), np.nan)
    ellip_arr = np.full(len(strict), np.nan)
    t0 = time.time()
    for tname, grp in strict.groupby('_hst_tile'):
        if not tname:
            continue
        d, h = _open_2d(_hst_path(tname))
        wcs = WCS(h) if h is not None else None
        for r in grp.itertuples():
            f, e = measure_hst_extent(d, wcs, r.ra, r.dec)
            i = strict.index.get_loc(r.Index)
            fwhm_arr[i] = f
            ellip_arr[i] = e
        print(f'  HST tile {tname:>5s}: {len(grp)} measurements   ({time.time()-t0:.1f}s)',
              flush=True)

    strict['img_fwhm_hst_as'] = fwhm_arr
    strict['img_ellip_hst']   = ellip_arr
    veto_img = ((strict['img_fwhm_hst_as'] > HST_FWHM_GAL_AS) |
                (strict['img_ellip_hst']   > HST_ELLIP_GAL))
    veto_img = veto_img.fillna(False)
    print(f'\nImage-extent veto: {int(veto_img.sum()):,} of {len(strict):,} '
          f'flagged as galaxy/artifact')

    v9 = strict[~veto_img].copy()
    v9.to_parquet(OUT / 'orphans_v9.parquet', index=False)
    print(f'orphans_v9 (strict + image-extent): {len(v9):,}')

    # ---- Step 3: build webpage v04 ----
    # All v9 orphans have HST+Euclid detections; the missing band is JWST.
    sub = v9.sort_values('snr', ascending=False).head(TOP_N).reset_index(drop=True)
    sub['seq']     = np.arange(1, len(sub) + 1)
    sub['_pd_idx'] = sub.index.values
    print(f'\nWebpage v04 will show top {len(sub):,} valid orphans')

    # Use HST F814W as the position reference (most stable; nearly all sources
    # are detected there).  But the original `found_in` may be Euclid VIS for
    # some — those still pass strict rule.

    # Tile assignment for cutouts
    for mi, ba in [('HST','F814W'), ('JWST','F115W'),
                    ('JWST','F150W'), ('JWST','F277W'), ('JWST','F444W'),
                    ('Euclid','VIS'), ('Euclid','NIR-Y'),
                    ('Euclid','NIR-J'), ('Euclid','NIR-H')]:
        col = f'_tile_{mi}_{ba.replace("-","_")}'
        sub[col], _ = assign_tiles(sub['ra'].values, sub['dec'].values, tile_df, mi, ba)

    # Build cutouts band by band (tile-grouped)
    paths = {}

    def _grouped(targets, mi, ba, builder, suffix):
        col = f'_tile_{mi}_{ba.replace("-","_")}'
        out = {}
        for tname, grp in targets.groupby(col):
            if not tname:
                for idx in grp.index: out[idx] = None
                continue
            t0 = time.time()
            p = (_hst_path(tname) if mi == 'HST' else
                 (_jwst_path(tname, ba) if mi == 'JWST' else None))
            if mi == 'Euclid':
                # need filename from tile_df
                row = tile_df[(tile_df['mission']=='Euclid') &
                              (tile_df['band']==ba) &
                              (tile_df['tile']==tname)]
                p = EUCLID_DIR / row['file'].iloc[0] if len(row) else None
            data, hdr = _open_2d(p) if p is not None else (None, None)
            wcs = WCS(hdr) if hdr is not None else None
            for r in grp.itertuples():
                slug = f'star_orph_{r.Index:04d}_{suffix}.png'
                out_png = PNG_DIRS[{'HST':'hst','JWST':'jwst','Euclid':'euclid'}[mi]] / slug
                out[r.Index] = builder(data, wcs, r, out_png, slug)
            print(f'  {mi:6s} {ba:6s} tile {str(tname):>10s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
                  flush=True)
        return out

    # HST single-band
    def b_hst(data, wcs, r, out_png, slug):
        if out_png.exists(): return slug
        img = cut_array(data, wcs, r.ra, r.dec) if data is not None else None
        ok = render_gray(img, out_png, [f'ID={r.star_id}', 'HST F814W'])
        return slug if ok else None

    print('\nBuilding HST F814W thumbnails:')
    paths['hst'] = _grouped(sub, 'HST', 'F814W', b_hst, 'hst')

    # JWST RGB jwst1: F115/F150/F277  jwst2: F150/F277/F444 — three-band, tile-grouped
    def build_jwst_rgb(b_band, g_band, r_band, suffix):
        result = {}
        col_b = f'_tile_JWST_{b_band}'
        # All three bands share JWST tiling
        for tname, grp in sub.groupby(col_b):
            if not tname:
                for idx in grp.index: result[idx] = None
                continue
            t0 = time.time()
            data = {}; wcs_ = {}
            for band in (b_band, g_band, r_band):
                d, h = _open_2d(_jwst_path(tname, band))
                data[band] = d; wcs_[band] = WCS(h) if h is not None else None
            for r in grp.itertuples():
                slug = f'star_orph_{r.Index:04d}_{suffix}.png'
                out_png = PNG_DIRS['jwst'] / slug
                if out_png.exists():
                    result[r.Index] = slug; continue
                cuts = {}
                for band in (b_band, g_band, r_band):
                    cuts[band] = (cut_array(data[band], wcs_[band], r.ra, r.dec)
                                   if data[band] is not None and wcs_[band] is not None
                                   else None)
                labels = [f'ID={r.star_id}', f'{b_band}/{g_band}/{r_band}']
                ok = render_rgb_jwst(cuts[b_band], cuts[g_band], cuts[r_band],
                                      out_png, labels)
                result[r.Index] = slug if ok else None
            print(f'  jwst_{suffix:<6} tile {tname:>5s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
                  flush=True)
        return result

    print('\nBuilding JWST RGB jwst1 (F115/F150/F277):')
    paths['jwst1'] = build_jwst_rgb('F115W', 'F150W', 'F277W', 'jwst1')
    print('Building JWST RGB jwst2 (F150/F277/F444):')
    paths['jwst2'] = build_jwst_rgb('F150W', 'F277W', 'F444W', 'jwst2')

    # Euclid VIS single + NISP RGB
    def b_euvis(data, wcs, r, out_png, slug):
        if out_png.exists(): return slug
        img = cut_array(data, wcs, r.ra, r.dec) if data is not None else None
        ok = render_gray(img, out_png, [f'ID={r.star_id}', 'Euclid VIS'])
        return slug if ok else None

    print('\nBuilding Euclid VIS thumbnails:')
    paths['vis'] = _grouped(sub, 'Euclid', 'VIS', b_euvis, 'euclid_vis')

    # NISP RGB: need 3 bands per tile
    print('\nBuilding Euclid NISP RGB (Y/J/H):')
    nisp_paths = {}
    col = f'_tile_Euclid_NIR_J'
    for tname, grp in sub.groupby(col):
        if not tname:
            for idx in grp.index: nisp_paths[idx] = None
            continue
        t0 = time.time()
        data = {}; wcs_ = {}
        for band in ('NIR-Y', 'NIR-J', 'NIR-H'):
            row = tile_df[(tile_df['mission']=='Euclid') &
                          (tile_df['band']==band) &
                          (tile_df['tile']==tname)]
            if len(row):
                d, h = _open_2d(EUCLID_DIR / row['file'].iloc[0])
                data[band] = d; wcs_[band] = WCS(h) if h is not None else None
            else:
                data[band] = None; wcs_[band] = None
        for r in grp.itertuples():
            slug = f'star_orph_{r.Index:04d}_euclid_nisp.png'
            out_png = PNG_DIRS['euclid'] / slug
            if out_png.exists():
                nisp_paths[r.Index] = slug; continue
            cuts = {}
            for band in ('NIR-Y', 'NIR-J', 'NIR-H'):
                cuts[band] = (cut_array(data[band], wcs_[band], r.ra, r.dec)
                               if data[band] is not None and wcs_[band] is not None
                               else None)
            ok = render_rgb_euclid(cuts['NIR-Y'], cuts['NIR-J'], cuts['NIR-H'],
                                    out_png, [f'ID={r.star_id}', 'Y/J/H'])
            nisp_paths[r.Index] = slug if ok else None
        print(f'  euclid_nisp tile {tname:>10s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
              flush=True)
    paths['nisp'] = nisp_paths

    # ---- Write HTML ----
    def write_page(rows, page_idx, n_pages):
        rows_html = []
        for r in rows:
            idx = r['_pd_idx']
            head = (f"#{r['seq']}  ID={r['star_id']}  RA={r['ra']:.6f}  Dec={r['dec']:.6f}"
                    f"  mag_DAO={r['mag']:+.2f}  SNR={r['snr']:.0f}  "
                    f"HST_fwhm={r.get('img_fwhm_hst_as',float('nan')):.3f}″  "
                    f"HST_ellip={r.get('img_ellip_hst',float('nan')):.2f}")
            def img(rel, t):
                return f"<img src='{rel}' title='{t}'>" if rel else f"<span>no {t}</span>"
            cells = [
                img(paths['jwst1'].get(idx) and f"./jwst/{paths['jwst1'][idx]}",   'JWST F115/F150/F277'),
                img(paths['jwst2'].get(idx) and f"./jwst/{paths['jwst2'][idx]}",   'JWST F150/F277/F444'),
                img(paths['hst'].get(idx)   and f"./hst/{paths['hst'][idx]}",       'HST F814W'),
                img(paths['vis'].get(idx)   and f"./euclid/{paths['vis'][idx]}",    'Euclid VIS'),
                img(paths['nisp'].get(idx)  and f"./euclid/{paths['nisp'][idx]}",   'Euclid NISP YJH'),
            ]
            rows_html.append(
                "<div class='row'>\n"
                f"  <h3>{head}</h3>\n"
                "  <div class='imgs'>\n    " + "\n    ".join(cells) +
                "\n  </div>\n</div>"
            )
        pag = [f"<a href='missing_jwst_index{max(1,page_idx-1):04d}.html'>&laquo;</a>"]
        for p in range(1, n_pages + 1):
            cls = " class='active'" if p == page_idx else ""
            pag.append(f"<a href='missing_jwst_index{p:04d}.html'{cls}>{p}</a>")
        pag.append(f"<a href='missing_jwst_index{min(n_pages,page_idx+1):04d}.html'>&raquo;</a>")

        html = f"""<!doctype html><html><head>
<title>Star orphans v04: missing_jwst (page {page_idx}/{n_pages})</title>
{CSS}</head><body>
<div class='header'>
<h2 style='margin:4px 0;'>Star orphans v04: <code>missing_jwst</code> &mdash; page {page_idx} / {n_pages}</h2>
<div style='font-size:11px;color:#666'>
Strict rule: HST F814W and Euclid VIS BOTH have cross-matched detections.
The "orphan" character: no JWST F115W cross-match within the brightness-aware radius.
HST F814W image-based extent veto (FWHM &gt; {HST_FWHM_GAL_AS:.3f}″ or ellip &gt; {HST_ELLIP_GAL}) applied.
Saturated stars (DAO sat + Gaia bright + peak + local) vetoed.  Catalog galaxy veto applied.
DAO mag is instrumental; sorted by DAO SNR.</div>
<div class='pagination'>{chr(10).join(pag)}</div>
</div>
{chr(10).join(rows_html)}
</body></html>"""
        (HTML_ROOT / f'missing_jwst_index{page_idx:04d}.html').write_text(html)

    n_pages = max(1, int(np.ceil(len(sub) / PER_PAGE)))
    for page_idx in range(1, n_pages + 1):
        rows = sub.iloc[(page_idx-1)*PER_PAGE: page_idx*PER_PAGE].to_dict('records')
        write_page(rows, page_idx, n_pages)
    print(f'\n→ {n_pages} HTML pages')

    idx_html = HTML_ROOT / 'index.html'
    idx_html.write_text(f"""<!doctype html><html><head>
<title>Star orphans — v04 index</title>{CSS}</head><body>
<div class='header'><h2>Star orphans v04 — strict rule</h2></div>
<p style='padding:14px'>
<b>Strict rule</b>: HST F814W AND Euclid VIS must BOTH have cross-matched
detections (per the user's specification of 2026-05-26).
"Orphan" character: missing JWST F115W cross-match.
HST F814W image-based extent veto removes galaxies/artifacts whose catalog
match was missed.
</p>
<ul style='padding:14px'>
  <li><a href='missing_jwst_index0001.html'>missing_jwst (HST + Euclid detected, JWST missing)</a> &mdash; <b>{len(v9):,}</b> candidates, top {len(sub):,} shown</li>
</ul>
<p style='padding:14px;font-size:12px;color:#666'>
Source files: <code>csvfiles_star/orphans_v8.parquet</code> (strict rule applied),
<code>csvfiles_star/orphans_v9.parquet</code> (+ image-extent veto).
Earlier versions: <a href='../v01/'>v01</a>, <a href='../v02/'>v02</a>,
<a href='../v03/'>v03</a>.</p>
</body></html>""")
    print(f'Wrote {idx_html}')


if __name__ == '__main__':
    main()
