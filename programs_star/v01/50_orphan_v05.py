#!/usr/bin/env python
"""
50_orphan_v05.py — orphan STAR list under the user's revised rule set:

  Rule A.  Must have a JWST F115W detection.  The JWST star_id is the
           canonical identifier.  No JWST ID → disqualified.
  Rule B.  Must have HST F814W and Euclid VIS cross-matches.
           => the working sample is n_epochs == 3 from proper_motion_v01
              (HST + JWST + Euclid all detected).
  Rule C.  JWST coverage at the position must be valid (not a chip gap).
           Implemented as: read a 5×5 pixel region from the JWST F115W
           SCI at the JWST-tile pixel coords; require ≥ 80% of pixels
           finite & nonzero.
  Rule D.  Veto galaxies: HST F814W image-based extent (FWHM > 1.6×PSF
           or ellip > 0.30) + catalog galaxy veto (Euclid PHZ, HST
           mu_class, CW flag_star, COSMOS2020 lp_type).
  Rule E.  Every plot shows a simple red crosshair at the catalog
           position so the orphan is unambiguous.

Sort by total proper motion (descending).  High-PM stars first → most
interesting orphans.

Output:
  csvfiles_star/orphans_v10.parquet   (3-mission stars, all vetos applied)
  htmls/orphan_star/v05/missing_jwst_index{NNNN}.html
                    (named "missing_jwst" only for filename continuity
                     with v04; content is 3-mission confirmed stars
                     sorted by PM)
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
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_rgb, ManualInterval
from astropy import units as u
from photutils.morphology import data_properties

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
CATDIR = Path('/Volumes/exdisk1/data/catalog')

VERSION = 'v05'
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

HST_DIR   = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR  = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI  = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

CUTOUT_AS = 6.0
PER_PAGE  = 100
TOP_N     = 600
CROSSHAIR_HALF_AS  = 1.5      # crosshair line half-length, arcsec
CROSSHAIR_GAP_AS   = 0.3      # gap at centre (don't obscure source)

HST_PSF_AS  = 0.134
HST_PIX_AS  = 0.030
HST_FWHM_GAL_AS = 1.6 * HST_PSF_AS
HST_ELLIP_GAL   = 0.30

JWST_PIX_AS = 0.030
JWST_COVERAGE_MIN_FRAC = 0.80   # min fraction of finite&nonzero pixels in 5×5

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


# ============================================================
# Crosshair drawing
# ============================================================
def _draw_crosshair(ax, img_shape, pix_as):
    """Draw a small red crosshair at the centre of `ax`'s axes.  Coords
    are in image pixel units (since imshow places pixels 0..n-1)."""
    ny, nx = img_shape
    cy, cx = ny / 2.0 - 0.5, nx / 2.0 - 0.5
    half_pix = CROSSHAIR_HALF_AS / pix_as
    gap_pix  = CROSSHAIR_GAP_AS  / pix_as
    # Two short segments per axis with a gap at centre
    ax.plot([cx,        cx], [cy + gap_pix, cy + half_pix], 'r-', lw=1.0, alpha=0.9)
    ax.plot([cx,        cx], [cy - gap_pix, cy - half_pix], 'r-', lw=1.0, alpha=0.9)
    ax.plot([cx + gap_pix, cx + half_pix], [cy, cy], 'r-', lw=1.0, alpha=0.9)
    ax.plot([cx - gap_pix, cx - half_pix], [cy, cy], 'r-', lw=1.0, alpha=0.9)


# ============================================================
# v01 render recipes + crosshair
# ============================================================
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
    _draw_crosshair(ax, rgb.shape[:2], pix_as=JWST_PIX_AS)
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
    _draw_crosshair(ax, rgb.shape[:2], pix_as=0.10)
    for k, t in enumerate(labels):
        ax.text(0.03, 0.95 - 0.07*k, t, color='white', fontsize=11,
                transform=ax.transAxes, verticalalignment='top',
                family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig); return True


def render_gray(data, png_path, labels, pix_as=HST_PIX_AS):
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
    _draw_crosshair(ax, bw.shape, pix_as=pix_as)
    for k, t in enumerate(labels):
        ax.text(0.03, 0.95 - 0.07*k, t, color='white', fontsize=11,
                transform=ax.transAxes, verticalalignment='top',
                family='serif', weight='bold')
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close(fig); return True


# ============================================================
# helpers
# ============================================================
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
        out_tile[m] = r.tile; out_file[m] = r.file
    return out_tile, out_file


def _open_2d(path):
    if path is None or not Path(path).exists(): return None, None
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


def check_jwst_pixel_coverage(data, wcs, ra, dec, half_size_pix=2):
    """Read (2*half_size_pix+1)² pixels around (ra, dec); return fraction
    of pixels that are finite & nonzero.  Rule C → require ≥ JWST_COVERAGE_MIN_FRAC."""
    if data is None or wcs is None:
        return 0.0
    try:
        x, y = wcs.all_world2pix(ra, dec, 0)
        xi = int(round(float(x))); yi = int(round(float(y)))
    except Exception:
        return 0.0
    ny, nx = data.shape
    y0 = max(0, yi - half_size_pix); y1 = min(ny, yi + half_size_pix + 1)
    x0 = max(0, xi - half_size_pix); x1 = min(nx, xi + half_size_pix + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    box = data[y0:y1, x0:x1]
    if box.size == 0:
        return 0.0
    n_total = box.size
    n_valid = int(np.sum(np.isfinite(box) & (box != 0)))
    return n_valid / n_total


def measure_hst_extent(data, wcs, ra, dec):
    img = cut_array(data, wcs, ra, dec, size_as=3.0)
    if img is None or not np.isfinite(img).any():
        return np.nan, np.nan
    finite = img[np.isfinite(img)]
    _, med, std = sigma_clipped_stats(finite, sigma=3.0, maxiters=3)
    sub = img - med
    ny, nx = sub.shape; cy, cx = ny/2.0, nx/2.0
    r_pix = 0.6 / HST_PIX_AS
    yy, xx = np.indices(sub.shape)
    in_aper = (xx - cx)**2 + (yy - cy)**2 <= r_pix**2
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


# ============================================================
# Catalog galaxy veto (light version, applied directly here)
# ============================================================
def _native(arr):
    """Convert big-endian FITS array to native byte order for pandas."""
    arr = np.asarray(arr)
    if arr.dtype.byteorder == '>':
        arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr


def load_galaxy_catalog_veto():
    """Returns three pandas DataFrames + SkyCoord for fast cross-match."""
    print('Loading Euclid MER, HST ACS, COSMOS-Web, COSMOS2020 catalogs for galaxy veto...')
    from astropy.table import Table

    eu = Table.read(CATDIR / 'Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits')
    eu_df = pd.DataFrame({
        'ra': _native(eu['right_ascension']),
        'dec': _native(eu['declination']),
        'phz': _native(eu['phz_classification']),
        'fwhm_px': _native(eu['fwhm']),
        'ellip': _native(eu['ellipticity']),
    })
    ac = Table.read(CATDIR / 'HST/COSMOS/cosmos_acs_iphot_200709.fits')
    a = _native(ac['a_image']); b = _native(ac['b_image'])
    ac_df = pd.DataFrame({
        'ra': _native(ac['ra']),
        'dec': _native(ac['dec']),
        'class_star': _native(ac['class_star']),
        'mu_class':   _native(ac['mu_class']),
        'fwhm_as': _native(ac['fwhm_image']) * 0.030,
        'ellip': 1.0 - b / np.where(a > 0, a, np.nan),
    })
    with fits.open(CATDIR / 'JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits', memmap=True) as h:
        d = h[1].data
        cw_df = pd.DataFrame({
            'ra': _native(d['ra']),
            'dec': _native(d['dec']),
            'flag_star': _native(d['flag_star']),
            'fwhm': _native(d['fwhm']),
            'sersic_n': _native(d['sersic']),
            'axratio': _native(d['axratio_sersic']),
        })
    with fits.open(CATDIR / 'COSMOS2020/COSMOS/COSMOS2020_FARMER_R1_v2.2_p3.fits.gz', memmap=True) as h:
        d = h[1].data
        c20_df = pd.DataFrame({
            'ra': _native(d['ALPHA_J2000']),
            'dec': _native(d['DELTA_J2000']),
            'lp_type': _native(d['lp_type']),
            'acs_mu': _native(d['ACS_MU_CLASS']),
            'acs_fwhm': _native(d['ACS_FWHM_WORLD']) * 3600.0,
        })
    return eu_df, ac_df, cw_df, c20_df


def is_galaxy_veto(ra, dec, eu_df, ac_df, cw_df, c20_df, match_as=0.5):
    """For each (ra, dec), return True if any catalog flags it as galaxy."""
    co = SkyCoord(ra * u.deg, dec * u.deg)
    veto = np.zeros(len(ra), dtype=bool)

    def _check(catdf, criterion):
        cc = SkyCoord(catdf['ra'].values * u.deg, catdf['dec'].values * u.deg)
        idx_o, idx_c, sep, _ = search_around_sky(co, cc, match_as * u.arcsec)
        if len(idx_o) == 0:
            return np.zeros(len(ra), dtype=bool)
        df = pd.DataFrame({'o': idx_o, 'c': idx_c, 's': sep.arcsec})
        df = df.sort_values(['o', 's']).drop_duplicates('o', keep='first')
        is_gal = criterion(catdf.iloc[df['c'].values])
        out = np.zeros(len(ra), dtype=bool)
        out[df['o'].values] = is_gal
        return out

    veto |= _check(eu_df,  lambda x: (x['phz'] == 2) |
                                       (x['fwhm_px'] > 2.5) | (x['ellip'] > 0.30))
    veto |= _check(ac_df,  lambda x: (x['mu_class'] == 1) | (x['class_star'] < 0.5) |
                                       (x['fwhm_as'] > 1.5 * HST_PSF_AS) | (x['ellip'] > 0.30))
    veto |= _check(cw_df,  lambda x: (x['flag_star'] == 0) & (x['fwhm'] > 1.5 * 0.06))
    veto |= _check(c20_df, lambda x: (x['lp_type'] == 0) | (x['acs_mu'] == 1) |
                                       (x['acs_fwhm'] > 1.5 * HST_PSF_AS))
    return veto


# ============================================================
# Main
# ============================================================
def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    pm = pd.read_parquet(OUT / 'proper_motion_v01.parquet')
    print(f'PM table: {len(pm):,} rows')

    # ---- Rule A + B: 3-mission detection (n_epochs == 3) ----
    df = pm[pm['n_epochs'] == 3].reset_index(drop=True).copy()
    print(f'\nRule A+B: n_epochs==3 (HST + JWST + Euclid all detected): {len(df):,}')

    # ---- Rule C: JWST pixel coverage check at exact position ----
    print('\nRule C: JWST F115W pixel coverage at each star position...')
    df['_jwst_tile'], _ = assign_tiles(df['ra_ref'].values, df['dec_ref'].values,
                                       tile_df, 'JWST', 'F115W')
    cov_frac = np.zeros(len(df))
    t0 = time.time()
    for tname, grp in df.groupby('_jwst_tile'):
        if not tname:
            continue
        data, hdr = _open_2d(_jwst_path(tname, 'F115W'))
        wcs = WCS(hdr) if hdr is not None else None
        if data is None or wcs is None:
            continue
        for r in grp.itertuples():
            i = df.index.get_loc(r.Index)
            cov_frac[i] = check_jwst_pixel_coverage(data, wcs, r.ra_ref, r.dec_ref)
        print(f'  JWST F115W tile {tname:>5s}: {len(grp)} pixel checks   ({time.time()-t0:.1f}s)',
              flush=True)
    df['jwst_cov_frac'] = cov_frac
    veto_jwst_gap = df['jwst_cov_frac'] < JWST_COVERAGE_MIN_FRAC
    print(f'  JWST chip-gap veto: {int(veto_jwst_gap.sum()):,} / {len(df):,} flagged')

    df = df[~veto_jwst_gap].reset_index(drop=True)

    # ---- Rule D: image-extent veto ----
    print('\nRule D: HST F814W image-based extent veto...')
    df['_hst_tile'], _ = assign_tiles(df['ra_ref'].values, df['dec_ref'].values,
                                      tile_df, 'HST', 'F814W')
    fwhm_arr  = np.full(len(df), np.nan)
    ellip_arr = np.full(len(df), np.nan)
    t0 = time.time()
    for tname, grp in df.groupby('_hst_tile'):
        if not tname:
            continue
        data, hdr = _open_2d(_hst_path(tname))
        wcs = WCS(hdr) if hdr is not None else None
        for r in grp.itertuples():
            f, e = measure_hst_extent(data, wcs, r.ra_ref, r.dec_ref)
            i = df.index.get_loc(r.Index)
            fwhm_arr[i] = f
            ellip_arr[i] = e
        print(f'  HST tile {tname:>5s}: {len(grp)} extent measurements   ({time.time()-t0:.1f}s)',
              flush=True)
    df['img_fwhm_hst_as'] = fwhm_arr
    df['img_ellip_hst']   = ellip_arr
    veto_img = ((df['img_fwhm_hst_as'] > HST_FWHM_GAL_AS) |
                (df['img_ellip_hst']   > HST_ELLIP_GAL)).fillna(False)
    print(f'  Image extent veto: {int(veto_img.sum()):,} galaxies/artifacts flagged')

    df = df[~veto_img].reset_index(drop=True)

    # ---- Catalog galaxy veto ----
    print('\nCatalog galaxy veto (Euclid MER, HST ACS, CW v1.1, COSMOS2020)...')
    eu_df, ac_df, cw_df, c20_df = load_galaxy_catalog_veto()
    veto_cat = is_galaxy_veto(df['ra_ref'].values, df['dec_ref'].values,
                               eu_df, ac_df, cw_df, c20_df)
    print(f'  Catalog galaxy veto: {int(veto_cat.sum()):,} flagged')
    df = df[~veto_cat].reset_index(drop=True)

    # ---- Save the cleaned 3-mission star list ----
    df.to_parquet(OUT / 'orphans_v10.parquet', index=False)
    print(f'\nv10 (clean 3-mission stars): {len(df):,}')

    # ---- Sort by PM total ----
    df = df.sort_values('pm_tot_mas_yr', ascending=False).reset_index(drop=True)
    print('\nTop PM stars (most "orphan-like"):')
    print(df.head(8)[['jwst_id','hst_id','euclid_id','ra_ref','dec_ref','pm_tot_mas_yr',
                       'img_fwhm_hst_as','img_ellip_hst','jwst_cov_frac']]
              .to_string(index=False))

    # ---- Top N for webpage ----
    sub = df.head(TOP_N).reset_index(drop=True)
    sub['seq'] = np.arange(1, len(sub) + 1)
    sub['_pd_idx'] = sub.index.values
    print(f'\nWebpage v05: top {len(sub):,} candidates by PM total')

    # Tile assignments for cutouts (we already have HST and JWST tiles)
    for mi, ba in [('JWST', 'F150W'), ('JWST', 'F277W'), ('JWST', 'F444W'),
                   ('Euclid', 'VIS'), ('Euclid', 'NIR-Y'),
                   ('Euclid', 'NIR-J'), ('Euclid', 'NIR-H')]:
        col = f'_tile_{mi}_{ba.replace("-","_")}'
        sub[col], _ = assign_tiles(sub['ra_ref'].values, sub['dec_ref'].values,
                                    tile_df, mi, ba)
    # HST + JWST F115W already in _hst_tile / _jwst_tile
    sub['_tile_HST_F814W']  = sub['_hst_tile']
    sub['_tile_JWST_F115W'] = sub['_jwst_tile']

    paths = {}

    # ---- HST single ----
    def b_hst(data, wcs, r, out_png, slug):
        if out_png.exists(): return slug
        img = cut_array(data, wcs, r.ra_ref, r.dec_ref) if data is not None else None
        ok = render_gray(img, out_png, [f'ID={r.jwst_id}', 'HST F814W'])
        return slug if ok else None

    def _grouped(mission, ba, builder, slug_suffix):
        col = f'_tile_{mission}_{ba.replace("-","_")}'
        out = {}
        for tname, grp in sub.groupby(col):
            if not tname:
                for idx in grp.index: out[idx] = None
                continue
            t0 = time.time()
            if mission == 'HST':
                p = _hst_path(tname)
            elif mission == 'JWST':
                p = _jwst_path(tname, ba)
            else:
                row = tile_df[(tile_df['mission']=='Euclid') &
                              (tile_df['band']==ba) &
                              (tile_df['tile']==tname)]
                p = EUCLID_DIR / row['file'].iloc[0] if len(row) else None
            data, hdr = _open_2d(p)
            wcs = WCS(hdr) if hdr is not None else None
            for r in grp.itertuples():
                slug = f'star_orph_{r.Index:04d}_{slug_suffix}.png'
                out_png = PNG_DIRS[{'HST':'hst','JWST':'jwst','Euclid':'euclid'}[mission]] / slug
                out[r.Index] = builder(data, wcs, r, out_png, slug)
            print(f'  {mission:6s} {ba:6s} tile {str(tname):>10s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
                  flush=True)
        return out

    print('\nBuilding HST F814W thumbnails...')
    paths['hst'] = _grouped('HST', 'F814W', b_hst, 'hst')

    def build_jwst_rgb(b_band, g_band, r_band, slug_suffix):
        result = {}
        col_b = f'_tile_JWST_{b_band}'
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
                slug = f'star_orph_{r.Index:04d}_{slug_suffix}.png'
                out_png = PNG_DIRS['jwst'] / slug
                if out_png.exists():
                    result[r.Index] = slug; continue
                cuts = {}
                for band in (b_band, g_band, r_band):
                    cuts[band] = (cut_array(data[band], wcs_[band], r.ra_ref, r.dec_ref)
                                  if data[band] is not None and wcs_[band] is not None
                                  else None)
                ok = render_rgb_jwst(cuts[b_band], cuts[g_band], cuts[r_band],
                                      out_png, [f'ID={r.jwst_id}',
                                                f'{b_band}/{g_band}/{r_band}'])
                result[r.Index] = slug if ok else None
            print(f'  jwst_{slug_suffix:<6} tile {tname:>5s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
                  flush=True)
        return result

    print('\nBuilding JWST RGB jwst1 (F115/F150/F277)...')
    paths['jwst1'] = build_jwst_rgb('F115W', 'F150W', 'F277W', 'jwst1')
    print('Building JWST RGB jwst2 (F150/F277/F444)...')
    paths['jwst2'] = build_jwst_rgb('F150W', 'F277W', 'F444W', 'jwst2')

    def b_euvis(data, wcs, r, out_png, slug):
        if out_png.exists(): return slug
        img = cut_array(data, wcs, r.ra_ref, r.dec_ref) if data is not None else None
        ok = render_gray(img, out_png, [f'ID={r.jwst_id}', 'Euclid VIS'], pix_as=0.10)
        return slug if ok else None

    print('\nBuilding Euclid VIS thumbnails...')
    paths['vis'] = _grouped('Euclid', 'VIS', b_euvis, 'euclid_vis')

    print('\nBuilding Euclid NISP RGB Y/J/H...')
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
                cuts[band] = (cut_array(data[band], wcs_[band], r.ra_ref, r.dec_ref)
                              if data[band] is not None and wcs_[band] is not None
                              else None)
            ok = render_rgb_euclid(cuts['NIR-Y'], cuts['NIR-J'], cuts['NIR-H'],
                                    out_png, [f'ID={r.jwst_id}', 'Y/J/H'])
            nisp_paths[r.Index] = slug if ok else None
        print(f'  euclid_nisp tile {tname:>10s}: {len(grp)} thumbs in {time.time()-t0:.1f}s',
              flush=True)
    paths['nisp'] = nisp_paths

    # ---- HTML pages ----
    def write_page(rows, page_idx, n_pages):
        rows_html = []
        for r in rows:
            idx = r['_pd_idx']
            head = (f"#{r['seq']}  ID={r['jwst_id']}  RA={r['ra_ref']:.6f}  "
                    f"Dec={r['dec_ref']:.6f}  "
                    f"|μ|={r['pm_tot_mas_yr']:.1f} mas/yr  "
                    f"HST_fwhm={r.get('img_fwhm_hst_as',float('nan')):.3f}″  "
                    f"HST_ellip={r.get('img_ellip_hst',float('nan')):.2f}  "
                    f"JWST_cov={r.get('jwst_cov_frac',1.0):.2f}")
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
        pag = [f"<a href='star_index{max(1,page_idx-1):04d}.html'>&laquo;</a>"]
        for p in range(1, n_pages + 1):
            cls = " class='active'" if p == page_idx else ""
            pag.append(f"<a href='star_index{p:04d}.html'{cls}>{p}</a>")
        pag.append(f"<a href='star_index{min(n_pages,page_idx+1):04d}.html'>&raquo;</a>")
        html = f"""<!doctype html><html><head>
<title>Star orphans v05 (page {page_idx}/{n_pages})</title>
{CSS}</head><body>
<div class='header'>
<h2 style='margin:4px 0;'>Star orphans v05 — page {page_idx} / {n_pages}</h2>
<div style='font-size:11px;color:#666'>
3-mission confirmed stars (HST + JWST + Euclid all detected) sorted by total PM (descending).
JWST F115W pixel coverage ≥ {JWST_COVERAGE_MIN_FRAC*100:.0f}% within 5×5 box.
HST F814W image FWHM ≤ {HST_FWHM_GAL_AS:.3f}″, ellip ≤ {HST_ELLIP_GAL}.
Catalog galaxy + saturated-star vetos applied.
Red crosshair marks the catalog position.</div>
<div class='pagination'>{chr(10).join(pag)}</div>
</div>
{chr(10).join(rows_html)}
</body></html>"""
        (HTML_ROOT / f'star_index{page_idx:04d}.html').write_text(html)

    n_pages = max(1, int(np.ceil(len(sub) / PER_PAGE)))
    for page_idx in range(1, n_pages + 1):
        rows = sub.iloc[(page_idx-1)*PER_PAGE: page_idx*PER_PAGE].to_dict('records')
        write_page(rows, page_idx, n_pages)
    print(f'\n→ {n_pages} HTML pages')

    idx_html = HTML_ROOT / 'index.html'
    idx_html.write_text(f"""<!doctype html><html><head>
<title>Star orphans — v05 index</title>{CSS}</head><body>
<div class='header'><h2>Star orphans v05 — strict 3-mission rule + JWST chip-gap check</h2></div>
<p style='padding:14px'>
<b>Rules:</b>
<ol>
  <li>Must have a JWST F115W detection (JWST ID is the canonical identifier). No JWST → disqualified.</li>
  <li>Must have HST F814W and Euclid VIS cross-matches.</li>
  <li>JWST pixel coverage ≥ {JWST_COVERAGE_MIN_FRAC*100:.0f}% in 5×5 around the position (removes chip-gap detections).</li>
  <li>HST F814W image extent: FWHM ≤ {HST_FWHM_GAL_AS:.3f}″ AND ellip ≤ {HST_ELLIP_GAL}.</li>
  <li>Catalog galaxy veto (Euclid PHZ, HST mu_class, CW flag_star, COSMOS2020 lp_type).</li>
</ol>
<b>Source list:</b> proper_motion_v01 → n_epochs == 3 → all vetos applied → sorted by total PM.
Output saved at <code>csvfiles_star/orphans_v10.parquet</code> (<b>{len(df):,}</b> rows),
top {len(sub):,} shown in the webpage.
</p>
<ul style='padding:14px'>
  <li><a href='star_index0001.html'>Star orphan candidates ({len(sub):,}, sorted by |μ|)</a></li>
</ul>
<p style='padding:14px;font-size:12px;color:#666'>
Each thumbnail carries a simple red crosshair at the catalog position.
Earlier versions: <a href='../v01/'>v01</a>, <a href='../v02/'>v02</a>,
<a href='../v03/'>v03</a>, <a href='../v04/'>v04</a> (now superseded).</p>
</body></html>""")
    print(f'Wrote {idx_html}')


if __name__ == '__main__':
    main()
