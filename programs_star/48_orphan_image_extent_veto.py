#!/usr/bin/env python
"""
48_orphan_image_extent_veto.py — measure source extent directly from
                                  HST F814W (sharpest PSF) image cutouts
                                  and veto galaxies whose fit FWHM exceeds
                                  ~1.8 × PSF.

Method:
  For each orphan:
   1. Make a 3″ HST F814W cutout centred on the orphan position.
   2. Background-subtract using sigma-clipped median.
   3. Compute photutils data_properties on a centred sub-region
      (5×PSF half-width) → semimajor/semiminor/ellipticity, FWHM via
      σ × √(8 ln 2).
   4. Veto rules:
        fwhm_HST  > 1.6 × 0.134″          → galaxy
        ellip_HST > 0.30                   → galaxy
   5. Also same check on JWST F115W (sharpest IR PSF) if available.

This catches galaxies that the catalog-based veto missed because they
had no catalog source within 0.5″.

Input  : orphans_v6_with_flags.parquet   (catalog flags already attached)
Output : orphans_v7.parquet              (additional image-extent veto)
         orphans_v7_with_flags.parquet   (all v6 + image-extent measurements)
"""
from __future__ import annotations
import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy import units as u
from photutils.morphology import data_properties

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

HST_DIR  = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')

HST_PSF_AS  = 0.134
JWST_PSF_AS = 0.057
HST_PIX_AS  = 0.030
JWST_PIX_AS = 0.030

CUTOUT_AS = 3.0          # 100 px in HST/JWST
APER_RADIUS_AS = 0.6     # 5 × HST PSF half-width

# Veto rules
HST_FWHM_GAL_AS  = 1.6 * HST_PSF_AS    # ~0.22″
HST_ELLIP_GAL    = 0.30
JWST_FWHM_GAL_AS = 1.6 * JWST_PSF_AS   # ~0.09″
JWST_ELLIP_GAL   = 0.30


def _open_2d(path):
    if not Path(path).exists(): return None, None
    with fits.open(path, memmap=True) as h:
        for hdu in h:
            if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                return np.asarray(hdu.data), hdu.header.copy()
    return None, None


def _hst_path(tile_int): return HST_DIR / f'acs_I_030mas_{int(tile_int):03d}_sci.fits'
def _jwst_path(tile):
    if tile == 'A10':
        return JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    return JWST_DIR / f'mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'


def assign_tiles(ra, dec, tile_df, mission, band):
    sel = tile_df[(tile_df['mission'] == mission) & (tile_df['band'] == band)]
    out_tile = np.full(len(ra), '', dtype=object)
    for r in sel.itertuples():
        ra_min  = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        ra_max  = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        m = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max) & (out_tile == '')
        out_tile[m] = r.tile
    return out_tile


def measure_extent(data, wcs, ra, dec, pix_as=HST_PIX_AS, size_as=CUTOUT_AS,
                    aper_as=APER_RADIUS_AS):
    """Return (fwhm_as, ellip, peak/bg_std) from a small cutout."""
    if data is None or wcs is None:
        return np.nan, np.nan, np.nan
    try:
        c = SkyCoord(ra * u.deg, dec * u.deg)
        cu = Cutout2D(data, c, size=size_as * u.arcsec, wcs=wcs,
                      mode='partial', fill_value=np.nan)
    except Exception:
        return np.nan, np.nan, np.nan
    img = cu.data
    if not np.isfinite(img).any():
        return np.nan, np.nan, np.nan
    # Background
    finite = img[np.isfinite(img)]
    _, med, std = sigma_clipped_stats(finite, sigma=3.0, maxiters=3)
    sub = img - med
    # Aperture mask
    ny, nx = sub.shape
    cy, cx = ny / 2.0, nx / 2.0
    r_pix = aper_as / pix_as
    y, x = np.indices(sub.shape)
    in_aper = (x - cx)**2 + (y - cy)**2 <= r_pix**2
    if not in_aper.any():
        return np.nan, np.nan, np.nan
    pos = sub.copy()
    pos[~np.isfinite(pos)] = 0
    # Threshold for moments: only pixels above 2σ inside the aperture
    thr = 2.0 * std
    mask = in_aper & np.isfinite(sub) & (sub > thr)
    if mask.sum() < 5:
        return np.nan, np.nan, std
    try:
        cat = data_properties(pos, mask=mask)
        a = float(cat.semimajor_sigma.value) * pix_as       # arcsec
        b = float(cat.semiminor_sigma.value) * pix_as
        ellip = 1.0 - b / max(a, 1e-9)
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(max(a*b, 1e-12))
        peak_snr = float(np.nanmax(sub[in_aper])) / max(std, 1e-9)
        return fwhm, ellip, peak_snr
    except Exception:
        return np.nan, np.nan, std


def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    df = pd.read_parquet(OUT / 'orphans_v6_with_flags.parquet')
    print(f'orphans_v6_with_flags: {len(df):,}')

    # We need image-based extent only for the v6-pass-through (galaxy_veto==False)
    # AND only the top ~600 per category that the webpage shows.
    # But for completeness, compute for all v5 orphans (after sat_v3 + common region).
    # That's 336k rows × ~150 ms per cutout = ~14 hours.  Cap to top 5000 per
    # category by SNR.
    pieces = []
    for cat in ('HST F814W', 'JWST F115W', 'Euclid VIS'):
        sub = df[df['found_in'] == cat].sort_values('snr', ascending=False).head(5000)
        pieces.append(sub)
    work = pd.concat(pieces, ignore_index=False).copy()
    print(f'measuring image extent for top-5000-per-category: {len(work):,}')

    # Assign HST tile
    ra  = work['ra'].values
    dec = work['dec'].values
    work['_hst_tile']  = assign_tiles(ra, dec, tile_df, 'HST',  'F814W')
    work['_jwst_tile'] = assign_tiles(ra, dec, tile_df, 'JWST', 'F115W')

    # HST extent
    fwhm_hst  = np.full(len(work), np.nan)
    ellip_hst = np.full(len(work), np.nan)
    snr_hst   = np.full(len(work), np.nan)

    t0 = time.time()
    for tname, grp in work.groupby('_hst_tile'):
        if not tname:
            continue
        d, h = _open_2d(_hst_path(tname))
        wcs = WCS(h) if h is not None else None
        for r in grp.itertuples():
            f, e, s = measure_extent(d, wcs, r.ra, r.dec, pix_as=HST_PIX_AS)
            i = work.index.get_loc(r.Index)
            fwhm_hst[i]  = f
            ellip_hst[i] = e
            snr_hst[i]   = s
        print(f'  HST tile {tname:>5s}: {len(grp)} extent measurements   t={time.time()-t0:.1f}s',
              flush=True)

    # JWST extent
    fwhm_jw  = np.full(len(work), np.nan)
    ellip_jw = np.full(len(work), np.nan)
    snr_jw   = np.full(len(work), np.nan)

    for tname, grp in work.groupby('_jwst_tile'):
        if not tname:
            continue
        d, h = _open_2d(_jwst_path(tname))
        wcs = WCS(h) if h is not None else None
        for r in grp.itertuples():
            f, e, s = measure_extent(d, wcs, r.ra, r.dec, pix_as=JWST_PIX_AS,
                                      aper_as=APER_RADIUS_AS * 0.5)  # tighter for sharp JWST
            i = work.index.get_loc(r.Index)
            fwhm_jw[i]  = f
            ellip_jw[i] = e
            snr_jw[i]   = s
        print(f'  JWST tile {tname:>5s}: {len(grp)} extent measurements   t={time.time()-t0:.1f}s',
              flush=True)

    work['img_fwhm_hst_as']  = fwhm_hst
    work['img_ellip_hst']    = ellip_hst
    work['img_peak_snr_hst'] = snr_hst
    work['img_fwhm_jw_as']   = fwhm_jw
    work['img_ellip_jw']     = ellip_jw
    work['img_peak_snr_jw']  = snr_jw

    # Veto rules (any one fires → galaxy)
    veto_hst = ((work['img_fwhm_hst_as']  > HST_FWHM_GAL_AS)  |
                (work['img_ellip_hst']    > HST_ELLIP_GAL))
    veto_jw  = ((work['img_fwhm_jw_as']   > JWST_FWHM_GAL_AS) |
                (work['img_ellip_jw']     > JWST_ELLIP_GAL))
    work['veto_img_gal'] = veto_hst.fillna(False) | veto_jw.fillna(False)
    print('\nImage extent veto breakdown:')
    print(f'  HST FWHM/ellip galaxy : {int(veto_hst.sum()):,}')
    print(f'  JWST FWHM/ellip galaxy: {int(veto_jw.sum()):,}')
    print(f'  Any extent veto       : {int(work["veto_img_gal"].sum()):,}')

    # Build orphans_v7 = v6 (already had galaxy_veto applied) ∪ work with veto_img_gal=False
    v6 = df.copy()
    # Merge the image-extent flags onto v6 by index
    v6.loc[work.index, 'img_fwhm_hst_as']  = work['img_fwhm_hst_as'].values
    v6.loc[work.index, 'img_ellip_hst']    = work['img_ellip_hst'].values
    v6.loc[work.index, 'img_peak_snr_hst'] = work['img_peak_snr_hst'].values
    v6.loc[work.index, 'img_fwhm_jw_as']   = work['img_fwhm_jw_as'].values
    v6.loc[work.index, 'img_ellip_jw']     = work['img_ellip_jw'].values
    v6.loc[work.index, 'img_peak_snr_jw']  = work['img_peak_snr_jw'].values
    v6.loc[work.index, 'veto_img_gal']     = work['veto_img_gal'].values

    v6['veto_img_gal']    = v6.get('veto_img_gal', False).fillna(False)
    v6['veto_total']      = v6['veto_galaxy'] | v6['veto_img_gal']

    v7 = v6[~v6['veto_total']].copy()
    v7.to_parquet(OUT / 'orphans_v7.parquet', index=False)
    v6.to_parquet(OUT / 'orphans_v7_with_flags.parquet', index=False)
    print(f'\norphans_v7 (image-extent + catalog galaxy veto): {len(v7):,}')

    # By category
    print('\nv7 breakdown by found_in:')
    print(v7.groupby('found_in').size().to_string())


if __name__ == '__main__':
    main()
