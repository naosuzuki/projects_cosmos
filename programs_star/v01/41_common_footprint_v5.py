#!/usr/bin/env python
"""
41_common_footprint_v5.py — orphans restricted to the HST ∩ JWST ∩ Euclid
                            common pixel-coverage region.

The bbox-only footprints overstated coverage; the user's example orphans
at (150.33, 2.20) and similar were in HST/Euclid bbox AND in our JWST
bbox but in a JWST chip gap, so they should NOT have been "orphans".

Build a unified 3″/pixel coverage mask covering the full COSMOS area and
write orphans_v5.parquet = orphans_v4 ∩ common_region.

Output
------
  csvfiles_star/footprints/common_footprint_v5.fits        3″/pix mask
  csvfiles_star/orphans_v5.parquet                          common-region filter
  csvfiles_star/orphans_v5_summary.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.io.fits import Header

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
MASKS_DIR = OUT / 'footprints' / 'masks'

RA_MIN, RA_MAX   = 149.30, 151.20
DEC_MIN, DEC_MAX =   1.40,   3.30
OUT_PIX_AS = 3.0   # arcsec per pixel of unified mask


def build_output_wcs(out_pix_arcsec):
    pix_deg = out_pix_arcsec / 3600.0
    nx = int(np.ceil((RA_MAX  - RA_MIN ) / pix_deg))
    ny = int(np.ceil((DEC_MAX - DEC_MIN) / pix_deg))
    w = WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = [-pix_deg, pix_deg]
    w.wcs.crval = [RA_MAX, DEC_MIN]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w, ny, nx


def project_band_mask(band_prefix: str, out_wcs, ny, nx) -> np.ndarray:
    """OR per-tile downsampled masks onto the common sky grid."""
    mask = np.zeros((ny, nx), dtype=bool)
    paths = sorted(MASKS_DIR.glob(f'mask_*_{band_prefix}_*.npz'))
    if not paths:
        return mask
    for p in paths:
        arc = np.load(p, allow_pickle=False)
        m = arc['mask'].astype(bool)
        ds = int(arc['downsample'])
        hdr_str = arc['header'].item() if arc['header'].ndim == 0 else arc['header'][0]
        hdr = Header.fromstring(hdr_str.decode('ascii') if isinstance(hdr_str, bytes) else hdr_str)
        tile_wcs = WCS(hdr)
        iy, ix = np.where(m)
        if iy.size == 0:
            continue
        px = ix * ds + ds / 2.0
        py = iy * ds + ds / 2.0
        ra, dec = tile_wcs.all_pix2world(px, py, 0)
        x, y = out_wcs.all_world2pix(ra, dec, 0)
        xi = np.round(x).astype(int)
        yi = np.round(y).astype(int)
        ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        mask[yi[ok], xi[ok]] = True
    return mask


def in_mask(ra, dec, mask, out_wcs):
    x, y = out_wcs.all_world2pix(ra, dec, 0)
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    ny, nx = mask.shape
    ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    out = np.zeros(len(ra), dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def main():
    out_wcs, ny, nx = build_output_wcs(OUT_PIX_AS)
    print(f'Common grid: {ny} × {nx} at {OUT_PIX_AS}″/px')

    print('\nProjecting per-tile masks → common grid:')
    hst_mask  = project_band_mask('F814W', out_wcs, ny, nx)
    jwst_mask = project_band_mask('F115W', out_wcs, ny, nx)
    print(f'  HST F814W : {int(hst_mask.sum()):>6,d} px = {hst_mask.sum() * (OUT_PIX_AS/60)**2:.1f} arcmin²')
    print(f'  JWST F115W: {int(jwst_mask.sum()):>6,d} px = {jwst_mask.sum() * (OUT_PIX_AS/60)**2:.1f} arcmin²')

    # Dilate masks by 1 pixel (3″) to be slightly forgiving on edge
    try:
        from scipy.ndimage import binary_dilation
        hst_mask  = binary_dilation(hst_mask, iterations=1)
        jwst_mask = binary_dilation(jwst_mask, iterations=1)
    except ImportError:
        pass

    # Euclid VIS bbox — use existing band footprint WKT (covers full COSMOS)
    # For "common", HST ∩ JWST is the binding constraint; Euclid covers more.
    common = hst_mask & jwst_mask
    cov_px = int(common.sum())
    cov_arcmin2 = cov_px * (OUT_PIX_AS / 60.0) ** 2
    print(f'\n  HST ∩ JWST common: {cov_px:,} px = {cov_arcmin2:.1f} arcmin² = '
          f'{cov_arcmin2/3600.0:.3f} deg²')

    # Save common mask
    out_path = OUT / 'footprints' / 'common_footprint_v5.fits'
    h = out_wcs.to_header()
    h['COMMENT'] = 'HST F814W AND JWST F115W actual pixel coverage'
    h['COVPX']  = cov_px
    h['COVAM2'] = cov_arcmin2
    fits.writeto(out_path, common.astype(np.uint8), h, overwrite=True)
    print(f'Wrote {out_path}')

    # Apply to orphan list (use v4 which has Gaia/peak/local vetoes already
    # applied — we further restrict to common region)
    print('\nApplying common-region filter to orphans_v4:')
    orph = pd.read_parquet(OUT / 'orphans_v4.parquet')
    print(f'  orphans_v4: {len(orph):,}')

    in_common = in_mask(orph['ra'].values, orph['dec'].values, common, out_wcs)
    orph['in_common_v5'] = in_common
    n_in = int(in_common.sum())
    print(f'  inside HST∩JWST common: {n_in:,}  ({100 * in_common.mean():.1f} %)')

    orph_v5 = orph[in_common].copy()
    orph_v5.to_parquet(OUT / 'orphans_v5.parquet', index=False)
    print(f'  → orphans_v5.parquet  ({len(orph_v5):,} rows)')

    # High-confidence v5
    sharp_map = {}
    for band in ('F814W', 'F115W', 'VIS'):
        d = pd.read_parquet(OUT / f'good_stars_{band}.parquet')[['star_id', 'sharpness']]
        sharp_map[band] = d.set_index('star_id')['sharpness'].to_dict()
    found_to_band = {'HST F814W': 'F814W', 'JWST F115W': 'F115W', 'Euclid VIS': 'VIS'}
    def lookup_sharp(row):
        band = found_to_band.get(row.found_in, '')
        return sharp_map.get(band, {}).get(row.star_id, np.nan)
    orph_v5['sharpness'] = orph_v5.apply(lookup_sharp, axis=1)
    hi = orph_v5[(orph_v5['snr'] > 20) & (orph_v5['sharpness'].between(0.5, 0.75))].copy()
    hi.to_parquet(OUT / 'orphans_highconf_v5.parquet', index=False)
    print(f'  orphans_highconf_v5: {len(hi):,}')

    # Breakdown by found_in + summary
    print('\norphans_v5 breakdown:')
    print(orph_v5.groupby('found_in').size().to_string())
    print('\norphans_highconf_v5 breakdown:')
    if len(hi):
        print(hi.groupby('found_in').size().to_string())

    # Persist the per-found_in totals
    sm = orph_v5.groupby('found_in').size().rename('n_orph_v5').to_frame()
    sm['n_orph_highconf_v5'] = hi.groupby('found_in').size()
    sm.to_csv(OUT / 'orphans_v5_summary.csv')

    # Refresh asteroid + transient candidates restricted to common region
    print('\nRefreshing candidate lists v5:')
    ast = orph_v5[
        (orph_v5['found_in'] == 'HST F814W') &
        orph_v5['missing_jwst'] & orph_v5['missing_euclid'] &
        (orph_v5['snr'] > 30) &
        (orph_v5['sharpness'].between(0.55, 0.70))
    ].copy()
    ast['ra_r']  = (ast['ra']  * 36000).round() / 36000
    ast['dec_r'] = (ast['dec'] * 36000).round() / 36000
    ast = ast.sort_values('snr', ascending=False).drop_duplicates(['ra_r','dec_r']).drop(columns=['ra_r','dec_r'])
    ast.to_csv(OUT / 'asteroid_candidates_v5.csv', index=False)
    print(f'  asteroid candidates v5: {len(ast):,}')

    sn = orph_v5[
        orph_v5['found_in'].isin(['JWST F115W', 'Euclid VIS']) &
        orph_v5['missing_hst'] &
        (orph_v5['snr'] > 30) &
        (orph_v5['sharpness'].between(0.55, 0.70))
    ].copy()
    sn['ra_r']  = (sn['ra']  * 36000).round() / 36000
    sn['dec_r'] = (sn['dec'] * 36000).round() / 36000
    sn = sn.sort_values('snr', ascending=False).drop_duplicates(['ra_r','dec_r']).drop(columns=['ra_r','dec_r'])
    sn.to_csv(OUT / 'sn_candidates_v5.csv', index=False)
    print(f'  modern SN/transient candidates v5: {len(sn):,}')


if __name__ == '__main__':
    main()
