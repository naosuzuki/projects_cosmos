#!/usr/bin/env python
"""
52_step3_catalog_match.py — NEW STEP 3.

Match every DAO detection to its mission's published catalog within
ONE MOSAIC PIXEL.  Any DAO detection without a catalog match is dropped
(not a real object).  The catalog ID becomes the primary key.

  HST F814W  → HST/COSMOS/cosmos_acs_iphot_200709.fits      (1 ACS pixel  = 0.030″)
  JWST       → JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits (1 NIRCam pix = 0.030″)
               (one row per CW id, with up to 4 band DAO sets per row)
  Euclid VIS → Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits     (1 VIS pixel  = 0.100″)
  Euclid NISP→ Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits     (1 NISP pix   = 0.100″)
               (one row per MER id with up to 3 NIR-Y/J/H DAO sets)

Outputs:
  csvfiles_star/cat_matched_HST.parquet         + .csv
  csvfiles_star/cat_matched_JWST.parquet        + .csv
  csvfiles_star/cat_matched_Euclid_VIS.parquet  + .csv
  csvfiles_star/cat_matched_Euclid_NISP.parquet + .csv

All DAO parameters per band are preserved (x, y, ra, dec, sharpness,
roundness1, roundness2, npix, peak, flux, mag, sky_med, sky_std,
fwhm_pix, tile, snr) plus the per-band separation to the catalog source.
Catalog magnitudes & errors per band are also included.
"""
from __future__ import annotations
import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.wcs import FITSFixedWarning
from astropy import units as u

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
DAO_DIR = OUT / 'dao'
CAT_DIR = Path('/Volumes/exdisk1/data/catalog')

HST_PIX_AS    = 0.030
JWST_PIX_AS   = 0.030
EU_VIS_PIX_AS = 0.100
EU_NISP_PIX_AS = 0.100

DAO_COLS_KEEP = ['x', 'y', 'ra', 'dec', 'sharpness', 'roundness1', 'roundness2',
                 'npix', 'peak', 'flux', 'mag', 'sky_med', 'sky_std',
                 'fwhm_pix', 'tile']


def _native(arr):
    arr = np.asarray(arr)
    if arr.dtype.byteorder == '>':
        arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr


def best_dao_per_cat(dao_df: pd.DataFrame, cat_ra: np.ndarray, cat_dec: np.ndarray,
                     radius_as: float):
    """Search-around DAO ↔ catalog.  For each catalog source keep the
    closest DAO row.  Returns (cat_idx_array, dao_idx_array, sep_arcsec)."""
    if len(dao_df) == 0:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    dao_co = SkyCoord(dao_df['ra'].values * u.deg, dao_df['dec'].values * u.deg)
    cat_co = SkyCoord(cat_ra * u.deg, cat_dec * u.deg)
    idx_d, idx_c, sep, _ = search_around_sky(dao_co, cat_co, radius_as * u.arcsec)
    if len(idx_d) == 0:
        return (np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    df = pd.DataFrame({'d': idx_d, 'c': idx_c, 's': sep.arcsec})
    df = df.sort_values(['c', 's']).drop_duplicates('c', keep='first')
    return df['c'].values, df['d'].values, df['s'].values


# ============================================================
# HST
# ============================================================
def step3_hst():
    print('\n========== HST F814W ==========')
    dao = pd.read_parquet(DAO_DIR / 'dao_F814W.parquet')
    dao['snr'] = dao['peak'] / dao['sky_std']
    print(f'  DAO F814W detections: {len(dao):,}')

    cat = Table.read(CAT_DIR / 'HST/COSMOS/cosmos_acs_iphot_200709.fits')
    print(f'  ACS i-phot catalog : {len(cat):,}')

    cat_ra  = _native(cat['ra'])
    cat_dec = _native(cat['dec'])
    cat_num = _native(cat['number'])
    cat_mag = _native(cat['mag_auto'])
    cat_err = _native(cat['magerr_auto'])
    cat_cs  = _native(cat['class_star'])
    cat_mu  = _native(cat['mu_class'])
    cat_fl  = _native(cat['flags'])
    cat_fwhm = _native(cat['fwhm_image']) * 0.030    # → arcsec
    cat_a    = _native(cat['a_image']) * 0.030
    cat_b    = _native(cat['b_image']) * 0.030

    t0 = time.time()
    idx_c, idx_d, sep_as = best_dao_per_cat(dao, cat_ra, cat_dec, HST_PIX_AS)
    print(f'  Best DAO per catalog id within {HST_PIX_AS*1000:.0f} mas: '
          f'{len(idx_c):,} ({time.time()-t0:.1f}s)')

    if len(idx_c) == 0:
        print('  no matches'); return

    d = dao.iloc[idx_d].reset_index(drop=True)[DAO_COLS_KEEP + ['snr']]
    d = d.add_prefix('dao_F814W_')
    out = pd.DataFrame({
        'hst_id'              : cat_num[idx_c],
        'cat_ra'              : cat_ra[idx_c],
        'cat_dec'             : cat_dec[idx_c],
        'cat_mag_F814W'       : cat_mag[idx_c],
        'cat_magerr_F814W'    : cat_err[idx_c],
        'cat_class_star'      : cat_cs[idx_c],
        'cat_mu_class'        : cat_mu[idx_c],
        'cat_flags'           : cat_fl[idx_c],
        'cat_fwhm_image_as'   : cat_fwhm[idx_c],
        'cat_a_image_as'      : cat_a[idx_c],
        'cat_b_image_as'      : cat_b[idx_c],
        'cat_sep_arcsec_F814W': sep_as,
    })
    out = pd.concat([out, d], axis=1)

    p = OUT / 'cat_matched_HST.parquet'
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix('.csv'), index=False)
    print(f'  → {p}  ({len(out):,} rows, {len(out.columns)} cols)')


# ============================================================
# JWST  (4 bands → one row per CW v1.1 id)
# ============================================================
def step3_jwst():
    print('\n========== JWST (F115/F150/F277/F444 → CW v1.1) ==========')

    with fits.open(CAT_DIR / 'JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits', memmap=True) as h:
        c = h[1].data
        # Extract just the columns we need (avoid copying the whole 287-col table)
        cw_id   = _native(c['id'])
        cw_ra   = _native(c['ra'])
        cw_dec  = _native(c['dec'])
        cw_flag_star = _native(c['flag_star'])
        cw_flag_blend = _native(c['flag_blend'])
        cw_fwhm = _native(c['fwhm'])
        cw_sersic = _native(c['sersic'])
        cw_axratio = _native(c['axratio_sersic'])
        cw_radius = _native(c['radius_sersic'])
        # Per-band catalog mags & SNR
        cw_per_band = {}
        for band in ('f115w', 'f150w', 'f277w', 'f444w'):
            cw_per_band[band] = {
                'mag_auto': _native(c[f'mag_auto_{band}']),
                'snr':       _native(c[f'snr_{band}']),
            }
    print(f'  CW v1.1 catalog : {len(cw_id):,}')

    # Per band match
    band_match = {}   # band → (idx_cat, idx_dao, sep_as, dao_df)
    for band in ('F115W', 'F150W', 'F277W', 'F444W'):
        dao = pd.read_parquet(DAO_DIR / f'dao_{band}.parquet')
        dao['snr'] = dao['peak'] / dao['sky_std']
        print(f'  DAO {band} detections: {len(dao):,}')
        t0 = time.time()
        idx_c, idx_d, sep = best_dao_per_cat(dao, cw_ra, cw_dec, JWST_PIX_AS)
        print(f'    best DAO per CW id ≤ {JWST_PIX_AS*1000:.0f} mas: '
              f'{len(idx_c):,} ({time.time()-t0:.1f}s)')
        band_match[band] = (idx_c, idx_d, sep, dao)

    # Union of catalog ids that have AT LEAST one band DAO match
    matched_cat_ids = set()
    for band in band_match:
        idx_c, _, _, _ = band_match[band]
        matched_cat_ids.update(idx_c.tolist())
    matched_cat_ids = np.array(sorted(matched_cat_ids), dtype=int)
    print(f'\n  Unique CW ids with ≥1 band DAO match: {len(matched_cat_ids):,}')

    # Build the per-CW-id table
    out = pd.DataFrame({
        'jwst_id'      : cw_id[matched_cat_ids],
        'cat_ra'       : cw_ra[matched_cat_ids],
        'cat_dec'      : cw_dec[matched_cat_ids],
        'cat_flag_star': cw_flag_star[matched_cat_ids],
        'cat_flag_blend': cw_flag_blend[matched_cat_ids],
        'cat_fwhm_as'  : cw_fwhm[matched_cat_ids],
        'cat_sersic_n' : cw_sersic[matched_cat_ids],
        'cat_axratio'  : cw_axratio[matched_cat_ids],
        'cat_radius_sersic' : cw_radius[matched_cat_ids],
    })
    for band in ('F115W', 'F150W', 'F277W', 'F444W'):
        b = band.lower()
        out[f'cat_mag_{band}']     = cw_per_band[b]['mag_auto'][matched_cat_ids]
        out[f'cat_snr_{band}']     = cw_per_band[b]['snr'][matched_cat_ids]

    # Attach per-band DAO photometry (NaN for non-matched bands)
    cat_id_to_pos = {int(cid): i for i, cid in enumerate(matched_cat_ids)}
    for band in ('F115W', 'F150W', 'F277W', 'F444W'):
        idx_c, idx_d, sep, dao = band_match[band]
        # row positions in `out` for the matched cat indices
        pos = np.array([cat_id_to_pos[int(ci)] for ci in idx_c])
        # init all band cols as NaN
        for col in DAO_COLS_KEEP + ['snr']:
            out_col = f'dao_{band}_{col}'
            if col == 'tile':
                out[out_col] = pd.Series(['']*len(out), dtype=object)
            else:
                out[out_col] = np.nan
        out[f'dao_{band}_sep_arcsec'] = np.nan
        # fill in the matched rows
        dao_rows = dao.iloc[idx_d].reset_index(drop=True)
        for col in DAO_COLS_KEEP + ['snr']:
            out_col = f'dao_{band}_{col}'
            out.loc[out.index[pos], out_col] = dao_rows[col].values
        out.loc[out.index[pos], f'dao_{band}_sep_arcsec'] = sep

    # Counts
    print('\n  Per-band DAO match presence (non-null dao_<band>_ra):')
    for band in ('F115W', 'F150W', 'F277W', 'F444W'):
        n = int(out[f'dao_{band}_ra'].notna().sum())
        print(f'    {band}: {n:,}')
    n_all4 = int(((out['dao_F115W_ra'].notna()) & (out['dao_F150W_ra'].notna()) &
                  (out['dao_F277W_ra'].notna()) & (out['dao_F444W_ra'].notna())).sum())
    print(f'    all 4 bands: {n_all4:,}')

    p = OUT / 'cat_matched_JWST.parquet'
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix('.csv'), index=False)
    print(f'  → {p}  ({len(out):,} rows, {len(out.columns)} cols)')


# ============================================================
# Euclid VIS  (1 band → 1 row per MER object_id)
# ============================================================
def step3_euclid_vis():
    print('\n========== Euclid VIS ==========')
    dao = pd.read_parquet(DAO_DIR / 'dao_VIS.parquet')
    dao['snr'] = dao['peak'] / dao['sky_std']
    print(f'  DAO VIS detections: {len(dao):,}')

    cat = Table.read(CAT_DIR / 'Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits')
    print(f'  MER catalog       : {len(cat):,}')

    cat_id  = _native(cat['object_id'])
    cat_ra  = _native(cat['right_ascension'])
    cat_dec = _native(cat['declination'])
    flux_vis = _native(cat['flux_vis_psf'])
    fluxerr_vis = _native(cat['fluxerr_vis_psf'])
    with np.errstate(invalid='ignore', divide='ignore'):
        cat_mag_vis = -2.5 * np.log10(np.where(flux_vis > 0, flux_vis, np.nan)) + 23.9
        cat_magerr_vis = 2.5 / np.log(10) * (fluxerr_vis / np.where(flux_vis > 0, flux_vis, np.nan))
    cat_fwhm = _native(cat['fwhm'])
    cat_ellip = _native(cat['ellipticity'])
    cat_phz = _native(cat['phz_classification'])
    cat_plp = _native(cat['point_like_prob'])
    cat_sp  = _native(cat['spurious_flag'])
    cat_vd  = _native(cat['vis_det'])

    t0 = time.time()
    idx_c, idx_d, sep_as = best_dao_per_cat(dao, cat_ra, cat_dec, EU_VIS_PIX_AS)
    print(f'  Best DAO per MER id within {EU_VIS_PIX_AS*1000:.0f} mas: '
          f'{len(idx_c):,} ({time.time()-t0:.1f}s)')

    if len(idx_c) == 0:
        print('  no matches'); return

    d = dao.iloc[idx_d].reset_index(drop=True)[DAO_COLS_KEEP + ['snr']]
    d = d.add_prefix('dao_VIS_')
    out = pd.DataFrame({
        'euclid_id'            : cat_id[idx_c],
        'cat_ra'               : cat_ra[idx_c],
        'cat_dec'              : cat_dec[idx_c],
        'cat_mag_VIS'          : cat_mag_vis[idx_c],
        'cat_magerr_VIS'       : cat_magerr_vis[idx_c],
        'cat_fwhm_pix'         : cat_fwhm[idx_c],
        'cat_ellipticity'      : cat_ellip[idx_c],
        'cat_phz_classification': cat_phz[idx_c],
        'cat_point_like_prob'  : cat_plp[idx_c],
        'cat_spurious_flag'    : cat_sp[idx_c],
        'cat_vis_det'          : cat_vd[idx_c],
        'cat_sep_arcsec_VIS'   : sep_as,
    })
    out = pd.concat([out, d], axis=1)

    p = OUT / 'cat_matched_Euclid_VIS.parquet'
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix('.csv'), index=False)
    print(f'  → {p}  ({len(out):,} rows, {len(out.columns)} cols)')


# ============================================================
# Euclid NISP  (3 bands Y/J/H → 1 row per MER object_id)
# ============================================================
def step3_euclid_nisp():
    print('\n========== Euclid NISP (Y/J/H → MER) ==========')

    cat = Table.read(CAT_DIR / 'Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits')
    cat_id   = _native(cat['object_id'])
    cat_ra   = _native(cat['right_ascension'])
    cat_dec  = _native(cat['declination'])
    cat_phz  = _native(cat['phz_classification'])
    cat_plp  = _native(cat['point_like_prob'])
    cat_fwhm = _native(cat['fwhm'])
    cat_ellip = _native(cat['ellipticity'])
    cat_sp   = _native(cat['spurious_flag'])

    cat_flux = {}
    cat_ferr = {}
    cat_mag  = {}
    cat_merr = {}
    for band, fcol, ecol in [('Y', 'flux_y_sersic', 'fluxerr_y_sersic'),
                             ('J', 'flux_j_sersic', 'fluxerr_j_sersic'),
                             ('H', 'flux_h_sersic', 'fluxerr_h_sersic')]:
        flx = _native(cat[fcol]); ferr = _native(cat[ecol])
        with np.errstate(invalid='ignore', divide='ignore'):
            mag = -2.5 * np.log10(np.where(flx > 0, flx, np.nan)) + 23.9
            mer = 2.5 / np.log(10) * (ferr / np.where(flx > 0, flx, np.nan))
        cat_flux[band] = flx; cat_ferr[band] = ferr
        cat_mag[band]  = mag; cat_merr[band] = mer
    print(f'  MER catalog: {len(cat_id):,}')

    band_match = {}
    for band in ('NIR_Y', 'NIR_J', 'NIR_H'):
        dao = pd.read_parquet(DAO_DIR / f'dao_{band}.parquet')
        dao['snr'] = dao['peak'] / dao['sky_std']
        print(f'  DAO {band} detections: {len(dao):,}')
        t0 = time.time()
        idx_c, idx_d, sep = best_dao_per_cat(dao, cat_ra, cat_dec, EU_NISP_PIX_AS)
        print(f'    best DAO per MER id ≤ {EU_NISP_PIX_AS*1000:.0f} mas: '
              f'{len(idx_c):,} ({time.time()-t0:.1f}s)')
        band_match[band] = (idx_c, idx_d, sep, dao)

    matched_cat_ids = set()
    for band in band_match:
        idx_c, _, _, _ = band_match[band]
        matched_cat_ids.update(idx_c.tolist())
    matched_cat_ids = np.array(sorted(matched_cat_ids), dtype=int)
    print(f'\n  Unique MER ids with ≥1 NISP DAO match: {len(matched_cat_ids):,}')

    out = pd.DataFrame({
        'euclid_id'             : cat_id[matched_cat_ids],
        'cat_ra'                : cat_ra[matched_cat_ids],
        'cat_dec'               : cat_dec[matched_cat_ids],
        'cat_phz_classification': cat_phz[matched_cat_ids],
        'cat_point_like_prob'   : cat_plp[matched_cat_ids],
        'cat_fwhm_pix'          : cat_fwhm[matched_cat_ids],
        'cat_ellipticity'       : cat_ellip[matched_cat_ids],
        'cat_spurious_flag'     : cat_sp[matched_cat_ids],
    })
    for short in ('Y', 'J', 'H'):
        band_label = f'NIR_{short}'
        out[f'cat_mag_{band_label}']    = cat_mag[short][matched_cat_ids]
        out[f'cat_magerr_{band_label}'] = cat_merr[short][matched_cat_ids]

    cat_id_to_pos = {int(cid): i for i, cid in enumerate(matched_cat_ids)}
    for band in ('NIR_Y', 'NIR_J', 'NIR_H'):
        idx_c, idx_d, sep, dao = band_match[band]
        pos = np.array([cat_id_to_pos[int(ci)] for ci in idx_c])
        for col in DAO_COLS_KEEP + ['snr']:
            out_col = f'dao_{band}_{col}'
            if col == 'tile':
                out[out_col] = pd.Series(['']*len(out), dtype=object)
            else:
                out[out_col] = np.nan
        out[f'dao_{band}_sep_arcsec'] = np.nan
        dao_rows = dao.iloc[idx_d].reset_index(drop=True)
        for col in DAO_COLS_KEEP + ['snr']:
            out_col = f'dao_{band}_{col}'
            out.loc[out.index[pos], out_col] = dao_rows[col].values
        out.loc[out.index[pos], f'dao_{band}_sep_arcsec'] = sep

    print('\n  Per-band DAO match presence:')
    for band in ('NIR_Y', 'NIR_J', 'NIR_H'):
        n = int(out[f'dao_{band}_ra'].notna().sum())
        print(f'    {band}: {n:,}')
    n_all3 = int(((out['dao_NIR_Y_ra'].notna()) & (out['dao_NIR_J_ra'].notna()) &
                  (out['dao_NIR_H_ra'].notna())).sum())
    print(f'    all 3 NISP bands: {n_all3:,}')

    p = OUT / 'cat_matched_Euclid_NISP.parquet'
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix('.csv'), index=False)
    print(f'  → {p}  ({len(out):,} rows, {len(out.columns)} cols)')


def main():
    step3_hst()
    step3_jwst()
    step3_euclid_vis()
    step3_euclid_nisp()
    print('\nDone.')


if __name__ == '__main__':
    main()
