#!/usr/bin/env python
"""
46_orphan_galaxy_veto.py — Use catalog FWHM/ellipticity + survey-native
star/galaxy classifiers to veto galaxy-core orphans.

Catalogs cross-matched (0.5″ tolerance):
  COSMOS-Web v1.1   : flag_star, fwhm, axratio_sersic, e1/e2, radius_sersic
  Euclid MER DR1    : phz_classification, fwhm, ellipticity, point_like_prob,
                       sersic_sersic_vis_radius
  HST ACS i-phot    : class_star, mu_class, fwhm_image, a/b_image (→ ellipticity)
  COSMOS2020 Farmer : lp_type, ACS_MU_CLASS, ACS_A_WORLD/B_WORLD, ACS_FWHM_WORLD

Galaxy criteria (any one of these → veto as galaxy):
  - phz_classification == 2          (Euclid PHZ-galaxy)
  - mu_class == 1                    (ACS-galaxy)
  - ACS_MU_CLASS == 1                (COSMOS2020 ACS-galaxy)
  - lp_type == 0                     (COSMOS2020 LePhare-galaxy)
  - flag_star == False AND fwhm > 1.4 × PSF (compact galaxy in CW)
  - Euclid fwhm > 2.5 px  (= 0.25″ at 100 mas, well above 0.194″ VIS PSF)
  - Euclid ellipticity > 0.30
  - HST ACS class_star < 0.50

Also retains catalog data on each orphan for later inspection.

Input
-----
  csvfiles_star/orphans_v5.parquet  (HST∩JWST common region, sat_v3 vetoed)

Output
------
  csvfiles_star/orphans_v6.parquet                 (galaxy-vetoed)
  csvfiles_star/orphans_v6_with_flags.parquet      (all rows + veto flags)
  csvfiles_star/orphans_v6_summary.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT  = Path('/Users/suzuki/github/projects_cosmos')
OUT   = ROOT / 'csvfiles_star'
CATDIR = Path('/Volumes/exdisk1/data/catalog')

MATCH_AS = 0.5
PSF_AS = {'F814W': 0.134, 'F115W': 0.057, 'VIS': 0.194, 'NIR_J': 0.537}
EU_PIX = 0.10
ACS_PIX = 0.03

# Galaxy thresholds
EU_FWHM_GAL_PIX     = 2.5      # > 0.25″ in VIS (PSF ≈ 0.194″ at 100 mas)
EU_ELLIP_GAL        = 0.30
HST_CLASS_STAR_GAL  = 0.50     # SExtractor class_star: < this = galaxy-like
CW_FWHM_GAL_FACTOR  = 1.5      # fwhm > 1.5 × PSF → extended in CW


def load_eu():
    print('Loading Euclid MER DR1...')
    eu = Table.read(CATDIR / 'Euclid/COSMOS/cosmos_mer_dr1r1_minimal.fits')
    return pd.DataFrame({
        'ra':            np.asarray(eu['right_ascension']),
        'dec':           np.asarray(eu['declination']),
        'eu_fwhm_px':    np.asarray(eu['fwhm']),
        'eu_ellip':      np.asarray(eu['ellipticity']),
        'eu_phz':        np.asarray(eu['phz_classification']),
        'eu_point_prob': np.asarray(eu['point_like_prob']),
        'eu_sersic_r':   np.asarray(eu['sersic_sersic_vis_radius']),
        'eu_object_id':  np.asarray(eu['object_id']),
    })


def load_hst_acs():
    print('Loading HST ACS i-phot...')
    ac = Table.read(CATDIR / 'HST/COSMOS/cosmos_acs_iphot_200709.fits')
    a = np.asarray(ac['a_image']); b = np.asarray(ac['b_image'])
    ellip = 1.0 - b / np.where(a > 0, a, np.nan)
    return pd.DataFrame({
        'ra':           np.asarray(ac['ra']),
        'dec':          np.asarray(ac['dec']),
        'hst_class_star': np.asarray(ac['class_star']),
        'hst_mu_class': np.asarray(ac['mu_class']),
        'hst_fwhm_as':  np.asarray(ac['fwhm_image']) * ACS_PIX,
        'hst_ellip':    ellip,
        'hst_flags':    np.asarray(ac['flags']),
        'hst_mag_auto': np.asarray(ac['mag_auto']),
    })


def load_cw():
    print('Loading CW v1.1 ext 1 PHOT...')
    with fits.open(CATDIR / 'JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits', memmap=True) as h:
        d = h[1].data
        return pd.DataFrame({
            'ra':              np.asarray(d['ra']),
            'dec':             np.asarray(d['dec']),
            'cw_flag_star':    np.asarray(d['flag_star']),
            'cw_flag_blend':   np.asarray(d['flag_blend']),
            'cw_fwhm':         np.asarray(d['fwhm']),       # arcsec (CW convention)
            'cw_sersic_n':     np.asarray(d['sersic']),
            'cw_axratio':      np.asarray(d['axratio_sersic']),
            'cw_radius_sersic': np.asarray(d['radius_sersic']),
            'cw_mag_f115w':    np.asarray(d['mag_auto_f115w']),
        })


def load_c2020():
    print('Loading COSMOS2020 Farmer...')
    with fits.open(CATDIR / 'COSMOS2020/COSMOS/COSMOS2020_FARMER_R1_v2.2_p3.fits.gz', memmap=True) as h:
        d = h[1].data
        return pd.DataFrame({
            'ra':            np.asarray(d['ALPHA_J2000']),
            'dec':           np.asarray(d['DELTA_J2000']),
            'c20_lp_type':   np.asarray(d['lp_type']),
            'c20_acs_mu':    np.asarray(d['ACS_MU_CLASS']),
            'c20_acs_a':     np.asarray(d['ACS_A_WORLD']) * 3600.0,  # deg→arcsec
            'c20_acs_b':     np.asarray(d['ACS_B_WORLD']) * 3600.0,
            'c20_acs_fwhm':  np.asarray(d['ACS_FWHM_WORLD']) * 3600.0,
        })


def nearest_match(orph_ra, orph_dec, cat_ra, cat_dec, max_as=MATCH_AS):
    """For each orphan, return (idx_cat, sep_as) of nearest within max_as.
    idx_cat == -1 if none within radius."""
    co = SkyCoord(orph_ra * u.deg, orph_dec * u.deg)
    cc = SkyCoord(cat_ra  * u.deg, cat_dec  * u.deg)
    idx_o, idx_c, sep, _ = search_around_sky(co, cc, max_as * u.arcsec)
    if len(idx_o) == 0:
        return np.full(len(orph_ra), -1, dtype=np.int64), np.full(len(orph_ra), np.nan)
    df = pd.DataFrame({'o': idx_o, 'c': idx_c, 's': sep.arcsec})
    df = df.sort_values(['o', 's']).drop_duplicates('o', keep='first')
    out_idx = np.full(len(orph_ra), -1, dtype=np.int64)
    out_sep = np.full(len(orph_ra), np.nan)
    out_idx[df['o'].values] = df['c'].values
    out_sep[df['o'].values] = df['s'].values
    return out_idx, out_sep


def main():
    orph = pd.read_parquet(OUT / 'orphans_v5.parquet')
    print(f'orphans_v5: {len(orph):,}')

    eu  = load_eu()
    hst = load_hst_acs()
    cw  = load_cw()
    c20 = load_c2020()

    ra  = orph['ra'].values
    dec = orph['dec'].values

    print('\nCross-matching to catalogs (0.5"):')
    eu_idx,  eu_sep  = nearest_match(ra, dec, eu['ra'].values,  eu['dec'].values)
    print(f'  Euclid MER  : {(eu_idx  >= 0).sum():,}/{len(orph):,} matched')
    hst_idx, hst_sep = nearest_match(ra, dec, hst['ra'].values, hst['dec'].values)
    print(f'  HST ACS     : {(hst_idx >= 0).sum():,}/{len(orph):,} matched')
    cw_idx,  cw_sep  = nearest_match(ra, dec, cw['ra'].values,  cw['dec'].values)
    print(f'  CW v1.1     : {(cw_idx  >= 0).sum():,}/{len(orph):,} matched')
    c20_idx, c20_sep = nearest_match(ra, dec, c20['ra'].values, c20['dec'].values)
    print(f'  COSMOS2020  : {(c20_idx >= 0).sum():,}/{len(orph):,} matched')

    # Build augmented dataframe
    def _take(catdf, idx, col):
        out = np.full(len(idx), np.nan, dtype=float)
        ok = idx >= 0
        out[ok] = catdf[col].values[idx[ok]]
        return out

    orph = orph.copy()
    orph['eu_sep']     = eu_sep
    orph['eu_fwhm_px'] = _take(eu,  eu_idx,  'eu_fwhm_px')
    orph['eu_ellip']   = _take(eu,  eu_idx,  'eu_ellip')
    orph['eu_phz']     = _take(eu,  eu_idx,  'eu_phz')
    orph['eu_point_prob'] = _take(eu, eu_idx, 'eu_point_prob')
    orph['eu_sersic_r']= _take(eu,  eu_idx,  'eu_sersic_r')

    orph['hst_sep']        = hst_sep
    orph['hst_class_star'] = _take(hst, hst_idx, 'hst_class_star')
    orph['hst_mu_class']   = _take(hst, hst_idx, 'hst_mu_class')
    orph['hst_fwhm_as']    = _take(hst, hst_idx, 'hst_fwhm_as')
    orph['hst_ellip']      = _take(hst, hst_idx, 'hst_ellip')

    orph['cw_sep']         = cw_sep
    orph['cw_flag_star']   = _take(cw, cw_idx, 'cw_flag_star')
    orph['cw_fwhm']        = _take(cw, cw_idx, 'cw_fwhm')
    orph['cw_sersic_n']    = _take(cw, cw_idx, 'cw_sersic_n')
    orph['cw_axratio']     = _take(cw, cw_idx, 'cw_axratio')
    orph['cw_radius_sersic'] = _take(cw, cw_idx, 'cw_radius_sersic')

    orph['c20_sep']        = c20_sep
    orph['c20_lp_type']    = _take(c20, c20_idx, 'c20_lp_type')
    orph['c20_acs_mu']     = _take(c20, c20_idx, 'c20_acs_mu')
    orph['c20_acs_a']      = _take(c20, c20_idx, 'c20_acs_a')
    orph['c20_acs_b']      = _take(c20, c20_idx, 'c20_acs_b')
    orph['c20_acs_fwhm']   = _take(c20, c20_idx, 'c20_acs_fwhm')

    # ---- Galaxy veto rules ----
    veto_eu_phz   = (orph['eu_phz']  == 2)
    veto_eu_morph = (
        (orph['eu_fwhm_px'] > EU_FWHM_GAL_PIX) |
        (orph['eu_ellip']   > EU_ELLIP_GAL) |
        (orph['eu_sersic_r'] * EU_PIX > 0.30)
    )
    veto_hst_mu      = (orph['hst_mu_class'] == 1)
    veto_hst_class   = (orph['hst_class_star'] < HST_CLASS_STAR_GAL) & \
                        (orph['hst_class_star'].notna())
    veto_hst_morph   = (orph['hst_fwhm_as'] > 1.5 * PSF_AS['F814W']) | \
                        (orph['hst_ellip']  > 0.30)
    veto_cw_flag     = (orph['cw_flag_star'] == 0)  # not-flagged-as-star
    veto_cw_morph    = (orph['cw_fwhm'] > CW_FWHM_GAL_FACTOR * 0.06)  # 6 PSF-FWHM in CW ~ 0.06 arcsec
    veto_c20_lp      = (orph['c20_lp_type']  == 0)
    veto_c20_mu      = (orph['c20_acs_mu']   == 1)
    veto_c20_morph   = (orph['c20_acs_fwhm'] > 1.5 * PSF_AS['F814W'])

    # Combine
    orph['veto_eu_gal']   = (veto_eu_phz.fillna(False) | veto_eu_morph.fillna(False))
    orph['veto_hst_gal']  = (veto_hst_mu.fillna(False) | veto_hst_class.fillna(False) |
                              veto_hst_morph.fillna(False))
    orph['veto_cw_gal']   = (veto_cw_flag.fillna(False) & veto_cw_morph.fillna(False))
    orph['veto_c20_gal']  = (veto_c20_lp.fillna(False)  | veto_c20_mu.fillna(False) |
                              veto_c20_morph.fillna(False))
    orph['veto_galaxy']   = (orph['veto_eu_gal'] | orph['veto_hst_gal'] |
                              orph['veto_cw_gal'] | orph['veto_c20_gal'])

    # Counts
    print('\nGalaxy-veto breakdown (per-source counts; ORed):')
    print(f'  Euclid PHZ/morph   : {int(orph["veto_eu_gal"].sum()):,}')
    print(f'  HST mu/cls/morph   : {int(orph["veto_hst_gal"].sum()):,}')
    print(f'  CW flag_star+fwhm  : {int(orph["veto_cw_gal"].sum()):,}')
    print(f'  COSMOS2020 type/mu : {int(orph["veto_c20_gal"].sum()):,}')
    print(f'  ANY veto            : {int(orph["veto_galaxy"].sum()):,}')

    orph.to_parquet(OUT / 'orphans_v6_with_flags.parquet', index=False)
    v6 = orph[~orph['veto_galaxy']].copy()
    v6.to_parquet(OUT / 'orphans_v6.parquet', index=False)
    print(f'\norphans_v6 (after galaxy veto): {len(v6):,}  (was {len(orph):,})')

    # By category
    print('\nBy found_in:')
    print(v6.groupby('found_in').size().to_string())

    summary = []
    for k, g in v6.groupby('found_in'):
        summary.append(dict(found_in=k, n_v5=int((orph['found_in']==k).sum()),
                            n_v6=len(g),
                            galaxy_vetoed=int((orph['found_in']==k).sum() - len(g))))
    pd.DataFrame(summary).to_csv(OUT / 'orphans_v6_summary.csv', index=False)
    print(f'\nWrote {OUT/"orphans_v6_summary.csv"}')


if __name__ == '__main__':
    main()
