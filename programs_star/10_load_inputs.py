#!/usr/bin/env python
"""
10_load_inputs.py — v01

Read the 5 aggregated COSMOS catalogs, apply per-catalog star-selection
rules, and write 4 per-mission star-candidate CSVs to
csvfiles_star/inputs/.  Downstream scripts (11_, 12_, ...) cross-match
across missions.

Star-selection rules (v01):

  HST ACS i-phot          mu_class == 2  AND  flags == 0
  Euclid MER DR1          phz_classification == 1  AND  spurious_flag == 0
  JWST COSMOS-Web v1.1    flag_star == True  (CW's conservative star flag)
  COSMOS2020 Farmer       lp_type == 1  OR  ACS_MU_CLASS == 2

Photometric conventions:
  - HST, CW, COSMOS2020 catalogs already provide AB mags in `mag_*` cols.
  - Euclid MER stores fluxes in microJanskies (AB ZP = 23.9) with no
    unit stamp; converted here.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

CATDIR = Path('/Volumes/exdisk1/data/catalog')
OUTDIR = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/inputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

UJY_ZP = 23.9   # AB mag = -2.5 log10(flux_uJy) + 23.9


def flux_to_mag(flux, fluxerr):
    """Convert microJansky flux to AB mag and propagated 1-sigma magerr."""
    flux = np.asarray(flux, dtype=np.float64)
    fluxerr = np.asarray(fluxerr, dtype=np.float64)
    mag = np.where(flux > 0, -2.5 * np.log10(np.where(flux > 0, flux, 1.0)) + UJY_ZP, np.nan)
    # 1-sigma mag error = (2.5/ln10) * (fluxerr/flux)
    magerr = np.where(flux > 0, (2.5 / np.log(10)) * (fluxerr / np.where(flux > 0, flux, 1.0)), np.nan)
    return mag, magerr


# ---------------- HST ACS i-phot ----------------
def load_hst():
    path = CATDIR / 'cosmos_acs_iphot_200709.fits'
    print(f'\n[HST]  reading {path.name}')
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        sel = (d['mu_class'] == 2) & (d['flags'] == 0)
        t = Table()
        t['hst_id']        = d['number'][sel].astype(np.int64)
        t['ra']            = d['ra'][sel]
        t['dec']           = d['dec'][sel]
        t['epoch']         = 2007.0                      # ACS-COSMOS centroid
        t['mag_F814W']     = d['mag_auto'][sel]
        t['mag_err_F814W'] = d['magerr_auto'][sel]
        t['mag_best_F814W']= d['mag_best'][sel]
        t['class_star']    = d['class_star'][sel]        # SExtractor stellarity 0-1
        t['mu_class']      = d['mu_class'][sel]
        t['flags']         = d['flags'][sel]
        t['fwhm_world']    = d['fwhm_world'][sel]        # arcsec
        t['kron_radius']   = d['kron_radius'][sel]
        t['flux_max']      = d['flux_max'][sel]
        t['field']         = d['field'][sel]             # ACS tile id
        t['star_flag_source'] = 'mu_class==2 & flags==0'
    print(f'        kept {len(t):,} of {len(d):,}  ({100*len(t)/len(d):.2f} %)')
    return t


# ---------------- Euclid MER DR1 ----------------
def load_euclid():
    path = CATDIR / 'cosmos_mer_dr1r1_minimal.fits'
    print(f'\n[Euclid]  reading {path.name}')
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        sel = (d['phz_classification'] == 1) & (d['spurious_flag'] == 0)
        t = Table()
        t['euclid_id']  = d['object_id'][sel].astype(np.int64)
        t['ra']         = d['right_ascension'][sel]
        t['dec']        = d['declination'][sel]
        t['epoch']      = 2024.5                        # Q1 epoch placeholder
        for band, fcol, ecol in [
            ('VIS',   'flux_vis_psf',    'fluxerr_vis_psf'),
            ('NIR_Y', 'flux_y_sersic',   'fluxerr_y_sersic'),
            ('NIR_J', 'flux_j_sersic',   'fluxerr_j_sersic'),
            ('NIR_H', 'flux_h_sersic',   'fluxerr_h_sersic'),
        ]:
            mag, magerr = flux_to_mag(d[fcol][sel], d[ecol][sel])
            t[f'mag_{band}']     = mag.astype(np.float32)
            t[f'mag_err_{band}'] = magerr.astype(np.float32)
        t['fwhm']               = d['fwhm'][sel]                  # arcsec
        t['mumax_minus_mag']    = d['mumax_minus_mag'][sel]       # star locus
        t['phz_classification'] = d['phz_classification'][sel]
        t['phz_star_prob']      = d['phz_star_prob'][sel]
        t['vis_det']            = d['vis_det'][sel]
        t['det_quality_flag']   = d['det_quality_flag'][sel]
        t['tile_index']         = d['tile_index'][sel]
        t['star_flag_source']   = 'phz_classification==1 & spurious_flag==0'
    print(f'        kept {len(t):,} of {len(d):,}  ({100*len(t)/len(d):.2f} %)')
    return t


# ---------------- JWST COSMOS-Web v1.1 ----------------
def load_jwst():
    path = CATDIR / 'COSMOSWeb_mastercatalog_v1.1.fits'
    print(f'\n[JWST CW]  reading {path.name} (ext 1 PHOT + ext 2 LEPHARE)')
    with fits.open(path, memmap=True) as h:
        d  = h[1].data
        dl = h[2].data
        sel = d['flag_star'].astype(bool)
        t = Table()
        t['jwst_id']  = d['id'][sel].astype(np.int64)
        t['tile']     = d['tile'][sel]
        t['ra']       = d['ra'][sel]
        t['dec']      = d['dec'][sel]
        t['epoch']    = 2024.0                          # CW NIRCam median epoch
        # JWST NIRCam + MIRI bands
        for band in ('f115w','f150w','f277w','f444w','f770w'):
            t[f'mag_{band.upper()}']     = d[f'mag_auto_{band}'][sel].astype(np.float32)
            t[f'mag_err_{band.upper()}'] = (1.0857 / np.where(d[f'snr_{band}'][sel] > 0,
                                                              d[f'snr_{band}'][sel], np.nan)).astype(np.float32)
            t[f'snr_{band.upper()}']     = d[f'snr_{band}'][sel].astype(np.float32)
        # CW-reprocessed HST F814W (their version, different from cosmos_acs_iphot)
        t['mag_F814W_cw']     = d['mag_auto_hst-f814w'][sel].astype(np.float32)
        t['snr_F814W_cw']     = d['snr_hst-f814w'][sel].astype(np.float32)
        # CW flags
        t['flag_star']      = d['flag_star'][sel]
        t['flag_star_hsc']  = d['flag_star_hsc'][sel]
        t['flag_blend']     = d['flag_blend'][sel]
        t['fwhm']           = d['fwhm'][sel]
        t['radius_sersic']  = d['radius_sersic'][sel]
        t['sersic']         = d['sersic'][sel]
        t['axratio_sersic'] = d['axratio_sersic'][sel]
        # LePhare star-template fit
        t['lphare_chi_star'] = dl['chi_star'][sel].astype(np.float32)
        t['lphare_mod_star'] = dl['mod_star'][sel].astype(np.int32)
        t['lphare_chi2_gal'] = dl['chi2_best'][sel].astype(np.float32)
        t['lphare_zfinal']   = dl['zfinal'][sel].astype(np.float32)
        t['star_flag_source'] = 'flag_star==True'
    print(f'        kept {len(t):,} of {len(d):,}  ({100*len(t)/len(d):.3f} %)')
    return t


# ---------------- COSMOS2020 Farmer ----------------
def load_cosmos2020():
    path = CATDIR / 'COSMOS2020_FARMER_R1_v2.2_p3.fits.gz'
    print(f'\n[COSMOS2020]  reading {path.name}')
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        sel = (d['lp_type'] == 1) | (d['ACS_MU_CLASS'] == 2)
        t = Table()
        t['c2020_id']     = d['ID'][sel].astype(np.int64)
        t['ra']           = d['ALPHA_J2000'][sel]
        t['dec']          = d['DELTA_J2000'][sel]
        t['epoch']        = 2015.0                       # COSMOS2020 imaging mean
        # core broadband mags
        for col in ('CFHT_u_MAG','HSC_g_MAG','HSC_r_MAG','HSC_i_MAG','HSC_z_MAG','HSC_y_MAG',
                    'UVISTA_Y_MAG','UVISTA_J_MAG','UVISTA_H_MAG','UVISTA_Ks_MAG',
                    'IRAC_CH1_MAG','IRAC_CH2_MAG','IRAC_CH3_MAG','IRAC_CH4_MAG',
                    'ACS_F814W_MAG'):
            err = col.replace('_MAG','_MAGERR')
            t[col.lower()]     = d[col][sel].astype(np.float32)
            t[err.lower()]     = d[err][sel].astype(np.float32) if err in d.names else np.nan
        # classifier columns
        t['lp_type']        = d['lp_type'][sel]
        t['lp_chis']        = d['lp_chis'][sel].astype(np.float32)
        t['lp_mods']        = d['lp_mods'][sel].astype(np.int32)
        t['acs_mu_class']   = d['ACS_MU_CLASS'][sel]
        t['ez_star_teff']   = d['ez_star_teff'][sel].astype(np.float32)
        t['ez_star_chi2']   = d['ez_star_min_chi2'][sel].astype(np.float32)
        t['flag_combined']  = d['FLAG_COMBINED'][sel]
        t['star_flag_source'] = 'lp_type==1 | ACS_MU_CLASS==2'
    print(f'        kept {len(t):,} of {len(d):,}  ({100*len(t)/len(d):.2f} %)')
    return t


def main():
    out = {
        'hst':         (load_hst,        'star_hst_v01.csv'),
        'euclid':      (load_euclid,     'star_euclid_v01.csv'),
        'jwst':        (load_jwst,       'star_jwst_v01.csv'),
        'cosmos2020':  (load_cosmos2020, 'star_cosmos2020_v01.csv'),
    }
    summary = []
    for key, (fn, fname) in out.items():
        t = fn()
        p = OUTDIR / fname
        t.write(p, format='csv', overwrite=True)
        summary.append((key, len(t), str(p), t.colnames[:6]))
        print(f'        → {p}   ({len(t):,} rows, {len(t.colnames)} cols)')

    print('\n' + '=' * 78)
    print(f'{"mission":<12} {"n_stars":>10}  output file')
    print('-' * 78)
    for key, n, p, _ in summary:
        print(f'{key:<12} {n:>10,}  {p}')


if __name__ == '__main__':
    main()
