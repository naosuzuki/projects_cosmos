#!/usr/bin/env python
"""
74_download_desi_spectra.py — for each master 4-way star with a DESI
match, download its spectrum from DESI DR1 (or fall back to EDR), and
save it to /Volumes/exdisk1/data/DESI/cosmos_{star,quasar,galaxy}/.

Adds a `desi_type` column to the master:
   'star' / 'quasar' / 'galaxy' / NaN
based on DESI's SPECTYPE classification ('STAR'/'QSO'/'GALAXY').

Output spectrum file (per target):
   /Volumes/exdisk1/data/DESI/cosmos_<type>/DESI-<targetid>.fits
   HDU 0 (PRIMARY) — header with TARGETID, SPECTYPE, Z, ZERR, ZWARN,
     SURVEY, PROGRAM, HEALPIX, DATA_RELEASE, RA, DEC, MASTER_PRI_ID
   HDU 1 (SPECTRUM) — BinTable WAVELENGTH, FLUX, IVAR (and MODEL, MASK
     where SPARCL provides them).

Also writes:
   csvfiles_star/master_stars_4way_with_desi.parquet   (updated, w/ desi_type)
   csvfiles_star/desi_dr1_download_log.parquet         (one row per attempt)
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

OUT  = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star')
DESI_ROOT = Path('/Volumes/exdisk1/data/DESI')
SUBDIRS = {'STAR': 'cosmos_star', 'QSO': 'cosmos_quasar', 'GALAXY': 'cosmos_galaxy'}
TYPE_MAP = {'STAR': 'star', 'QSO': 'quasar', 'GALAXY': 'galaxy'}
DATASETS = ['DESI-DR1', 'DESI-EDR']


def add_desi_type_column(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    sp = m['desi_spectype'].astype(str).str.strip().str.upper()
    m['desi_type'] = sp.map(TYPE_MAP).where(m['has_desi_spec'], None)
    return m


def save_spectrum(rec, master_row, dest: Path):
    """Save a SPARCL record to dest as a tiny FITS file."""
    hdr = fits.Header()
    hdr['EXTNAME'] = 'PRIMARY'
    hdr['TARGETID'] = (int(rec.get('specid', master_row['desi_targetid'])), 'DESI TARGETID')
    hdr['SPECTYPE'] = (str(rec.get('spectype', master_row.get('desi_spectype', ''))), 'DESI spectype')
    z = rec.get('redshift', master_row.get('desi_z', np.nan))
    hdr['Z']        = (float(z) if z is not None else np.nan, 'Best-fit redshift')
    zerr = rec.get('redshift_err', master_row.get('desi_zerr', np.nan))
    hdr['ZERR']     = (float(zerr) if zerr is not None else np.nan, 'Redshift error')
    zwarn = master_row.get('desi_zwarn', 0)
    hdr['ZWARN']    = (int(zwarn) if pd.notna(zwarn) else 0, 'ZWARN flag')
    hdr['SURVEY']   = str(master_row.get('desi_survey', ''))
    hdr['PROGRAM']  = str(master_row.get('desi_program', ''))
    hdr['HEALPIX']  = (int(master_row['desi_healpix']) if pd.notna(master_row.get('desi_healpix')) else -1, 'DESI HEALPIX')
    hdr['DATAREL']  = str(rec.get('data_release', ''))
    hdr['RA']       = float(master_row.get('cat_ra_jwst', np.nan))
    hdr['DEC']      = float(master_row.get('cat_dec_jwst', np.nan))
    hdr['JWSTID']   = (int(master_row['jwst_id']) if pd.notna(master_row.get('jwst_id')) else -1, 'Master JWST id')
    hdr['F814W']    = float(master_row.get('cat_mag_F814W_hst', np.nan))

    prim = fits.PrimaryHDU(header=hdr)

    wl = np.asarray(rec.get('wavelength'), dtype=np.float32)
    fl = np.asarray(rec.get('flux'),       dtype=np.float32)
    iv = rec.get('ivar')
    mo = rec.get('model')
    mk = rec.get('mask')
    cols = [fits.Column(name='WAVELENGTH', format='E', array=wl, unit='Angstrom'),
            fits.Column(name='FLUX',       format='E', array=fl, unit='1e-17 erg/s/cm2/A')]
    if iv is not None: cols.append(fits.Column(name='IVAR',  format='E', array=np.asarray(iv,np.float32)))
    if mo is not None: cols.append(fits.Column(name='MODEL', format='E', array=np.asarray(mo,np.float32)))
    if mk is not None: cols.append(fits.Column(name='MASK',  format='J', array=np.asarray(mk,np.int32)))
    spec_hdu = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    fits.HDUList([prim, spec_hdu]).writeto(dest, overwrite=True)


def main():
    t0 = time.time()
    print('Loading master ...')
    m = pd.read_parquet(OUT/'master_stars_4way_with_desi.parquet')
    m = add_desi_type_column(m)
    matched = m[m['has_desi_spec']].copy()
    print(f'matched master rows: {len(matched):,}')

    # set up dirs
    for sub in SUBDIRS.values():
        (DESI_ROOT/sub).mkdir(parents=True, exist_ok=True)

    # one-shot save the updated master
    m.to_parquet(OUT/'master_stars_4way_with_desi.parquet', index=False)
    m.to_csv(OUT/'master_stars_4way_with_desi.csv', index=False)
    print('master_stars_4way_with_desi updated with desi_type column.')

    # quick distribution
    print('\ndesi_type distribution (matched only):')
    print(matched['desi_type'].value_counts().to_string())

    # SPARCL fetch
    print('\nFetching spectra from SPARCL (DESI-DR1 then DESI-EDR) ...')
    from sparcl.client import SparclClient
    client = SparclClient()
    all_ids = matched['desi_targetid'].astype(np.int64).tolist()
    id_to_row = {int(matched['desi_targetid'].iat[i]): i for i in range(len(matched))}

    # request in batches to keep memory bounded
    BATCH = 500
    found = {}   # specid → (dataset, rec)
    for start in range(0, len(all_ids), BATCH):
        batch = all_ids[start:start+BATCH]
        for ds in DATASETS:
            need = [b for b in batch if b not in found]
            if not need: break
            try:
                r = client.retrieve_by_specid(specid_list=need, dataset_list=[ds],
                                              include=['specid','spectype','redshift','redshift_err',
                                                        'data_release','wavelength','flux','ivar','model','mask'])
            except Exception as e:
                print(f'[batch {start} {ds}] error: {repr(e)[:120]}')
                continue
            for rec in r.records:
                sid = int(rec.get('specid', -1))
                if sid in id_to_row and sid not in found:
                    found[sid] = (ds, rec)
        print(f'  processed batch starting {start}: cumulative found {len(found):,}/{len(all_ids):,}')

    print(f'\nFound {len(found):,} / {len(all_ids):,} spectra ({100*len(found)/len(all_ids):.1f}%)')

    # save each spectrum + build log
    log_rows = []
    for sid, ridx in id_to_row.items():
        row = matched.iloc[ridx]
        sp = str(row['desi_spectype']).strip().upper()
        sub = SUBDIRS.get(sp)
        if sub is None:
            log_rows.append({'targetid': sid, 'status': 'unknown_spectype', 'dataset': None,
                             'spectype': sp, 'survey': row['desi_survey']})
            continue
        if sid not in found:
            log_rows.append({'targetid': sid, 'status': 'not_in_dr1_or_edr', 'dataset': None,
                             'spectype': sp, 'survey': row['desi_survey']})
            continue
        ds, rec = found[sid]
        dest = DESI_ROOT/sub/f'DESI-{sid}.fits'
        try:
            save_spectrum(rec, row, dest)
            log_rows.append({'targetid': sid, 'status': 'saved', 'dataset': ds,
                             'spectype': sp, 'survey': row['desi_survey'],
                             'path': str(dest)})
        except Exception as e:
            log_rows.append({'targetid': sid, 'status': f'save_error: {repr(e)[:80]}',
                             'dataset': ds, 'spectype': sp, 'survey': row['desi_survey']})

    log = pd.DataFrame(log_rows)
    log.to_parquet(OUT/'desi_dr1_download_log.parquet', index=False)
    log.to_csv(OUT/'desi_dr1_download_log.csv', index=False)
    print('\nDownload log saved.')
    print('Status counts:')
    print(log['status'].value_counts().to_string())
    print('\nBy type and survey (saved only):')
    saved = log[log['status']=='saved']
    print(pd.crosstab(saved['spectype'], saved['survey']))

    # counts in each subdir
    print('\nSpectra written:')
    for sp, sub in SUBDIRS.items():
        d = DESI_ROOT/sub
        n = sum(1 for _ in d.glob('*.fits'))
        print(f'  {sub:<18} {n:,}')

    print(f'\nWall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
