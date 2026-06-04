#!/usr/bin/env python
"""
20_footprints.py  —  Step 1: per-tile footprints, per-band footprints,
                              common-coverage region, and coverage lookup.

Reads the WCS header of every COSMOS mosaic and computes its on-sky
bounding polygon.  Aggregates per band (union over tiles) and computes
the all-band intersection (the region where every filter has coverage).
Finally, joins coverage-per-band columns onto each catalog star CSV
written by 10_load_inputs.py.

A10 f115w workaround:
  Parent dir's `..._A10_v1.0_i2d.fits.gz` is a corrupted gzip.  For that
  tile + band only we substitute the WCS from scidir/`..._A10_v0_8_sci.fits`.

Outputs (under csvfiles_star/footprints/):
  tile_polygons.csv   : mission, band, tile, file, 4 corners + center
  band_footprints.wkt : 9 lines of WKT, one polygon per band
  common_footprint.wkt
  coverage_lookup.csv : for every star in inputs/*v01.csv → cov_<band>
"""
from __future__ import annotations
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely import wkt as shapely_wkt

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star' / 'footprints'
INP  = ROOT / 'csvfiles_star' / 'inputs'
OUT.mkdir(parents=True, exist_ok=True)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

BANDS_ORDER = ['F814W', 'F115W', 'F150W', 'F277W', 'F444W',
               'VIS', 'NIR-Y', 'NIR-J', 'NIR-H']


def corners_from_header(hdr) -> tuple:
    """Return a 4-point shapely Polygon in (ra, dec) from a FITS header.
    Uses pixel corners (0.5, 0.5) → (N1+0.5, N2+0.5)."""
    w = WCS(hdr)
    n1 = hdr.get('NAXIS1') or hdr.get('ZNAXIS1')
    n2 = hdr.get('NAXIS2') or hdr.get('ZNAXIS2')
    if n1 is None or n2 is None:
        return None
    px = np.array([0.5, n1 + 0.5, n1 + 0.5, 0.5])
    py = np.array([0.5, 0.5, n2 + 0.5, n2 + 0.5])
    sky = w.all_pix2world(px, py, 1)
    ra, dec = sky[0], sky[1]
    # close polygon
    poly = sg.Polygon(list(zip(ra, dec)))
    return poly, (ra, dec), w


def open_sci_header(path: Path):
    """Open SCI header. JWST i2d has SCI extension; HST sci.fits has primary;
    Euclid BGSUB-MOSAIC has SCI extension (or primary depending on version)."""
    with fits.open(path, memmap=True) as h:
        # find first ImageHDU with valid WCS
        for hdu in h:
            hdr = hdu.header
            if hdr.get('NAXIS', 0) >= 2 and ('CRVAL1' in hdr or 'CRPIX1' in hdr):
                return hdr.copy()
    return None


# ------------------------------------------------------------------
# Enumerate every mosaic file we want to footprint
# ------------------------------------------------------------------

def enumerate_files():
    files = []

    # HST: acs_I_030mas_<NNN>_sci.fits  (skip .gz, use .fits)
    pat = re.compile(r'acs_I_030mas_(\d+)_sci\.fits$')
    for p in sorted(HST_DIR.glob('acs_I_030mas_*_sci.fits')):
        m = pat.search(p.name)
        if m:
            files.append(dict(mission='HST',  band='F814W', tile=m.group(1), path=str(p)))

    # JWST: parent dir .fits per (band, tile).  Skip the A10 f115w (corrupt gz);
    # use scidir/..._A10_v0_8_sci.fits for that one.
    jwst_pat = re.compile(r'mosaic_nircam_(f\d+w)_COSMOS-Web_30mas_([AB]\d+)_v1\.0_i2d\.fits$')
    for p in sorted(JWST_DIR.glob('mosaic_nircam_*_i2d.fits')):
        m = jwst_pat.search(p.name)
        if not m: continue
        band = m.group(1).upper()
        tile = m.group(2)
        files.append(dict(mission='JWST', band=band, tile=tile, path=str(p)))
    # A10 f115w fallback
    a10 = JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    if a10.exists():
        files.append(dict(mission='JWST', band='F115W', tile='A10', path=str(a10)))

    # Euclid: 4 bands × ~60 tiles, sometimes reprocessed twice.
    # Pick the LATEST processing per (band, tile).
    euc_pat = re.compile(r'EUC_MER_BGSUB-MOSAIC-([A-Z\-]+)_TILE(\d+)-([0-9A-F]+)_(\d{8}T\d{6}\.\d+Z)_00\.00\.fits$')
    latest = {}
    for p in sorted(EUCLID_DIR.glob('*.fits')):
        m = euc_pat.search(p.name)
        if not m: continue
        band, tile, hsh, ts = m.group(1), m.group(2), m.group(3), m.group(4)
        key = (band, tile)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, p)
    for (band, tile), (ts, p) in latest.items():
        files.append(dict(mission='Euclid', band=band, tile=tile, path=str(p)))

    return files


# ------------------------------------------------------------------
# Build footprints
# ------------------------------------------------------------------

def main():
    files = enumerate_files()
    print(f'Enumerated {len(files)} mosaic files.')

    rows = []
    band_polys = {b: [] for b in BANDS_ORDER}
    for i, f in enumerate(files):
        try:
            hdr = open_sci_header(Path(f['path']))
            if hdr is None:
                print(f'  [WARN] no WCS header in {f["path"]}')
                continue
            result = corners_from_header(hdr)
            if result is None:
                continue
            poly, (ra, dec), w = result
            cdelt = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0  # arcsec/pix
            rows.append(dict(
                mission=f['mission'], band=f['band'], tile=f['tile'],
                file=Path(f['path']).name,
                ra_c1=ra[0], dec_c1=dec[0], ra_c2=ra[1], dec_c2=dec[1],
                ra_c3=ra[2], dec_c3=dec[2], ra_c4=ra[3], dec_c4=dec[3],
                ra_cen=float(np.mean(ra)), dec_cen=float(np.mean(dec)),
                pix_scale_as=cdelt,
                n1=hdr.get('NAXIS1') or hdr.get('ZNAXIS1'),
                n2=hdr.get('NAXIS2') or hdr.get('ZNAXIS2'),
            ))
            band_polys[f['band']].append(poly)
            if (i + 1) % 50 == 0:
                print(f'  read {i+1}/{len(files)} headers')
        except Exception as e:
            print(f'  [ERR] {f["path"]}: {e}')
    print(f'  read {len(rows)}/{len(files)} headers')

    # Save per-tile catalog
    tile_df = pd.DataFrame(rows)
    tile_path = OUT / 'tile_polygons.csv'
    tile_df.to_csv(tile_path, index=False)
    print(f'\nWrote {tile_path}  ({len(tile_df)} rows)')

    # Per-band footprint = union of tile polygons
    band_fp = {}
    print('\nPer-band footprints (deg²):')
    for b in BANDS_ORDER:
        if not band_polys[b]:
            print(f'  {b:<8} no tiles')
            continue
        u = unary_union(band_polys[b])
        band_fp[b] = u
        # crude area in deg² (RA*cos(dec) correction not applied — fine for inventory)
        area = u.area * np.cos(np.deg2rad(2.2))  # COSMOS centered ~ +2.2
        print(f'  {b:<8} n_tiles={len(band_polys[b]):3d}  bbox_area≈{u.area:.4f}  cos-deg² ≈ {area:.4f}')

    # Write band footprints as WKT
    wkt_path = OUT / 'band_footprints.wkt'
    with open(wkt_path, 'w') as fh:
        for b in BANDS_ORDER:
            if b in band_fp:
                fh.write(f'{b}\t{shapely_wkt.dumps(band_fp[b], rounding_precision=6)}\n')
    print(f'\nWrote {wkt_path}')

    # All-band intersection (common region)
    common = None
    for b in BANDS_ORDER:
        if b not in band_fp:
            print(f'  [warn] band {b} missing — common-region undefined')
            return
        common = band_fp[b] if common is None else common.intersection(band_fp[b])
    common_path = OUT / 'common_footprint.wkt'
    with open(common_path, 'w') as fh:
        fh.write(shapely_wkt.dumps(common, rounding_precision=6) + '\n')
    print(f'Wrote {common_path}   (area≈{common.area*np.cos(np.deg2rad(2.2)):.4f} deg²)')

    # Coverage lookup for each input star CSV
    print('\nBuilding coverage lookup for star CSVs:')
    all_cov = []
    for csv_path in sorted(INP.glob('star_*_v01.csv')):
        df = pd.read_csv(csv_path)
        df['_src_csv'] = csv_path.name
        for b in BANDS_ORDER:
            if b not in band_fp:
                df[f'cov_{b}'] = False
                continue
            fp = band_fp[b]
            # vectorised point-in-polygon using shapely.contains via STRtree is fastest,
            # but shapely 2.x has shapely.contains_xy
            try:
                from shapely import contains_xy
                df[f'cov_{b}'] = contains_xy(fp, df['ra'].values, df['dec'].values)
            except ImportError:
                df[f'cov_{b}'] = [fp.contains(sg.Point(r, d)) for r, d in zip(df['ra'], df['dec'])]
        # Count multi-band coverage
        cov_cols = [f'cov_{b}' for b in BANDS_ORDER]
        df['n_cov'] = df[cov_cols].sum(axis=1)
        df['has_all_9'] = df['n_cov'] == 9
        print(f'  {csv_path.name:<32}  n={len(df):>7,d}  in_all_9={df["has_all_9"].sum():>6,d}  any_no_jwst_coverage={(~df["cov_F115W"]).sum():,}')
        all_cov.append(df[['_src_csv', 'ra', 'dec'] + cov_cols + ['n_cov', 'has_all_9']].rename(columns={'_src_csv': 'src_csv'}))

    # Save a combined lookup
    cov_df = pd.concat(all_cov, ignore_index=True)
    cov_path = OUT / 'coverage_lookup.csv'
    cov_df.to_csv(cov_path, index=False)
    print(f'\nWrote {cov_path}   ({len(cov_df):,} rows)')

    # Also write the per-input augmented files (with cov_* + n_cov + has_all_9 cols)
    AUG = ROOT / 'csvfiles_star' / 'inputs_with_coverage'
    AUG.mkdir(exist_ok=True)
    for csv_path in sorted(INP.glob('star_*_v01.csv')):
        df = pd.read_csv(csv_path)
        for b in BANDS_ORDER:
            if b in band_fp:
                try:
                    from shapely import contains_xy
                    df[f'cov_{b}'] = contains_xy(band_fp[b], df['ra'].values, df['dec'].values)
                except ImportError:
                    df[f'cov_{b}'] = [band_fp[b].contains(sg.Point(r, d)) for r, d in zip(df['ra'], df['dec'])]
            else:
                df[f'cov_{b}'] = False
        df['n_cov'] = df[[f'cov_{b}' for b in BANDS_ORDER]].sum(axis=1)
        outp = AUG / csv_path.name
        df.to_csv(outp, index=False)
        print(f'  augmented → {outp}')


if __name__ == '__main__':
    main()
