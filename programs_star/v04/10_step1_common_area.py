#!/usr/bin/env python
"""
10_step1_common_area.py — STEP 1 of the v04 cross-mission stellar catalog.

Build the 4-way HST ∩ JWST ∩ Euclid VIS ∩ Euclid NISP common-area mask
from the actual pixel data (sci + err + wht of every mosaic), plus a
persistent RA/Dec → tile lookup table that downstream steps consume
without re-opening any FITS file.

Reference WCS:
  JWST F115W (30 mas, Gaia-DR3-tied through the COSMOS-Web v0.8
  pipeline).  All other-mission masks are reprojected onto F115W's
  per-tile pixel grids when intersecting; the global QC mask is on a
  separate synthetic 1″ COSMOS WCS for browse only.

Data locations (also recorded in
~/github/papers/26_jwsteuclidhst_note/ms.tex Appendix A):
  HST/ACS F814W 2003–2005   /Volumes/exdisk1/data/HST/COSMOS_ACS2005/
    mosaic_cosmos_web_2024jan_30mas_tile_B{1..10}_hst_acs_wfc_f814w_{drz,err,wht}.fits
  JWST NIRCam F115/F150/F277/F444W  /Volumes/exdisk1/data/JWST/COSMOS_v0.8/
    mosaic_nircam_<filter>_COSMOS-Web_30mas_<tile>_v1.0_i2d.fits
    [exception: F115W A10 uses scidir/ separate sci+err+wht files]
  Euclid VIS + NISP-Y/J/H   /Volumes/exdisk1/data/Euclid/COSMOS_DR1/
    EUC_MER_BGSUB-MOSAIC-<band>_TILE<id>-<hex>_<time>_00.00.fits

Outputs (csvfiles_star/v04/):
  tile_lookup.parquet
      One row per (mission, filter, tile) record: path, NAXIS, bounding
      box (ra_min/max, dec_min/max), 4-corner sky polygon, native pixel
      scale, file_format ("multi_extension" | "separate_sci_wht_err" |
      "single_hdu"), and the path to its coverage mask.

  coverage/<mission>_<filter>_<tile>.fits.gz
      Per-tile coverage mask on each mosaic's native WCS, uint8.
      Bit = 1 iff:
        sci ≠ NaN  AND  wht > 0  AND  err > 0  AND  err ≠ NaN  AND
        pixel is at least 10 pixels from any edge (drizzle-rim erosion).

  common_area_4way_1arcsec.fits.gz
      Global 4-way common-area mask on a synthetic COSMOS WCS at 1″
      pixel scale, ~3 000 × 3 000 px.  Used for footprint browse/QC,
      not for source-level work.

Decision points locked in the project plan (ms.tex Appendix B §B.2):
  Q1.1  edge criterion : sci≠NaN ∧ wht>0 ∧ err>0 ∧ err≠NaN
  Q1.2  edge erosion   : 10 native pixels per mission's mask
  Q1.3  A10 special    : F115W only; F150/F277/F444W A10 = main dir
  Q1.4  storage        : per-tile native masks + global 1″ downsampled
"""
from __future__ import annotations
import argparse
import gzip
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.ndimage import binary_erosion

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
HST_DIR     = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
JWST_DIR    = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCIDIR = JWST_DIR / 'scidir'
EUCLID_DIR  = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

OUT_DIR     = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
COV_DIR     = OUT_DIR / 'coverage'

# Edge criterion + erosion (Q1.1, Q1.2)
EROSION_NATIVE_PX = 10

# Global QC mask grid (Q1.4)
GLOBAL_WCS_CENTER = (150.1, 2.2)     # COSMOS field centre (deg)
GLOBAL_WCS_SIZE_DEG = 1.0            # square half-size: ±0.5° around centre
GLOBAL_PIXEL_SCALE_ARCSEC = 1.0

# ──────────────────────────────────────────────────────────────────────
# Tile record
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TileRecord:
    mission: str            # 'HST', 'JWST', 'Euclid'
    filter:  str            # 'F814W', 'F115W', 'VIS', 'NIR-H', ...
    tile_id: str            # 'B1', 'A10', 'TILE101538497', ...
    path:    str            # for multi_extension : single FITS path
                            # for separate_sci_wht_err : sci path (other two
                            #     are derived from naming)
                            # for single_hdu (Euclid) : the file
    file_format: str        # 'multi_extension' | 'separate_sci_wht_err' | 'single_hdu'
    naxis1: int = 0
    naxis2: int = 0
    pixscale_mas: float = 0.0
    ra_min: float = np.nan
    ra_max: float = np.nan
    dec_min: float = np.nan
    dec_max: float = np.nan
    corners_ra:  list[float] = field(default_factory=list)
    corners_dec: list[float] = field(default_factory=list)
    coverage_mask_path: str = ''
    n_pix_total: int = 0
    n_pix_observed: int = 0       # passes sci+err+wht criterion
    n_pix_after_erosion: int = 0  # final mask True count

    def as_row(self) -> dict:
        return {
            'mission': self.mission,
            'filter':  self.filter,
            'tile_id': self.tile_id,
            'path':    self.path,
            'file_format': self.file_format,
            'naxis1':  self.naxis1,
            'naxis2':  self.naxis2,
            'pixscale_mas': self.pixscale_mas,
            'ra_min':  self.ra_min,
            'ra_max':  self.ra_max,
            'dec_min': self.dec_min,
            'dec_max': self.dec_max,
            'corners_ra':  self.corners_ra,
            'corners_dec': self.corners_dec,
            'coverage_mask_path': self.coverage_mask_path,
            'n_pix_total': self.n_pix_total,
            'n_pix_observed': self.n_pix_observed,
            'n_pix_after_erosion': self.n_pix_after_erosion,
        }


# ──────────────────────────────────────────────────────────────────────
# Discovery — list every per-tile dataset
# ──────────────────────────────────────────────────────────────────────
def discover_hst() -> list[TileRecord]:
    """HST 2005-era F814W mosaics drizzled onto both JWST tile grids:
       - Apr-2023 reduction (`2023apr` in filename) → A1..A10 (southern half)
       - Jan-2024 reduction (`2024jan` in filename) → B1..B10 (northern half)

    Both share the same original 2003-2005 COSMOS ACS observations and
    the same separate sci/wht/err FITS layout.  Discovery globs the
    directory so any future reduction (e.g., 2024jun) is picked up
    automatically without code changes.
    """
    out = []
    pat = re.compile(
        r'^mosaic_cosmos_web_(\d{4}(?:jan|feb|mar|apr|may|jun|'
        r'jul|aug|sep|oct|nov|dec))_30mas_tile_([AB]\d+)'
        r'_hst_acs_wfc_f814w_drz\.fits$'
    )
    for f in sorted(HST_DIR.glob(
            'mosaic_cosmos_web_*_30mas_tile_*_hst_acs_wfc_f814w_drz.fits')):
        m = pat.match(f.name)
        if not m:
            continue
        # epoch = m.group(1)  # e.g. '2023apr' / '2024jan' — not stored
        tile = m.group(2)     # e.g. 'A3' / 'B5'
        out.append(TileRecord('HST', 'F814W', tile, str(f),
                              file_format='separate_sci_wht_err'))
    return out


def discover_jwst() -> list[TileRecord]:
    """JWST NIRCam: 4 bands × tiles A1-A10 + B1-B10.

    Known F115W corruptions in COSMOS_v0.8:
      - A10: multi-extension file absent.  scidir/ has full sci+err+wht.
      - A2/B4/B6: multi-extension files are TRUNCATED (actual byte length
        less than header advertises; reading the WHT extension hits EOF).
        scidir/ has the v0.8 sci file only; we synthesize err=1/wht=1
        from sci-finiteness in the reader.
    """
    F115W_BROKEN = {'A2', 'B4', 'B6', 'A10'}

    out = []
    pat = re.compile(r'mosaic_nircam_(f\d+w)_COSMOS-Web_30mas_([AB]\d+)_v1\.0_i2d\.fits$')
    for f in sorted(JWST_DIR.glob('mosaic_nircam_*_i2d.fits')):
        m = pat.search(f.name)
        if not m:
            continue
        flt = m.group(1).upper()
        tile = m.group(2)
        # Skip F115W tiles known to be corrupt — scidir fallback below
        if flt == 'F115W' and tile in F115W_BROKEN:
            continue
        out.append(TileRecord('JWST', flt, tile, str(f),
                              file_format='multi_extension'))

    # F115W A10: scidir has full sci+err+wht
    a10_sci = JWST_SCIDIR / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits'
    a10_err = JWST_SCIDIR / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v1.0_err.fits'
    a10_wht = JWST_SCIDIR / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v1.0_wht.fits'
    if a10_sci.exists() and a10_err.exists() and a10_wht.exists():
        out.append(TileRecord('JWST', 'F115W', 'A10', str(a10_sci),
                              file_format='separate_sci_wht_err'))
    else:
        print('  WARN: F115W A10 scidir incomplete; tile will be skipped')

    # F115W A2 / B4 / B6: scidir has sci only — synthesize err/wht via single_hdu reader
    for tile in ('A2', 'B4', 'B6'):
        sci = JWST_SCIDIR / f'mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits'
        if sci.exists():
            out.append(TileRecord('JWST', 'F115W', tile, str(sci),
                                  file_format='single_hdu'))
        else:
            print(f'  WARN: F115W {tile} truncated AND no scidir backup; tile skipped')
    return out


def discover_euclid() -> list[TileRecord]:
    """Euclid MER: 4 bands × ~60 tiles, single-HDU mosaics."""
    out = []
    pat = re.compile(r'EUC_MER_BGSUB-MOSAIC-([A-Z\-]+)_TILE(\d+)-')
    band_label = {'VIS': 'VIS', 'NIR-Y': 'NIR-Y', 'NIR-J': 'NIR-J', 'NIR-H': 'NIR-H'}
    for f in sorted(EUCLID_DIR.glob('EUC_MER_BGSUB-MOSAIC-*.fits')):
        m = pat.search(f.name)
        if not m:
            continue
        band = m.group(1)
        if band not in band_label:
            continue
        tile_id = f'TILE{m.group(2)}'
        out.append(TileRecord('Euclid', band_label[band], tile_id, str(f),
                              file_format='single_hdu'))
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-format readers — return (sci, err, wht, wcs, header)
# ──────────────────────────────────────────────────────────────────────
def _native(arr: np.ndarray) -> np.ndarray:
    """Force big-endian FITS arrays to native byte order (for fast ops)."""
    if arr.dtype.byteorder not in ('=', '|'):
        return arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr


def read_layers(rec: TileRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, WCS]:
    """Open the FITS for this record and return (sci, err, wht, wcs)."""
    if rec.file_format == 'multi_extension':
        with fits.open(rec.path, memmap=True) as h:
            sci = _native(np.asarray(h['SCI'].data))
            err = _native(np.asarray(h['ERR'].data))
            wht = _native(np.asarray(h['WHT'].data))
            wcs = WCS(h['SCI'].header)
        return sci, err, wht, wcs

    if rec.file_format == 'separate_sci_wht_err':
        sci_path = Path(rec.path)
        # HST naming: ..._drz.fits, ..._err.fits, ..._wht.fits
        # JWST A10:   ..._A10_v0_8_sci.fits, ..._A10_v1.0_err.fits, _wht.fits
        if sci_path.name.endswith('_drz.fits'):
            err_path = sci_path.with_name(sci_path.name.replace('_drz.fits', '_err.fits'))
            wht_path = sci_path.with_name(sci_path.name.replace('_drz.fits', '_wht.fits'))
        elif '_sci.fits' in sci_path.name:
            # JWST scidir: v0_8_sci → v1.0_err / v1.0_wht
            stem = sci_path.name.replace('_v0_8_sci.fits', '')
            err_path = sci_path.with_name(f'{stem}_v1.0_err.fits')
            wht_path = sci_path.with_name(f'{stem}_v1.0_wht.fits')
        else:
            raise ValueError(f'cannot derive err/wht paths for {sci_path}')
        with fits.open(sci_path, memmap=True) as h:
            sci = _native(np.asarray(h[0].data))
            wcs = WCS(h[0].header)
        with fits.open(err_path, memmap=True) as h:
            err = _native(np.asarray(h[0].data))
        with fits.open(wht_path, memmap=True) as h:
            wht = _native(np.asarray(h[0].data))
        return sci, err, wht, wcs

    if rec.file_format == 'single_hdu':
        # Euclid MER mosaics are background-subtracted SCI in HDU 0.
        # The MER releases do not ship per-pixel err/wht for these mosaics;
        # we treat err=1, wht=1 wherever sci is finite (this just turns the
        # mask into `sci ≠ NaN`, which is the coverage criterion the MER
        # team itself uses for these products).
        with fits.open(rec.path, memmap=True) as h:
            sci = _native(np.asarray(h[0].data))
            wcs = WCS(h[0].header)
        err = np.where(np.isfinite(sci), 1.0, np.nan).astype(np.float32)
        wht = np.where(np.isfinite(sci), 1.0, 0.0).astype(np.float32)
        return sci, err, wht, wcs

    raise ValueError(f'unknown file_format: {rec.file_format}')


# ──────────────────────────────────────────────────────────────────────
# Build per-tile mask
# ──────────────────────────────────────────────────────────────────────
def build_coverage_mask(rec: TileRecord) -> np.ndarray:
    """Return uint8 (0/1) mask: pixel observed AND ≥10 px from any edge."""
    sci, err, wht, wcs = read_layers(rec)

    rec.naxis1 = int(wcs.array_shape[1]) if wcs.array_shape else sci.shape[1]
    rec.naxis2 = int(wcs.array_shape[0]) if wcs.array_shape else sci.shape[0]
    rec.n_pix_total = int(sci.size)

    # pixel scale (mas) from WCS — pixel_scale_matrix is in deg/px
    s = np.abs(wcs.pixel_scale_matrix)
    pixscale = float(np.sqrt(s[0,0]**2 + s[1,0]**2) * 3600 * 1000)
    rec.pixscale_mas = pixscale

    # Edge criterion (Q1.1)
    ok = np.isfinite(sci) & (wht > 0) & np.isfinite(err) & (err > 0)
    rec.n_pix_observed = int(ok.sum())

    # Erosion (Q1.2): shrink coverage by EROSION_NATIVE_PX pixels from any edge
    eroded = binary_erosion(ok, iterations=EROSION_NATIVE_PX, border_value=0)
    rec.n_pix_after_erosion = int(eroded.sum())

    # Footprint corners on sky (for the lookup table)
    ny, nx = sci.shape
    corner_px = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]])
    sky = wcs.pixel_to_world_values(corner_px[:, 0], corner_px[:, 1])
    ra_c, dec_c = sky[0], sky[1]
    rec.corners_ra  = [float(x) for x in ra_c]
    rec.corners_dec = [float(x) for x in dec_c]
    rec.ra_min  = float(np.min(ra_c))
    rec.ra_max  = float(np.max(ra_c))
    rec.dec_min = float(np.min(dec_c))
    rec.dec_max = float(np.max(dec_c))

    return eroded.astype(np.uint8), wcs


def write_mask(mask: np.ndarray, wcs: WCS, dest: Path,
               rec: TileRecord | None = None):
    """Write the binary mask as a gzipped FITS with the source WCS.

    When `rec` is provided we also embed the pre/post-erosion pixel
    counts in the header so a later --reuse pass can rehydrate the
    lookup table's n_pix_observed / n_pix_after_erosion / n_pix_total
    columns without reopening the (much larger) source mosaic.
    """
    hdu = fits.PrimaryHDU(data=mask, header=wcs.to_header())
    hdu.header['BUNIT'] = 'coverage'
    if rec is not None:
        hdu.header['NPXTOT']  = (int(rec.n_pix_total),
                                 'image pixel count')
        hdu.header['NPXOBS']  = (int(rec.n_pix_observed),
                                 'pixels passing sci/wht/err criterion')
        hdu.header['NPXEROD'] = (int(rec.n_pix_after_erosion),
                                 'pixels after 10-px erosion (= sum of mask)')
        hdu.header['PIXSMAS'] = (float(rec.pixscale_mas),
                                 'native pixel scale [mas]')
    hdu.header['COMMENT'] = '1 = pixel observed and >=10 px from edge'
    # astropy writes .fits.gz automatically if the extension matches
    hdu.writeto(dest, overwrite=True)


# ──────────────────────────────────────────────────────────────────────
# Global 1″ QC mask
# ──────────────────────────────────────────────────────────────────────
def make_global_wcs() -> WCS:
    """A synthetic TAN WCS centred on COSMOS, 1″ pixels, ±0.5°."""
    n = int(2 * GLOBAL_WCS_SIZE_DEG * 3600 / GLOBAL_PIXEL_SCALE_ARCSEC) + 1
    w = WCS(naxis=2)
    w.wcs.crpix = [(n+1) / 2, (n+1) / 2]
    w.wcs.crval = list(GLOBAL_WCS_CENTER)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cdelt = [-GLOBAL_PIXEL_SCALE_ARCSEC / 3600,
                    GLOBAL_PIXEL_SCALE_ARCSEC / 3600]
    w.array_shape = (n, n)
    return w


def reproject_to_global(mask_native: np.ndarray, wcs_native: WCS,
                         global_wcs: WCS) -> np.ndarray:
    """Reproject a per-tile mask onto the global QC grid (nearest-neighbour)."""
    out, _ = reproject_interp((mask_native.astype(np.float32), wcs_native),
                              global_wcs, shape_out=global_wcs.array_shape,
                              order='nearest-neighbor')
    return (np.nan_to_num(out, nan=0) > 0.5).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--missions', nargs='+',
                   default=['HST', 'JWST', 'Euclid'],
                   choices=['HST', 'JWST', 'Euclid'],
                   help='Subset of missions to process (default all).')
    p.add_argument('--skip-global', action='store_true',
                   help='Skip the global 1" QC mask (per-tile masks only).')
    p.add_argument('--dry-run', action='store_true',
                   help='Discover and print records; do not read FITS data.')
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover all tile records
    print('Discovering tiles ...')
    records: list[TileRecord] = []
    if 'HST'    in args.missions: records += discover_hst()
    if 'JWST'   in args.missions: records += discover_jwst()
    if 'Euclid' in args.missions: records += discover_euclid()

    # Summary by mission/filter
    summary = pd.DataFrame([(r.mission, r.filter) for r in records],
                           columns=['mission', 'filter'])
    print(f'\n  total tiles discovered: {len(records):,}')
    print(summary.groupby(['mission', 'filter']).size().to_string())
    print()

    if args.dry_run:
        print('[dry-run] would now read FITS and build masks. exiting.')
        return

    # 2. Per-tile mask build
    print('Building per-tile coverage masks ...')
    for i, rec in enumerate(records, 1):
        # Skip if mask already exists (idempotent re-runs)
        dest = COV_DIR / f'{rec.mission}_{rec.filter}_{rec.tile_id}.fits.gz'
        rec.coverage_mask_path = str(dest)
        if dest.exists():
            # Already built; rehydrate stats from the file for the lookup table
            try:
                with fits.open(dest, memmap=True) as h:
                    mask = np.asarray(h[0].data)
                    wcs  = WCS(h[0].header)
                    hdr  = h[0].header
                rec.naxis1, rec.naxis2 = mask.shape[1], mask.shape[0]
                # Prefer header-stored counts (written by build path); fall
                # back to mask.sum() for tiles built before the telemetry fix.
                rec.n_pix_total         = int(hdr.get('NPXTOT',  mask.size))
                rec.n_pix_after_erosion = int(hdr.get('NPXEROD', mask.sum()))
                rec.n_pix_observed      = int(hdr.get('NPXOBS',
                                                       rec.n_pix_after_erosion))
                # bounding box + corners
                ny, nx = mask.shape
                corner_px = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]])
                ra_c, dec_c = wcs.pixel_to_world_values(corner_px[:, 0], corner_px[:, 1])
                rec.corners_ra  = [float(x) for x in ra_c]
                rec.corners_dec = [float(x) for x in dec_c]
                rec.ra_min, rec.ra_max  = float(np.min(ra_c)), float(np.max(ra_c))
                rec.dec_min, rec.dec_max = float(np.min(dec_c)), float(np.max(dec_c))
                rec.pixscale_mas = float(hdr.get('PIXSMAS',
                    np.sqrt(np.abs(wcs.pixel_scale_matrix)[0,0]**2 +
                            np.abs(wcs.pixel_scale_matrix)[1,0]**2) * 3600e3))
                print(f'  [{i:3d}/{len(records)}] reuse {dest.name}  '
                      f'{rec.n_pix_after_erosion/1e6:.1f}M px covered')
                continue
            except Exception as e:
                print(f'  [{i:3d}/{len(records)}] could not reuse {dest.name}: {e}; rebuilding')

        try:
            t_tile = time.time()
            mask, wcs = build_coverage_mask(rec)
            write_mask(mask, wcs, dest, rec)
            print(f'  [{i:3d}/{len(records)}] {rec.mission:6s} {rec.filter:6s} '
                  f'{rec.tile_id:8s}  '
                  f'{rec.naxis1}×{rec.naxis2} @ {rec.pixscale_mas:5.1f} mas  '
                  f'obs={rec.n_pix_observed/1e6:6.1f}M  '
                  f'eroded={rec.n_pix_after_erosion/1e6:6.1f}M  '
                  f'({time.time()-t_tile:4.1f}s)')
        except Exception as e:
            print(f'  [{i:3d}/{len(records)}] FAIL {rec.mission} {rec.filter} '
                  f'{rec.tile_id}: {e!r}')

    # 3. Write the tile lookup table
    df = pd.DataFrame([r.as_row() for r in records])
    lookup_path = OUT_DIR / 'tile_lookup.parquet'
    df.to_parquet(lookup_path, index=False)
    df.to_csv(OUT_DIR / 'tile_lookup.csv', index=False)
    print(f'\n[save] {lookup_path.name}: {len(df):,} rows × {len(df.columns)} cols')

    # 4. Global 4-way QC mask
    if not args.skip_global:
        print('\nBuilding global 1" 4-way common-area mask ...')
        gwcs = make_global_wcs()
        gshape = gwcs.array_shape
        print(f'  global grid: {gshape[1]}×{gshape[0]} px at {GLOBAL_PIXEL_SCALE_ARCSEC}"')

        # Per-mission union: for each mission, OR all per-tile masks together
        # (after reprojection to the global grid).
        groups = df.groupby(['mission', 'filter'])
        mission_filter_global: dict[tuple[str,str], np.ndarray] = {}
        for (mission, flt), gdf in groups:
            print(f'  merging {mission} {flt} ({len(gdf)} tiles) ...', flush=True)
            agg = np.zeros(gshape, dtype=np.uint8)
            for _, row in gdf.iterrows():
                p = Path(row['coverage_mask_path'])
                if not p.exists():
                    continue
                try:
                    with fits.open(p, memmap=True) as h:
                        m = np.asarray(h[0].data)
                        w = WCS(h[0].header)
                    rp = reproject_to_global(m, w, gwcs)
                    agg = np.maximum(agg, rp)
                except Exception as e:
                    print(f'    warn: skip {p.name}: {e}')
            mission_filter_global[(mission, flt)] = agg
            print(f'    -> {agg.sum()/1e6:.2f} Mpx covered globally')

        # 4-way intersection:
        #   HST F814W ∩ (all JWST bands union'd OR intersect'd?) ∩ VIS ∩ NISP
        # The user's plan uses INTERSECTION across the four mission groups.
        # Within JWST we use the INTERSECTION of all 4 bands (a star must
        # be detected in all NIRCam bands for 4-way membership).  Same
        # logic for NISP across Y/J/H.
        def _intersect_all(masks):
            out = None
            for m in masks:
                out = m.copy() if out is None else np.minimum(out, m)
            return out if out is not None else np.zeros(gshape, dtype=np.uint8)

        hst_mask  = mission_filter_global.get(('HST', 'F814W'))
        jwst_mask = _intersect_all(
            [mission_filter_global[k] for k in mission_filter_global
             if k[0] == 'JWST'])
        vis_mask  = mission_filter_global.get(('Euclid', 'VIS'))
        nisp_mask = _intersect_all(
            [mission_filter_global[k] for k in mission_filter_global
             if k[0] == 'Euclid' and k[1].startswith('NIR-')])

        # Build the 4-way intersection only over missions that produced a mask
        components = [m for m in (hst_mask, jwst_mask, vis_mask, nisp_mask)
                      if m is not None and m.size]
        if not components:
            print('  no mission masks available; skipping 4-way intersection')
        else:
            four_way = _intersect_all(components)
            global_path = OUT_DIR / 'common_area_4way_1arcsec.fits.gz'
            hdu = fits.PrimaryHDU(data=four_way, header=gwcs.to_header())
            hdu.header['BUNIT'] = 'coverage'
            hdu.header['COMMENT'] = ('4-way HST INTERSECT JWST INTERSECT EuclidVIS'
                                     ' INTERSECT EuclidNISP common-area mask on a'
                                     ' synthetic 1 arcsec COSMOS WCS')
            hdu.writeto(global_path, overwrite=True)
            area_arcsec2 = float(four_way.sum())  # 1 pixel = 1 arcsec²
            area_deg2 = area_arcsec2 / 3600**2
            print(f'\n[save] {global_path.name}: '
                  f'{four_way.sum()/1e6:.2f} Mpx ≈ {area_deg2:.4f} deg² 4-way common area')

    print(f'\nWall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
