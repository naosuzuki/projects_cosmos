"""Build a per-survey tile-lookup table.

For each survey (HST ACS, JWST NIRCam x4 bands, Euclid VIS + NIR-Y/J/H):
  - Scan the data directory for the tile FITS files
  - Read the WCS from the SCI HDU header only
  - Compute the tile's sky footprint (RA/Dec corners + center)
  - Save a row per tile: survey, band, tile_id, path, ra_min, ra_max,
    dec_min, dec_max, crval1, crval2, crpix1, crpix2, cd11, cd12, cd21,
    cd22, naxis1, naxis2

Output: /Users/suzuki/github/projects_cosmos/csvfiles/tile_lookup.fits

Usage:
    python build_tile_lookup.py

The lookup table is used by cutout_by_tile.py to assign sources to tiles
without re-opening every FITS file.  Reading just the header from a .gz
is fast (a few KB at the start of the file).
"""
import warnings
warnings.filterwarnings("ignore")

import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# ----------------------------------------------------------------------
HST_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
JWST_DIR = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8")
EU_DIR   = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")
OUT_FITS = Path("/Users/suzuki/github/projects_cosmos/csvfiles/tile_lookup.fits")

# Survey -> list of (band, glob pattern, tile-id regex)
SURVEYS = {
    "HST": [
        ("F814W", "acs_I_030mas_*_sci.fits.gz", r"acs_I_030mas_(\d+)_sci\.fits"),
    ],
    "JWST": [
        ("F115W", "mosaic_nircam_f115w_*_v1.0_i2d.fits*", r"30mas_([A-Z]\d+)_v1\.0"),
        ("F150W", "mosaic_nircam_f150w_*_v1.0_i2d.fits*", r"30mas_([A-Z]\d+)_v1\.0"),
        ("F277W", "mosaic_nircam_f277w_*_v1.0_i2d.fits*", r"30mas_([A-Z]\d+)_v1\.0"),
        ("F444W", "mosaic_nircam_f444w_*_v1.0_i2d.fits*", r"30mas_([A-Z]\d+)_v1\.0"),
    ],
    "Euclid": [
        ("VIS",   "EUC_MER_BGSUB-MOSAIC-VIS_TILE*.fits",  r"TILE(\d+)-"),
        ("Y",     "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE*.fits", r"TILE(\d+)-"),
        ("J",     "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE*.fits", r"TILE(\d+)-"),
        ("H",     "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE*.fits", r"TILE(\d+)-"),
    ],
}

# Where to look for files per survey
SURVEY_DIRS = {"HST": HST_DIR, "JWST": JWST_DIR, "Euclid": EU_DIR}


def read_tile_meta(path):
    """Return (wcs, header) for the SCI (or PRIMARY) HDU of `path`.
    Reads only the header, not the data."""
    with fits.open(path, memmap=True) as h:
        sci_names = [hdu.name for hdu in h]
        if "SCI" in sci_names:
            hdr = h["SCI"].header
        else:
            hdr = h[0].header
    return WCS(hdr), hdr


def tile_corners_radec(wcs, naxis1, naxis2):
    """Return (ra_min, ra_max, dec_min, dec_max) of the tile footprint."""
    # 4 corners: (0,0), (n1,0), (0,n2), (n1,n2)
    xs = np.array([0, naxis1 - 1, 0, naxis1 - 1])
    ys = np.array([0, 0, naxis2 - 1, naxis2 - 1])
    ras, decs = wcs.all_pix2world(xs, ys, 0)
    return float(np.min(ras)), float(np.max(ras)), float(np.min(decs)), float(np.max(decs))


def main():
    rows = []
    t_start = time.time()
    for survey, band_specs in SURVEYS.items():
        ddir = SURVEY_DIRS[survey]
        for band, pattern, tile_re_str in band_specs:
            tile_re = re.compile(tile_re_str)
            paths = sorted(ddir.glob(pattern))
            # De-dup macOS resource-fork files
            paths = [p for p in paths if not p.name.startswith("._")]
            # Prefer uncompressed over .gz when both exist (memmap-friendly)
            chosen = {}
            for p in paths:
                m = tile_re.search(p.name)
                if not m:
                    continue
                tile_id = m.group(1)
                # If we already have a non-gz path for this tile, skip the .gz
                if tile_id in chosen and not chosen[tile_id].name.endswith(".gz"):
                    continue
                # If we now see a non-gz and current is gz, replace
                if tile_id in chosen and p.name.endswith(".gz") and not chosen[tile_id].name.endswith(".gz"):
                    continue
                chosen[tile_id] = p
            print(f"\n{survey} {band}: {len(chosen)} unique tiles", flush=True)
            for tile_id, path in sorted(chosen.items()):
                try:
                    wcs, hdr = read_tile_meta(path)
                    n1 = int(hdr["NAXIS1"])
                    n2 = int(hdr["NAXIS2"])
                    ra_min, ra_max, dec_min, dec_max = tile_corners_radec(wcs, n1, n2)
                    cr1, cr2 = wcs.wcs.crval
                    cp1, cp2 = wcs.wcs.crpix
                    cd = wcs.wcs.cd if wcs.wcs.has_cd() else wcs.pixel_scale_matrix
                    rows.append(dict(
                        survey=survey, band=band, tile_id=tile_id,
                        path=str(path),
                        ra_min=ra_min, ra_max=ra_max,
                        dec_min=dec_min, dec_max=dec_max,
                        ra_center=(ra_min + ra_max) / 2.0,
                        dec_center=(dec_min + dec_max) / 2.0,
                        crval1=float(cr1), crval2=float(cr2),
                        crpix1=float(cp1), crpix2=float(cp2),
                        cd11=float(cd[0, 0]), cd12=float(cd[0, 1]),
                        cd21=float(cd[1, 0]), cd22=float(cd[1, 1]),
                        naxis1=n1, naxis2=n2,
                        is_gz=path.name.endswith(".gz"),
                    ))
                    print(f"  {tile_id:>10s}: RA[{ra_min:.4f}-{ra_max:.4f}] "
                          f"Dec[{dec_min:.4f}-{dec_max:.4f}]  "
                          f"{n1}x{n2}  gz={path.name.endswith('.gz')}", flush=True)
                except Exception as e:
                    print(f"  ERR {path.name}: {e}", flush=True)

    t = Table(rows)
    OUT_FITS.parent.mkdir(parents=True, exist_ok=True)
    t.write(OUT_FITS, overwrite=True)
    print(f"\nWrote {OUT_FITS}  ({len(t)} tiles)", flush=True)
    print(f"Time: {time.time() - t_start:.1f} sec", flush=True)


if __name__ == "__main__":
    main()
