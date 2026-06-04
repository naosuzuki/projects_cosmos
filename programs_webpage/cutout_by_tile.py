"""Cutout pipeline driven by the tile-lookup table.

Process sources GROUPED BY TILE: open each tile FITS exactly once, cut
all sources that fall on it, free memory before the next tile.  This is
the right pattern for thousands of sources.

Uses /Users/suzuki/github/projects_cosmos/csvfiles/tile_lookup.fits
(built by build_tile_lookup.py).

Usage:
    python cutout_by_tile.py --candidates <fits>  --out <png_dir>  [--north-up]  [--no-labels]
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.table import Table

# ----------------------------------------------------------------------
LOOKUP = Path("/Users/suzuki/github/projects_cosmos/csvfiles/tile_lookup.fits")
CUTOUT_SIZE = 6.0 * u.arcsec


def load_lookup():
    """Return dict {survey: {band: list-of-tile-rows}}."""
    t = Table.read(LOOKUP)
    out = {}
    for row in t:
        out.setdefault(row["survey"], {}).setdefault(row["band"], []).append(dict(row))
    return out


def assign_tile(ra, dec, tile_rows):
    """Return the tile_row whose footprint contains (ra, dec), or None."""
    # Fast rectangular pre-filter on RA/Dec corners
    for row in tile_rows:
        if (row["ra_min"] <= ra <= row["ra_max"] and
            row["dec_min"] <= dec <= row["dec_max"]):
            # Confirm with full WCS check (corners may bound a rotated tile loosely)
            wcs = _wcs_from_row(row)
            try:
                x, y = wcs.all_world2pix(ra, dec, 0)
            except Exception:
                continue
            if 0 <= x < row["naxis1"] and 0 <= y < row["naxis2"]:
                return row
    return None


def _wcs_from_row(row):
    """Re-create a WCS object from the lookup row's CD-matrix fields."""
    w = WCS(naxis=2)
    w.wcs.crval = [row["crval1"], row["crval2"]]
    w.wcs.crpix = [row["crpix1"], row["crpix2"]]
    w.wcs.cd    = [[row["cd11"], row["cd12"]],
                   [row["cd21"], row["cd22"]]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def group_sources_by_tile(sources, tile_rows):
    """Return dict {tile_path: [source rows]} for sources that hit a tile."""
    grouped = defaultdict(list)
    n_unassigned = 0
    for src in sources:
        tile = assign_tile(src["ra"], src["dec"], tile_rows)
        if tile is None:
            n_unassigned += 1
            continue
        grouped[tile["path"]].append(src)
    return grouped, n_unassigned


def open_tile(path):
    """Open SCI HDU; works for both i2d-style (with SCI ext) and PRIMARY-data files."""
    h = fits.open(path, memmap=True)
    sci_names = [hdu.name for hdu in h]
    sci_hdu = h["SCI"] if "SCI" in sci_names else h[0]
    return h, sci_hdu, WCS(sci_hdu.header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="FITS with id, ra, dec columns")
    ap.add_argument("--survey", default="JWST", choices=("HST", "JWST", "Euclid"))
    ap.add_argument("--band",   default="F150W", help="band string matching the lookup table")
    args = ap.parse_args()

    cat = Table.read(args.candidates)
    src_list = [dict(id=int(r["id"]), ra=float(r["ra"]), dec=float(r["dec"]))
                for r in cat]
    print(f"Sources: {len(src_list)}", flush=True)

    lookup = load_lookup()
    tile_rows = lookup[args.survey][args.band]
    print(f"{args.survey} {args.band}: {len(tile_rows)} tiles in lookup", flush=True)

    t0 = time.time()
    grouped, n_unassigned = group_sources_by_tile(src_list, tile_rows)
    print(f"Grouping: {time.time() - t0:.1f}s  "
          f"({len(grouped)} unique tiles, {n_unassigned} unassigned)", flush=True)
    for tile_path, srcs in sorted(grouped.items()):
        name = Path(tile_path).name
        print(f"  {name[:70]:70s}  {len(srcs):4d} sources", flush=True)

    print(f"\nReady to process by tile.  This is just a demonstration of the "
          f"grouping; plug in the actual cutout + render call here when needed.",
          flush=True)


if __name__ == "__main__":
    main()
