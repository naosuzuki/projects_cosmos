"""HST/ACS F814W grayscale cutouts for SN candidates.

Recipe = create_blackwhite_hst() from 02_readcatalog_hst.py:
  - 6 arcsec cutout (200x200 px at 30 mas)
  - sqrt stretch, minimum=0.0001, max from central 10x10, max*0.85
  - cmap='gray'

Usage:
  python 03_make_sn_cutouts_hst.py
  python 03_make_sn_cutouts_hst.py --list jwst_only --max 20
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
warnings.filterwarnings("ignore", message=".*UnitsWarning.*")

# ============================================================
HST_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
PNG_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles")
CAND_DIR = Path("/Users/suzuki/github/projects_euclid/fits_tbl")

CANDIDATE_LISTS = {
    "euclid_only": dict(
        fits=CAND_DIR / "sn_candidates_euclid_only_vs_jwst.fits",
        ra_col="right_ascension", dec_col="declination",
        id_col="object_id", mag_col="mag_vis_psf"),
    "jwst_only":   dict(
        fits=CAND_DIR / "sn_candidates_jwst_only_vs_euclid.fits",
        ra_col="ra", dec_col="dec",
        id_col="id", mag_col="mag_auto_f150w"),
    "acs_only":    dict(
        fits=CAND_DIR / "sn_candidates_acs_only.fits",
        ra_col="ra", dec_col="dec",
        id_col="number", mag_col="mag_auto"),
    "known34":     dict(
        fits=CAND_DIR / "known34_sn.fits",
        ra_col="ra", dec_col="dec",
        id_col="id", mag_col="mag_auto_f150w"),
    "known34c":    dict(
        # SN-centered re-cutout test list.  ra,dec point at the SN position
        # (not the master-catalog host).  host_ra/host_dec audit columns.
        fits=CAND_DIR / "known34c_single.fits",
        ra_col="ra", dec_col="dec",
        id_col="id", mag_col="mag_auto_f150w"),
}

CUTOUT_SIZE = 6.0 * u.arcsec
# ============================================================


def build_tile_index(directory, pattern="acs_I_030mas_*_sci.fits.gz"):
    """Map tile_string -> (path, WCS, NAXIS1, NAXIS2)."""
    tiles = {}
    for f in sorted(directory.glob(pattern)):
        # tile id is 4th token in name, e.g. acs_I_030mas_013_sci.fits.gz -> 013
        toks = f.name.split("_")
        tile = toks[3]
        try:
            with fits.open(f, memmap=True) as h:
                hdr = h[0].header
                wcs = WCS(hdr)
                tiles[tile] = dict(path=str(f), wcs=wcs,
                                   naxis1=hdr["NAXIS1"], naxis2=hdr["NAXIS2"])
        except Exception as e:
            print(f"  WARN: could not open {f.name}: {e}")
    return tiles


def find_tile(ra, dec, tile_index):
    """Return tile_string for the first tile containing (ra, dec)."""
    for tile, info in tile_index.items():
        try:
            x, y = info["wcs"].all_world2pix(ra, dec, 0)
        except Exception:
            continue
        if 0 <= x < info["naxis1"] and 0 <= y < info["naxis2"]:
            return tile
    return None


def render_gray(data, png_path, label_lines, no_labels=False):
    """create_blackwhite_hst() recipe.  no_labels=True skips the upper-left
    baked-in text so a post-processor can stamp polished labels."""
    minimum = 0.0001
    bw = np.where(data > minimum, data, minimum)
    bw = np.sqrt(bw)
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if val < 1.0 else val * 0.85

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.imshow(bw, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
    ax.axis("off")
    if not no_labels:
        for k, txt in enumerate(label_lines):
            ax.text(0.03, 0.95 - 0.07*k, txt, color="white",
                    fontsize=11, transform=ax.transAxes,
                    verticalalignment="top",
                    family="serif", weight="bold")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


def process_list(list_name, tiles, loaded, n_max=0, no_labels=False):
    """Process one candidate list, reusing tile index + data cache.

    no_labels=True skips the matplotlib upper-left text (post-process can
    stamp polished labels)."""
    cfg = CANDIDATE_LISTS[list_name]
    print(f"\n=== {list_name} ===", flush=True)
    print(f"Reading candidates: {cfg['fits']}", flush=True)
    cat = Table.read(cfg["fits"])
    if "host_has" in cat.colnames:
        cat = cat[cat["host_has"] == 1]
    if n_max:
        cat = cat[:n_max]
    print(f"  {len(cat)} candidates to process", flush=True)

    csv_path = CSV_DIR / f"sn_{list_name}_hst_index.csv"
    fh = open(csv_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["seq", "id", "ra", "dec", "mag", "tile", "hstpng"])

    n_ok = 0; n_missing_tile = 0
    for seq, row in enumerate(cat, 1):
        cid = int(row[cfg["id_col"]])
        ra  = float(row[cfg["ra_col"]])
        dec = float(row[cfg["dec_col"]])
        mag = float(row[cfg["mag_col"]]) if np.isfinite(row[cfg["mag_col"]]) else np.nan

        tile = find_tile(ra, dec, tiles)
        if tile is None:
            n_missing_tile += 1
            w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                        f"{mag:.2f}", "", ""])
            if seq % 50 == 0 or seq == len(cat):
                print(f"  [{seq:04d}/{len(cat)}] ok={n_ok} miss={n_missing_tile}", flush=True)
            continue

        if tile not in loaded:
            print(f"  [{seq:04d}/{len(cat)}] loading tile {tile}", flush=True)
            loaded[tile] = fits.getdata(tiles[tile]["path"]).astype(np.float32)
        data = loaded[tile]
        wcs  = tiles[tile]["wcs"]

        try:
            position = SkyCoord(ra, dec, unit="deg", frame="icrs")
            cut = Cutout2D(data, position, size=CUTOUT_SIZE, wcs=wcs)
        except Exception as e:
            print(f"  [{seq:04d}] CUTOUT FAILED: {e}", flush=True)
            w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                        f"{mag:.2f}", tile, ""])
            continue

        png_name = f"sn_{list_name}_{seq:04d}_hst.png"
        png_path = PNG_DIR / png_name
        label = [f"ID={cid}", f"F814W={mag:.2f}" if np.isfinite(mag) else "F814W=?"]
        render_gray(cut.data, png_path, label, no_labels=no_labels)
        n_ok += 1
        w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                    f"{mag:.2f}", tile, png_name])
        if seq % 50 == 0 or seq == len(cat):
            print(f"  [{seq:04d}/{len(cat)}] ok={n_ok} miss={n_missing_tile}", flush=True)

    fh.close()
    print(f"  Done {list_name}: {n_ok} PNGs, {n_missing_tile} outside footprint -> {csv_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", choices=list(CANDIDATE_LISTS.keys()),
                    default="euclid_only", help="Which candidate list to process.")
    ap.add_argument("--all", action="store_true",
                    help="Process all 3 lists in one process (shared tile cache).")
    ap.add_argument("--max", type=int, default=0,
                    help="Cap candidates per list (0 = no cap; default 0).")
    ap.add_argument("--no-labels", action="store_true",
                    help="Skip the baked-in upper-left labels.")
    args = ap.parse_args()

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Indexing HST tiles in {HST_DIR} ...", flush=True)
    tiles = build_tile_index(HST_DIR)
    print(f"  {len(tiles)} HST tiles indexed", flush=True)

    loaded = {}  # tile -> array, shared across all lists
    targets = list(CANDIDATE_LISTS.keys()) if args.all else [args.list]
    for ln in targets:
        process_list(ln, tiles, loaded, n_max=args.max, no_labels=args.no_labels)
    print(f"\nAll done. Cached tiles: {len(loaded)}", flush=True)


if __name__ == "__main__":
    main()
