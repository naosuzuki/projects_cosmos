"""Euclid VIS (grayscale) + NISP Y/J/H (RGB color) cutouts for SN candidates.

VIS recipe   = create_blackwhite_hst() (grayscale, sqrt, central 10x10, max*0.85)
NISP recipe  = create_colorimg_jwst() (RGB B=Y G=J R=H, sqrt, central 10x10, max*1.2)

Usage:
  python 03_make_sn_cutouts_euclid.py
  python 03_make_sn_cutouts_euclid.py --list jwst_only --max 50
"""
import argparse
import csv
import re
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
from astropy.visualization import make_rgb, ManualInterval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
warnings.filterwarnings("ignore", message=".*UnitsWarning.*")

# ============================================================
EU_DIR  = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")
PNG_DIR = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")
CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles")
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
}

CUTOUT_SIZE = 6.0 * u.arcsec
# ============================================================


TILE_RE = re.compile(r"_TILE(\d+)-")


def parse_tile(fname):
    m = TILE_RE.search(fname)
    return m.group(1) if m else None


def build_band_tile_index(band_keyword):
    """{tile_index_str: (path, wcs, naxis1, naxis2)} for one band."""
    tiles = {}
    pattern = f"EUC_MER_BGSUB-MOSAIC-{band_keyword}_TILE*.fits"
    for f in sorted(EU_DIR.glob(pattern)):
        tile = parse_tile(f.name)
        if tile is None:
            continue
        # If duplicate (multiple versions of same tile), prefer the newest by mtime
        if tile in tiles and f.stat().st_mtime <= Path(tiles[tile]["path"]).stat().st_mtime:
            continue
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
    for tile, info in tile_index.items():
        try:
            x, y = info["wcs"].all_world2pix(ra, dec, 0)
        except Exception:
            continue
        if 0 <= x < info["naxis1"] and 0 <= y < info["naxis2"]:
            return tile
    return None


def render_gray(data, png_path, label_lines):
    """Same recipe as HST grayscale."""
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
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color="white",
                fontsize=11, transform=ax.transAxes,
                verticalalignment="top",
                family="serif", weight="bold")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


def render_rgb(b_data, g_data, r_data, png_path, label_lines):
    """create_colorimg_jwst() recipe (B=Y, G=J, R=H for Euclid NISP)."""
    minimum = 1e-5
    b = np.sqrt(np.where(b_data > minimum, b_data, minimum))
    g = np.sqrt(np.where(g_data > minimum, g_data, minimum))
    r = np.sqrt(np.where(r_data > minimum, r_data, minimum))

    maximum = 0.001
    for img in (b, g, r):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        val = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if val > maximum:
            maximum = val
    if maximum < 1.0:
        maximum = 1.0
    else:
        maximum *= 1.2

    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color="white",
                fontsize=11, transform=ax.transAxes,
                verticalalignment="top",
                family="serif", weight="bold")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


def process_list(list_name, tile_vis, tile_y, tile_j, tile_h, cache, n_max=0):
    """Process one candidate list, reusing tile indices + data cache."""
    cfg = CANDIDATE_LISTS[list_name]
    print(f"\n=== {list_name} ===", flush=True)
    cat = Table.read(cfg["fits"])
    if "host_has" in cat.colnames:
        cat = cat[cat["host_has"] == 1]
    if n_max:
        cat = cat[:n_max]
    print(f"  {len(cat)} candidates to process", flush=True)

    csv_path = CSV_DIR / f"sn_{list_name}_euclid_index.csv"
    fh = open(csv_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["seq", "id", "ra", "dec", "mag", "tile",
                "euclid_vis_png", "euclid_nisp_png"])

    def get_data(path):
        if path not in cache:
            cache[path] = fits.getdata(path).astype(np.float32)
        return cache[path]

    n_vis = n_nisp = n_miss = 0
    for seq, row in enumerate(cat, 1):
        cid = int(row[cfg["id_col"]])
        ra  = float(row[cfg["ra_col"]])
        dec = float(row[cfg["dec_col"]])
        mag = float(row[cfg["mag_col"]]) if np.isfinite(row[cfg["mag_col"]]) else np.nan

        tile = find_tile(ra, dec, tile_vis)
        if tile is None:
            n_miss += 1
            w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                        f"{mag:.2f}", "", "", ""])
            if seq % 50 == 0 or seq == len(cat):
                print(f"  [{seq:04d}/{len(cat)}] VIS={n_vis} NISP={n_nisp} miss={n_miss}", flush=True)
            continue

        position = SkyCoord(ra, dec, unit="deg", frame="icrs")

        vis_png_name = ""
        try:
            data_vis = get_data(tile_vis[tile]["path"])
            cut_vis = Cutout2D(data_vis, position, size=CUTOUT_SIZE,
                               wcs=tile_vis[tile]["wcs"])
            vis_png_name = f"sn_{list_name}_{seq:04d}_euclid_vis.png"
            render_gray(cut_vis.data, PNG_DIR / vis_png_name,
                        [f"ID={cid}", f"VIS={mag:.2f}"
                         if np.isfinite(mag) else "VIS=?"])
            n_vis += 1
        except Exception as e:
            print(f"  [{seq:04d}] VIS cutout failed: {e}", flush=True)

        nisp_png_name = ""
        try:
            if (tile in tile_y) and (tile in tile_j) and (tile in tile_h):
                cy = Cutout2D(get_data(tile_y[tile]["path"]),
                              position, size=CUTOUT_SIZE, wcs=tile_y[tile]["wcs"])
                cj = Cutout2D(get_data(tile_j[tile]["path"]),
                              position, size=CUTOUT_SIZE, wcs=tile_j[tile]["wcs"])
                ch = Cutout2D(get_data(tile_h[tile]["path"]),
                              position, size=CUTOUT_SIZE, wcs=tile_h[tile]["wcs"])
                nisp_png_name = f"sn_{list_name}_{seq:04d}_euclid_nisp.png"
                render_rgb(cy.data, cj.data, ch.data, PNG_DIR / nisp_png_name,
                           [f"ID={cid}", "Y/J/H"])
                n_nisp += 1
        except Exception as e:
            print(f"  [{seq:04d}] NISP cutout failed: {e}", flush=True)

        w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                    f"{mag:.2f}", tile, vis_png_name, nisp_png_name])
        if seq % 50 == 0 or seq == len(cat):
            print(f"  [{seq:04d}/{len(cat)}] VIS={n_vis} NISP={n_nisp} miss={n_miss}", flush=True)

    fh.close()
    print(f"  Done {list_name}: VIS={n_vis} NISP={n_nisp} miss={n_miss} -> {csv_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", choices=list(CANDIDATE_LISTS.keys()),
                    default="euclid_only")
    ap.add_argument("--all", action="store_true",
                    help="Process all 3 lists in one process (shared tile cache).")
    ap.add_argument("--max", type=int, default=0,
                    help="Cap candidates per list (0 = no cap; default 0).")
    args = ap.parse_args()

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Indexing Euclid tiles ...", flush=True)
    tile_vis = build_band_tile_index("VIS")
    tile_y   = build_band_tile_index("NIR-Y")
    tile_j   = build_band_tile_index("NIR-J")
    tile_h   = build_band_tile_index("NIR-H")
    print(f"  VIS:{len(tile_vis)}  Y:{len(tile_y)}  J:{len(tile_j)}  H:{len(tile_h)}", flush=True)

    cache = {}  # path -> array, shared across all lists
    targets = list(CANDIDATE_LISTS.keys()) if args.all else [args.list]
    for ln in targets:
        process_list(ln, tile_vis, tile_y, tile_j, tile_h, cache, n_max=args.max)
    print(f"\nAll done. Cached arrays: {len(cache)}", flush=True)


if __name__ == "__main__":
    main()
