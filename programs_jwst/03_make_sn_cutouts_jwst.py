"""JWST NIRCam 2x RGB cutouts for SN candidates.

Recipe = create_colorimg_jwst() from 02_readcatalog_jwst.py:
  - 6 arcsec cutout (200x200 px at 30 mas)
  - sqrt stretch, min=0.0001, max from central 10x10, max*0.85
  - 2 RGB images per source:
       jwst1: B=F115W G=F150W R=F277W
       jwst2: B=F150W G=F277W R=F444W

Usage:
  python 03_make_sn_cutouts_jwst.py
  python 03_make_sn_cutouts_jwst.py --list jwst_only --max 20
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
JW_DIR  = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir")
PNG_DIR = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
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


# Filename pattern: mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits
# Tile id = token between 30mas and v0
TILE_RE = re.compile(r"_30mas_([A-Z]\d+)_v")


def parse_tile(fname):
    m = TILE_RE.search(fname)
    return m.group(1) if m else None


def build_band_tile_index(band):
    """{tile: (path, wcs, naxis1, naxis2)}"""
    tiles = {}
    pattern = f"mosaic_nircam_{band}_COSMOS-Web_30mas_*_sci.fits"
    for f in sorted(JW_DIR.glob(pattern)):
        tile = parse_tile(f.name)
        if tile is None:
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


def render_rgb(b_data, g_data, r_data, png_path, label_lines):
    """create_colorimg_jwst() recipe."""
    minimum = 1e-4
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
        maximum *= 0.85

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


def process_list(list_name, band_tiles, cache, n_max=0):
    """Process one candidate list, reusing tile indices + data cache."""
    cfg = CANDIDATE_LISTS[list_name]
    print(f"\n=== {list_name} ===", flush=True)
    cat = Table.read(cfg["fits"])
    if "host_has" in cat.colnames:
        cat = cat[cat["host_has"] == 1]
    if n_max:
        cat = cat[:n_max]
    print(f"  {len(cat)} candidates to process", flush=True)

    csv_path = CSV_DIR / f"sn_{list_name}_jwst_index.csv"
    fh = open(csv_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["seq", "id", "ra", "dec", "mag", "tile",
                "jwst1_png", "jwst2_png"])

    def get_data(path):
        if path not in cache:
            cache[path] = fits.getdata(path).astype(np.float32)
        return cache[path]

    n_ok = 0; n_miss = 0
    for seq, row in enumerate(cat, 1):
        cid = int(row[cfg["id_col"]])
        ra  = float(row[cfg["ra_col"]])
        dec = float(row[cfg["dec_col"]])
        mag = float(row[cfg["mag_col"]]) if np.isfinite(row[cfg["mag_col"]]) else np.nan

        tile = find_tile(ra, dec, band_tiles["f150w"])
        if tile is None:
            n_miss += 1
            w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                        f"{mag:.2f}", "", "", ""])
            if seq % 50 == 0 or seq == len(cat):
                print(f"  [{seq:04d}/{len(cat)}] ok={n_ok} miss={n_miss}", flush=True)
            continue

        position = SkyCoord(ra, dec, unit="deg", frame="icrs")

        jwst1_name = jwst2_name = ""
        try:
            cuts = {}
            for b in ("f115w", "f150w", "f277w", "f444w"):
                if tile not in band_tiles[b]:
                    raise RuntimeError(f"tile {tile} not in {b}")
                d = get_data(band_tiles[b][tile]["path"])
                cuts[b] = Cutout2D(d, position, size=CUTOUT_SIZE,
                                   wcs=band_tiles[b][tile]["wcs"]).data

            jwst1_name = f"sn_{list_name}_{seq:04d}_jwst1.png"
            render_rgb(cuts["f115w"], cuts["f150w"], cuts["f277w"],
                       PNG_DIR / jwst1_name,
                       [f"ID={cid}", "F115/F150/F277"])

            jwst2_name = f"sn_{list_name}_{seq:04d}_jwst2.png"
            render_rgb(cuts["f150w"], cuts["f277w"], cuts["f444w"],
                       PNG_DIR / jwst2_name,
                       [f"ID={cid}", "F150/F277/F444"])
            n_ok += 1
        except Exception as e:
            print(f"  [{seq:04d}] JWST cutout failed: {e}", flush=True)

        w.writerow([seq, cid, f"{ra:.6f}", f"{dec:.6f}",
                    f"{mag:.2f}", tile, jwst1_name, jwst2_name])
        if seq % 50 == 0 or seq == len(cat):
            print(f"  [{seq:04d}/{len(cat)}] ok={n_ok} miss={n_miss}", flush=True)

    fh.close()
    print(f"  Done {list_name}: {n_ok} pairs, {n_miss} outside footprint -> {csv_path}", flush=True)


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

    print("Indexing JWST tiles ...", flush=True)
    band_tiles = {b: build_band_tile_index(b) for b in
                  ("f115w", "f150w", "f277w", "f444w")}
    for b, t in band_tiles.items():
        print(f"  {b}: {len(t)} tiles", flush=True)

    cache = {}  # path -> array, shared across all lists
    targets = list(CANDIDATE_LISTS.keys()) if args.all else [args.list]
    for ln in targets:
        process_list(ln, band_tiles, cache, n_max=args.max)
    print(f"\nAll done. Cached arrays: {len(cache)}", flush=True)


if __name__ == "__main__":
    main()
