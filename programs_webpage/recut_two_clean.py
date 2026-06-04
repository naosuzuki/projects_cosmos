"""Fast single-source re-cutouts, NO baked-in labels (clean full-frame).

Cuts the 5 panels for both test sources (ID=318858 centered on SN coords,
ID=320233 centered on its catalog/SN coords) and saves them with prefix
'sn_known34c_<seq>'.  The labels are stamped later by test_sn_crosshair_two.py.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.visualization import make_rgb, ManualInterval
from scipy.ndimage import rotate as ndi_rotate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Two sources to re-cut (centered on SN coords)
SOURCES = [
    dict(id=318858, seq=1, sn_ra=150.185329, sn_dec=1.842655),
    dict(id=320233, seq=2, sn_ra=150.146362, sn_dec=1.866412),
]
# All on the same tiles (HST 041, JWST A9, Euclid 101541377)

CUTOUT_SIZE = 6.0 * u.arcsec
ID_TAG = "known34c"

OUT_HST    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
OUT_EUCLID = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")
OUT_JWST   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")

HST_TILE_SCI = "/Volumes/exdisk1/data/HST/COSMOS_v2.0/acs_I_030mas_041_sci.fits.gz"
JWST_TPL     = "/Volumes/exdisk1/data/JWST/COSMOS_v0.8/mosaic_nircam_{b}_COSMOS-Web_30mas_A9_v1.0_i2d.fits.gz"
EU_DIR       = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")
EU_TILE      = "101541377"

import glob
EU_PATHS = {b: sorted(glob.glob(str(EU_DIR / f"EUC_MER_BGSUB-MOSAIC-{b}_TILE{EU_TILE}-*.fits")))[-1]
            for b in ("VIS", "NIR-Y", "NIR-J", "NIR-H")}


# ============================================================
# Rendering recipes -- IDENTICAL to production but no ax.text() calls.

def _save_axless(fig):
    return fig

def render_gray(data, png_path):
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
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_jwst(b_data, g_data, r_data, png_path):
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
    maximum = max(1.0, maximum * 0.85)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_nisp(b_data, g_data, r_data, png_path):
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
    maximum = max(1.0, maximum * 1.2)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


# ============================================================
def compute_roll_angle(wcs, ra, dec):
    """Return the angle (deg) by which the image +y axis is rotated EAST of
    celestial north at (ra, dec).  Image is North-up if angle is 0.
    To rotate the image to North-up, rotate by -angle about the center.
    """
    x0, y0 = wcs.all_world2pix(ra, dec, 0)
    xn, yn = wcs.all_world2pix(ra, dec + 1.0/3600.0, 0)
    return float(np.degrees(np.arctan2(float(xn - x0), float(yn - y0))))


def cut(path, pos, north_up=False):
    """Cut a 6'' window around `pos`.  If `north_up`, rotate the result so
    celestial north points up (compensates the tile roll angle).
    """
    with fits.open(path, memmap=True) as h:
        hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(hdu.header)
        cut2 = Cutout2D(hdu.data.astype(np.float32), pos, size=CUTOUT_SIZE, wcs=wcs)
    if north_up:
        angle = compute_roll_angle(wcs, pos.ra.deg, pos.dec.deg)
        # Rotate the data array by -angle about center to compensate
        # (positive angle in scipy = counter-clockwise = +y -> -x direction)
        data = ndi_rotate(cut2.data, -angle, reshape=False, order=3,
                          mode="constant", cval=0.0)
        # Replace data in-place; WCS is now invalid for this rotated array
        # but we don't use it downstream.
        cut2.data[:] = data
    return cut2


def main():
    for src in SOURCES:
        pos = SkyCoord(src["sn_ra"], src["sn_dec"], unit="deg", frame="icrs")
        seqn = f"{src['seq']:04d}"
        print(f"\n==== ID={src['id']}  seq={seqn} ====")
        print(f"  SN @ RA={src['sn_ra']:.6f}  Dec={src['sn_dec']:.6f}")

        # HST
        print("  HST F814W ...")
        cut_hst = cut(HST_TILE_SCI, pos)
        render_gray(cut_hst.data, OUT_HST / f"sn_{ID_TAG}_{seqn}_hst.png")

        # Euclid VIS
        print("  Euclid VIS ...")
        cut_vis = cut(EU_PATHS["VIS"], pos)
        render_gray(cut_vis.data, OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_vis.png")

        # Euclid NISP Y/J/H
        print("  Euclid NISP Y/J/H ...")
        cy_d = cut(EU_PATHS["NIR-Y"], pos).data
        cj_d = cut(EU_PATHS["NIR-J"], pos).data
        ch_d = cut(EU_PATHS["NIR-H"], pos).data
        render_rgb_nisp(cy_d, cj_d, ch_d,
                        OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_nisp.png")

        # JWST jwst1 (F115/F150/F277) -- rotate to N-up
        print("  JWST F115/F150/F277 (N-up rotated) ...")
        c115 = cut(JWST_TPL.format(b="f115w"), pos, north_up=True).data
        c150 = cut(JWST_TPL.format(b="f150w"), pos, north_up=True).data
        c277 = cut(JWST_TPL.format(b="f277w"), pos, north_up=True).data
        render_rgb_jwst(c115, c150, c277,
                        OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst1.png")

        # JWST jwst2 (F150/F277/F444) -- reuse rotated c150 + c277
        print("  JWST F150/F277/F444 (N-up rotated) ...")
        c444 = cut(JWST_TPL.format(b="f444w"), pos, north_up=True).data
        render_rgb_jwst(c150, c277, c444,
                        OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst2.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
