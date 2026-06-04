"""Fast 5-source re-cutout: 2 JWST SN + 3 HST SN.

For each source we know the exact tiles (from sn_known34_*_index.csv) so
we skip the full tile-indexing step the production scripts do.  Just open
the specific tiles, cut, render PNG (no labels, JWST N-up rotated).
"""
import warnings
warnings.filterwarnings("ignore")

import glob
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
SOURCES = [
    dict(id=318858, seq=1, sn_ra=150.185329, sn_dec=1.842655,
         hst="041", jwst="A9",  euclid="101541377"),
    dict(id=320233, seq=2, sn_ra=150.146362, sn_dec=1.866412,
         hst="041", jwst="A9",  euclid="101541377"),
    dict(id=130972, seq=3, sn_ra=150.279917, sn_dec=2.041285,
         hst="052", jwst="A4",  euclid="101542818"),
    dict(id=371996, seq=4, sn_ra=150.282612, sn_dec=1.932054,
         hst="040", jwst="A10", euclid="101542818"),
    dict(id=471959, seq=5, sn_ra=150.278761, sn_dec=2.430970,
         hst="076", jwst="B3",  euclid="101545698"),
]
CUTOUT_SIZE = 6.0 * u.arcsec
ID_TAG = "known34c"

OUT_HST    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
OUT_EUCLID = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")
OUT_JWST   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")

HST_DIR  = "/Volumes/exdisk1/data/HST/COSMOS_v2.0"
JWST_DIR = "/Volumes/exdisk1/data/JWST/COSMOS_v0.8"
EU_DIR   = "/Volumes/exdisk1/data/Euclid/COSMOS_DR1"


# ============================================================
def render_gray(data, png_path):
    minimum = 0.0001
    bw = np.where(data > minimum, data, minimum); bw = np.sqrt(bw)
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if val < 1.0 else val * 0.85
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(bw, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
    ax.axis("off"); fig.tight_layout(pad=0)
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
        if val > maximum: maximum = val
    maximum = max(1.0, maximum * 0.85)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower"); ax.axis("off")
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
        if val > maximum: maximum = val
    maximum = max(1.0, maximum * 1.2)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower"); ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


# ============================================================
def compute_roll_angle(wcs, ra, dec):
    x0, y0 = wcs.all_world2pix(ra, dec, 0)
    xn, yn = wcs.all_world2pix(ra, dec + 1.0/3600.0, 0)
    return float(np.degrees(np.arctan2(float(xn - x0), float(yn - y0))))


def resolve_jwst_path(jwst_tile, band):
    """Prefer uncompressed."""
    base = f"{JWST_DIR}/mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v1.0_i2d.fits"
    if Path(base).exists():
        return base
    return base + ".gz"


def resolve_euclid_path(eu_tile, band):
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{eu_tile}-*.fits"))
    return matches[-1] if matches else None


def cut(path, pos, north_up_jwst=False):
    """Cut 6'' window.  If north_up_jwst is True, rotate -roll to put N up."""
    with fits.open(path, memmap=True) as h:
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(sci_hdu.header)
        c = Cutout2D(sci_hdu.data.astype(np.float32), pos, size=CUTOUT_SIZE, wcs=wcs)
    if north_up_jwst:
        angle = compute_roll_angle(wcs, pos.ra.deg, pos.dec.deg)
        data = ndi_rotate(c.data, -angle, reshape=False, order=3,
                          mode="constant", cval=0.0)
        c.data[:] = data
    return c


def main():
    for src in SOURCES:
        pos = SkyCoord(src["sn_ra"], src["sn_dec"], unit="deg", frame="icrs")
        seqn = f"{src['seq']:04d}"
        print(f"\n==== ID={src['id']} seq={seqn} ====", flush=True)
        print(f"  SN @ RA={src['sn_ra']:.6f} Dec={src['sn_dec']:.6f}", flush=True)

        # HST
        try:
            hst_path = f"{HST_DIR}/acs_I_030mas_{src['hst']}_sci.fits"
            if not Path(hst_path).exists():
                hst_path += ".gz"
            print(f"  HST F814W ({Path(hst_path).name}) ...", flush=True)
            c = cut(hst_path, pos)
            render_gray(c.data, OUT_HST / f"sn_{ID_TAG}_{seqn}_hst.png")
        except Exception as e:
            print(f"    HST FAILED: {e}", flush=True)

        # Euclid VIS
        try:
            p = resolve_euclid_path(src["euclid"], "VIS")
            print(f"  Euclid VIS ({Path(p).name}) ...", flush=True)
            c = cut(p, pos)
            render_gray(c.data, OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_vis.png")
        except Exception as e:
            print(f"    Euclid VIS FAILED: {e}", flush=True)

        # Euclid NISP Y/J/H
        try:
            cy_d = cut(resolve_euclid_path(src["euclid"], "NIR-Y"), pos).data
            cj_d = cut(resolve_euclid_path(src["euclid"], "NIR-J"), pos).data
            ch_d = cut(resolve_euclid_path(src["euclid"], "NIR-H"), pos).data
            print(f"  Euclid NISP Y/J/H ...", flush=True)
            render_rgb_nisp(cy_d, cj_d, ch_d, OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_nisp.png")
        except Exception as e:
            print(f"    Euclid NISP FAILED: {e}", flush=True)

        # JWST jwst1 (F115/F150/F277) N-up
        try:
            c115 = cut(resolve_jwst_path(src["jwst"], "f115w"), pos, north_up_jwst=True).data
            c150 = cut(resolve_jwst_path(src["jwst"], "f150w"), pos, north_up_jwst=True).data
            c277 = cut(resolve_jwst_path(src["jwst"], "f277w"), pos, north_up_jwst=True).data
            print(f"  JWST F115/F150/F277 (N-up) ...", flush=True)
            render_rgb_jwst(c115, c150, c277, OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst1.png")
            c444 = cut(resolve_jwst_path(src["jwst"], "f444w"), pos, north_up_jwst=True).data
            print(f"  JWST F150/F277/F444 (N-up) ...", flush=True)
            render_rgb_jwst(c150, c277, c444, OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst2.png")
        except Exception as e:
            print(f"    JWST FAILED: {e}", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
