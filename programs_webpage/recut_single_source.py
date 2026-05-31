"""Fast single-source re-cutout for the SN-centered known34c test.

For ID=318858, opens just the known tiles (HST 041, JWST A9 x4 bands,
Euclid 101541377 x4 bands), cuts 6" around the SN coords, and renders the
five PNGs using the SAME recipes the production cutout scripts use.

Output: PNGs at sn_known34c_0001_{hst,euclid_vis,euclid_nisp,jwst1,jwst2}.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Target -- the SN-centered re-cut of ID=318858
SN_RA, SN_DEC = 150.185329, 1.842655
ID_TAG  = "known34c"
SEQ_NUM = 1

CUTOUT_SIZE = 6.0 * u.arcsec

OUT_HST    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
OUT_EUCLID = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")
OUT_JWST   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")

# Known tiles for ID=318858 (from existing sn_known34_*_index.csv files)
HST_TILE  = "/Volumes/exdisk1/data/HST/COSMOS_v2.0/acs_I_030mas_041_sci.fits.gz"
JWST_DIR  = "/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir"
JWST_TILE = "A9"
JWST_TPL  = JWST_DIR + "/mosaic_nircam_{b}_COSMOS-Web_30mas_" + JWST_TILE + "_v0_8_sci.fits"
EU_DIR    = "/Volumes/exdisk1/data/Euclid/COSMOS_DR1"
EU_TILE   = "101541377"
EU_PATHS  = {
    "VIS":   None,  # filled in by glob below
    "NIR-Y": None,
    "NIR-J": None,
    "NIR-H": None,
}

# Locate each Euclid tile file (the date stamp varies per file).
import glob
for band in EU_PATHS:
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{EU_TILE}-*.fits"))
    if not matches:
        raise FileNotFoundError(f"no Euclid {band} tile for {EU_TILE}")
    EU_PATHS[band] = matches[-1]  # newest


# ============================================================
# Rendering recipes (copied verbatim from the production scripts)
# ============================================================

def render_gray(data, png_path, label_lines):
    """create_blackwhite_hst() recipe (also used for Euclid VIS).

    2026-05-28: auto-stretch tracks the central peak (no hard 1.0 floor on
    vmax). Previously `max(1.0, val * 0.85)` washed out HST + Euclid-VIS
    faint sources whose sqrt-peak fell below 1.0 (HST e⁻/s units make this
    typical for mag ≥ 22). Now vmax = central_peak × 0.85 directly, with a
    small floor only to avoid vmax = 0 on truly blank cutouts. Matches the
    JWST stretch in spirit (central-peak-driven) but tuned for HST/Euclid-VIS
    pixel units which are far smaller than JWST's MJy/sr.
    """
    minimum = 0.0001
    bw = np.where(data > minimum, data, minimum)
    bw = np.sqrt(bw)
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    # vmax: blend 1/3 old (floor=1.0) + 2/3 new (floor=0.05) per user 2026-05-28.
    # For bright sources (val*0.85 > 1.0) both branches collapse to val*0.85 →
    # unchanged. For faint sources the blend gives vmax somewhere between the
    # too-dim old (vmax=1.0) and the too-bright new (vmax=val*0.85).
    vmax_old = max(1.0,  val * 0.85)
    vmax_new = max(0.05, val * 0.85)
    maximum = (vmax_old + 2.0 * vmax_new) / 3.0

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


def render_rgb_jwst(b_data, g_data, r_data, png_path, label_lines):
    """JWST color recipe: sqrt, central max, max*0.85."""
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
    for k, txt in enumerate(label_lines):
        ax.text(0.03, 0.95 - 0.07*k, txt, color="white",
                fontsize=11, transform=ax.transAxes,
                verticalalignment="top",
                family="serif", weight="bold")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


def render_rgb_nisp(b_data, g_data, r_data, png_path, label_lines):
    """Euclid NISP color recipe.

    2026-05-28: matched to JWST recipe (minimum=1e-4, multiplier=0.85) per
    user feedback that the prior NISP-specific recipe (minimum=1e-5,
    multiplier=1.2) made NISP cutouts visibly brighter than the contemporaneous
    JWST RGBs on the same page.

    2026-05-28 (final iteration): per-channel background subtraction +
    JWST-style vmin/vmax (floor=1.0). NISP sky is 10-50x higher than
    HST/JWST/VIS in sqrt-units; just adjusting vmin/vmax can't compress
    the elevated sky pedestal into JWST's display range. After subtracting
    the per-channel median background, NISP residual values behave like
    JWST naturally does (bg ~ 0, source above), and the SAME vmin/vmax
    that work for JWST produce the SAME visual outcome for NISP.
    """
    minimum = 1e-4
    # Per-channel background subtraction (median of the cutout). NISP sky
    # is 10-50x higher than HST/JWST/VIS in sqrt-units; subtraction gives a
    # zero-mean image that scales like JWST.
    def _sub_bg(d):
        if d is None or not np.any(np.isfinite(d)):
            return d
        return d - float(np.nanmedian(d))
    b_data = _sub_bg(b_data)
    g_data = _sub_bg(g_data)
    r_data = _sub_bg(r_data)
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
    # vmax floor: JWST uses 1.0 because JWST peak_sqrt is typically 0.3-0.7,
    # safely under 1/0.85=1.18 (saturation threshold). NISP, even AFTER bg
    # subtraction, has peak_sqrt ~ 1.0-2.0 (its 300mas pixels integrate ~100x
    # more source photons per pixel than JWST's 30mas), so floor=1.0 saturates
    # many bright sources. floor=4.0 (user-tuned 2026-05-28) keeps bright
    # NISP sources well clear of saturation: peak_sqrt up to 4.7 displays as
    # grey rather than white. Typical NISP source (peak_sqrt~1.5) at 38% grey.
    maximum = max(4.0, maximum * 0.85)

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


# ============================================================
# Do the cuts
# ============================================================

def cut(path, pos):
    with fits.open(path, memmap=True) as h:
        wcs = WCS(h[0].header)
        data = h[0].data
        return Cutout2D(data.astype(np.float32), pos, size=CUTOUT_SIZE, wcs=wcs)


def main():
    pos = SkyCoord(SN_RA, SN_DEC, unit="deg", frame="icrs")
    print(f"Re-cutting around SN @ RA={SN_RA:.6f}  Dec={SN_DEC:.6f}")
    seqn = f"{SEQ_NUM:04d}"

    # ---------- HST ----------
    print("HST F814W ...")
    cut_hst = cut(HST_TILE, pos)
    name = f"sn_{ID_TAG}_{seqn}_hst.png"
    render_gray(cut_hst.data, OUT_HST / name, ["ID=318858", "F814W"])
    print(f"  wrote {OUT_HST / name}")

    # ---------- Euclid VIS ----------
    print("Euclid VIS ...")
    cut_vis = cut(EU_PATHS["VIS"], pos)
    name = f"sn_{ID_TAG}_{seqn}_euclid_vis.png"
    render_gray(cut_vis.data, OUT_EUCLID / name, ["ID=318858", "VIS"])
    print(f"  wrote {OUT_EUCLID / name}")

    # ---------- Euclid NISP Y/J/H ----------
    print("Euclid NISP Y/J/H ...")
    cy = cut(EU_PATHS["NIR-Y"], pos)
    cj = cut(EU_PATHS["NIR-J"], pos)
    ch = cut(EU_PATHS["NIR-H"], pos)
    name = f"sn_{ID_TAG}_{seqn}_euclid_nisp.png"
    render_rgb_nisp(cy.data, cj.data, ch.data, OUT_EUCLID / name,
                    ["ID=318858", "Y/J/H"])
    print(f"  wrote {OUT_EUCLID / name}")

    # ---------- JWST F115/F150/F277 ----------
    print("JWST jwst1 (F115/F150/F277) ...")
    c115 = cut(JWST_TPL.format(b="f115w"), pos)
    c150 = cut(JWST_TPL.format(b="f150w"), pos)
    c277 = cut(JWST_TPL.format(b="f277w"), pos)
    name = f"sn_{ID_TAG}_{seqn}_jwst1.png"
    render_rgb_jwst(c115.data, c150.data, c277.data, OUT_JWST / name,
                    ["ID=318858", "F115/F150/F277"])
    print(f"  wrote {OUT_JWST / name}")

    # ---------- JWST F150/F277/F444 ----------
    print("JWST jwst2 (F150/F277/F444) ...")
    c444 = cut(JWST_TPL.format(b="f444w"), pos)
    name = f"sn_{ID_TAG}_{seqn}_jwst2.png"
    render_rgb_jwst(c150.data, c277.data, c444.data, OUT_JWST / name,
                    ["ID=318858", "F150/F277/F444"])
    print(f"  wrote {OUT_JWST / name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
