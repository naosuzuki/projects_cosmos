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
    # MEASURED SN positions from HST F814W centroid (2026-05-26):
    # Centroid on 1.0"-1.5" search box around user's refined eyeball hint.
    # All centroids landed within 0.05-0.18" of the hint, confirming the
    # eyeball was a reasonable starting point; these are the data-driven
    # final positions.
    #
    # NB: seq=1/2/3 (not 3/4/5) so the output PNG naming
    # sn_known34c_000{1,2,3}_*.png aligns with the polish step's
    # row-index-as-seq convention when called with a 3-row CSV.
    # The v05 PNGs at seq 0003/0004/0005 are preserved in v05/.
    dict(id=130972, seq=1, sn_ra=150.279695, sn_dec=2.041101,
         hst="052", jwst="A4",  euclid="101542818",
         host_ra=150.279917, host_dec=2.041285,
         telescope="HST"),
    dict(id=371996, seq=2, sn_ra=150.282844, sn_dec=1.932259,
         hst="040", jwst="A10", euclid="101542818",
         host_ra=150.282612, host_dec=1.932054,
         telescope="HST"),
    dict(id=471959, seq=3, sn_ra=150.278904, sn_dec=2.430962,
         hst="076", jwst="B3",  euclid="101545698",
         host_ra=150.278761, host_dec=2.430970,
         telescope="HST"),
    # Euclid NISP-only SN; NISP-H centroid 0.04" from catalog → SN essentially
    # at host position (transient sits on top of host in NISP).
    dict(id=63924,  seq=4, sn_ra=149.968179, sn_dec=2.233132,
         hst="066", jwst="A2",  euclid="101544256",
         host_ra=149.968173, host_dec=2.233122,
         telescope="EUCLID"),
]

# Aperture-photometry constants (must match measure_sn_5.py)
APER_ARCSEC      = 0.20
HOST_RING_IN_AS  = 0.25
HOST_RING_OUT_AS = 0.40

# Band catalogue: (label, kind, sci_path_template_func, err_path_template_func)
def _hst_paths(tile):
    return (f"{HST_DIR}/acs_I_030mas_{tile}_sci.fits",
            f"{HST_DIR}/acs_I_030mas_{tile}_wht.fits")
def _eu_path(tile, band):
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{tile}-*.fits"))
    return (matches[-1] if matches else None, None)
def _jwst_paths(tile, band):
    return (resolve_jwst_path(tile, band), None)

BANDS_PHOT = [   # (label, kind, fn-returning(sci_path, err_path))
    ("F814W", "hst",    lambda src: _hst_paths(src["hst"])),
    ("VIS",   "euclid", lambda src: _eu_path(src["euclid"], "VIS")),
    ("Y",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-Y")),
    ("J",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-J")),
    ("H",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-H")),
    ("F115W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f115w")),
    ("F150W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f150w")),
    ("F277W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f277w")),
    ("F444W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f444w")),
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
def _draw_crosshair(ax, ny, nx, color="lime"):
    """Tiny '+' centered on the cutout (= SN position by construction)."""
    cy, cx = ny / 2 - 0.5, nx / 2 - 0.5
    gap = max(3, min(ny, nx) * 0.025)        # gap between arms
    arm = max(6, min(ny, nx) * 0.06)         # arm length
    lw  = max(1.0, min(ny, nx) / 200.0)
    ax.plot([cx, cx], [cy + gap, cy + gap + arm], color=color, lw=lw)
    ax.plot([cx, cx], [cy - gap, cy - gap - arm], color=color, lw=lw)
    ax.plot([cx + gap, cx + gap + arm], [cy, cy], color=color, lw=lw)
    ax.plot([cx - gap, cx - gap - arm], [cy, cy], color=color, lw=lw)


def render_gray(data, png_path, crosshair=True):
    minimum = 0.0001
    bw = np.where(data > minimum, data, minimum); bw = np.sqrt(bw)
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if val < 1.0 else val * 0.85
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(bw, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
    if crosshair: _draw_crosshair(ax, ny, nx)
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off"); fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_jwst(b_data, g_data, r_data, png_path, crosshair=True):
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
    ax.imshow(rgb, origin="lower")
    ny, nx = b.shape
    if crosshair: _draw_crosshair(ax, ny, nx, color="lime")
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_nisp(b_data, g_data, r_data, png_path, crosshair=True):
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
    ax.imshow(rgb, origin="lower")
    ny, nx = b.shape
    if crosshair: _draw_crosshair(ax, ny, nx, color="lime")
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


# ============================================================
def compute_roll_angle(wcs, ra, dec):
    x0, y0 = wcs.all_world2pix(ra, dec, 0)
    xn, yn = wcs.all_world2pix(ra, dec + 1.0/3600.0, 0)
    return float(np.degrees(np.arctan2(float(xn - x0), float(yn - y0))))


def resolve_jwst_path(jwst_tile, band):
    """Resolve JWST tile file with safe fallbacks.

    Preferred order:
      1. Uncompressed v1.0 i2d (.fits)
      2. v0.8 sci.fits (no ERR HDU but always uncompressed and intact)
      3. v1.0 i2d.fits.gz (LAST resort; gzipped, slow, and may be partial)

    We skip .gz when a v0.8 sci.fits is available because the .gz can be
    partial mid-download and astropy will crash on the corrupt block.
    """
    base = f"{JWST_DIR}/mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v1.0_i2d.fits"
    if Path(base).exists():
        return base
    v08 = f"{JWST_DIR}/scidir/mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v0_8_sci.fits"
    if Path(v08).exists():
        return v08
    return base + ".gz"


def resolve_euclid_path(eu_tile, band):
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{eu_tile}-*.fits"))
    return matches[-1] if matches else None


_FITS_CACHE = {}   # path -> open HDUList (kept open for the script's lifetime)
_WCS_CACHE  = {}   # path -> WCS (avoids re-parsing the header per call)
_Z_LOOKUP   = None # id -> zfinal (LEPHARE), built once on first lookup


# Catalog paths for host-z lookup (row-matched MASTER + LEPHARE).
MASTER_FITS  = "/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits"
LEPHARE_FITS = "/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1_lephare.fits"


def _open_cached(path):
    if path not in _FITS_CACHE:
        _FITS_CACHE[path] = fits.open(path, memmap=True)
    return _FITS_CACHE[path]


def _wcs_cached(path, hdu):
    if path not in _WCS_CACHE:
        _WCS_CACHE[path] = WCS(hdu.header)
    return _WCS_CACHE[path]


# ----- DAOStarFinder-based SN point-source detection -----
#
# v13 design: replace the home-brew "smooth + peak + bg-subtract" centroid
# (which kept finding the host galaxy instead of the SN) with the standard
# astropy/photutils DAOPHOT FIND implementation.  DAOStarFinder convolves
# the image with a Gaussian of the PSF FWHM, so its response is strong only
# for compact PSF-shaped sources and weak for extended host emission —
# exactly the discrimination we need for SN-near-host detection.
#
# Per-telescope detection band + PSF FWHM (used by --find-sn):
#   HST     -> F814W         FWHM ~ 0.090"
#   JWST    -> F115W (bluest, sharpest PSF, SN usually brighter blue)
#                            FWHM ~ 0.045"
#   EUCLID  -> NIR-H         FWHM ~ 0.48"

def _hst_path_fn(src):
    return f"{HST_DIR}/acs_I_030mas_{src['hst']}_sci.fits"

def _jwst_f115w_path_fn(src):
    return resolve_jwst_path(src["jwst"], "f115w")

def _eu_nisp_h_path_fn(src):
    return _eu_path(src["euclid"], "NIR-H")[0]

DETECT_BAND = {
    # telescope-tag -> (psf_fwhm_arcsec, path_fn(src) -> sci_path)
    "HST":         (0.090, _hst_path_fn),
    "JWST":        (0.045, _jwst_f115w_path_fn),
    "EUCLID":      (0.48,  _eu_nisp_h_path_fn),
    "EUCLID-NISP": (0.48,  _eu_nisp_h_path_fn),
    "NISP":        (0.48,  _eu_nisp_h_path_fn),
}


def _dao_fail(hint_ra, hint_dec, n=0):
    """Fallback dict when DAO cannot find a compact source: keep the hint."""
    return dict(success=False, sn_ra=float(hint_ra), sn_dec=float(hint_dec),
                n_candidates=int(n), peak=0.0, moved_arcsec=0.0)


def find_sn_via_dao(sci_path, hint_ra, hint_dec, psf_fwhm_arcsec=0.1,
                    search_arcsec=2.5, threshold_sigma=5.0):
    """Find SN point source near (hint_ra, hint_dec) using DAOStarFinder.

    Algorithm (all on a small cutout, sliced from the memory-mapped HDU):
      1. sigma_clipped_stats -> robust local background median + std
      2. DAOStarFinder(fwhm=PSF_in_pixels, threshold=Nσ) -> compact sources
         (the FWHM convolution kernel suppresses extended host emission)
      3. keep only candidates within `search_arcsec` of the hint
      4. pick the BRIGHTEST candidate (`peak` flux)
      5. return its WCS RA/Dec (already a subpixel centroid from DAO)

    File-open efficient: uses the shared _FITS_CACHE / _WCS_CACHE so the
    same FITS is opened only once per process even when cut() / aper_photometry
    read it again later.

    Returns dict:
        success         True if a compact source was found within search_arcsec
        sn_ra, sn_dec   refined SN position (= hint if no detection)
        n_candidates    number of compact sources DAO found in the search box
        peak            peak flux of the chosen source
        moved_arcsec    distance the position moved from the hint
    """
    try:
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        h = _open_cached(sci_path)
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = _wcs_cached(sci_path, sci_hdu)
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
        fwhm_px = max(1.5, psf_fwhm_arcsec / pix_scale)
        box_arcsec = max(search_arcsec * 2.0, 5.0)
        half = int(np.ceil(box_arcsec / pix_scale)) + 1
        sx, sy = wcs.all_world2pix(hint_ra, hint_dec, 0)
        sx, sy = float(sx), float(sy)
        ny = int(sci_hdu.header.get("NAXIS2", 0)) or sci_hdu.data.shape[-2]
        nx = int(sci_hdu.header.get("NAXIS1", 0)) or sci_hdu.data.shape[-1]
        x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
        y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
        # SLICE FIRST (memory-mapped, only the box), then astype.
        sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
        if sub.size == 0:
            return _dao_fail(hint_ra, hint_dec)
        finite = np.isfinite(sub) & (sub != 0)
        if not finite.any():
            return _dao_fail(hint_ra, hint_dec)
        _, bg_median, bg_std = sigma_clipped_stats(sub[finite], sigma=3.0, maxiters=3)
        finder = DAOStarFinder(fwhm=fwhm_px, threshold=threshold_sigma * bg_std)
        sources = finder(sub - bg_median)
        if sources is None or len(sources) == 0:
            return _dao_fail(hint_ra, hint_dec)
        hint_px_x = sx - x0
        hint_px_y = sy - y0
        dx_px = np.asarray(sources["xcentroid"]) - hint_px_x
        dy_px = np.asarray(sources["ycentroid"]) - hint_px_y
        r_arcsec = np.sqrt(dx_px**2 + dy_px**2) * pix_scale
        in_search = r_arcsec <= search_arcsec
        if not in_search.any():
            return _dao_fail(hint_ra, hint_dec, n=len(sources))
        candidates = sources[in_search]
        best_idx = int(np.argmax(np.asarray(candidates["peak"])))
        best = candidates[best_idx]
        cx_full = float(best["xcentroid"]) + x0
        cy_full = float(best["ycentroid"]) + y0
        sn_ra, sn_dec = wcs.all_pix2world(cx_full, cy_full, 0)
        moved = float(SkyCoord(hint_ra, hint_dec, unit="deg").separation(
                      SkyCoord(float(sn_ra), float(sn_dec), unit="deg")).to(u.arcsec).value)
        return dict(success=True, sn_ra=float(sn_ra), sn_dec=float(sn_dec),
                    n_candidates=int(in_search.sum()),
                    peak=float(best["peak"]),
                    moved_arcsec=moved)
    except Exception as e:
        print(f"      DAO error on {sci_path}: {type(e).__name__}: {e}", flush=True)
        return _dao_fail(hint_ra, hint_dec)


def find_sn_point_source(sci_path, hint_ra, hint_dec, kind="generic",
                         box_arcsec=2.0, search_arcsec=0.6, bg_inner=1.0, bg_outer=1.8):
    """Locate a SN point source near (hint_ra, hint_dec) in `sci_path`.

    Algorithm:
      1. Cut a `box_arcsec` box around the hint (memory-mapped slice — fast).
      2. Estimate local background from an annulus (bg_inner..bg_outer arcsec)
         around the box centre — the annulus stays within the box and samples
         the host's smooth light.
      3. Subtract the background → residual image (host's extended light gone).
      4. Smooth the residual lightly (Gaussian σ ≈ 0.4 arcsec, ~ Euclid PSF).
      5. Find the peak in the residual within `search_arcsec` of the hint
         — that's the SN point source.
      6. Subpixel centroid using intensity-weighted moments on a 3-px box.
    Returns (sn_ra, sn_dec) or (hint_ra, hint_dec) on failure.
    """
    try:
        from scipy.ndimage import gaussian_filter
        h = _open_cached(sci_path)
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = _wcs_cached(sci_path, sci_hdu)
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
        half_px = int(np.ceil(box_arcsec / pix_scale)) + 1
        sx, sy = wcs.all_world2pix(hint_ra, hint_dec, 0)
        sx, sy = float(sx), float(sy)
        ny = int(sci_hdu.header.get("NAXIS2", 0))
        nx = int(sci_hdu.header.get("NAXIS1", 0))
        x0 = max(0, int(sx) - half_px); x1 = min(nx, int(sx) + half_px + 1)
        y0 = max(0, int(sy) - half_px); y1 = min(ny, int(sy) + half_px + 1)
        sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
        if sub.size == 0:
            return float(hint_ra), float(hint_dec)
        # Distance from hint (in sub-box pixel coords)
        yy, xx = np.indices(sub.shape)
        hx = sx - x0; hy = sy - y0
        r_px = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        ring = (r_px >= bg_inner / pix_scale) & (r_px <= bg_outer / pix_scale)
        if ring.any():
            bg = float(np.nanmedian(sub[ring]))
        else:
            bg = float(np.nanmedian(sub))
        resid = sub - bg
        # Smooth ~ 1 PSF FWHM
        sigma_px = max(0.5, 0.4 / pix_scale)
        smoothed = gaussian_filter(np.where(np.isfinite(resid), resid, 0.0), sigma=sigma_px)
        # Only look within search_arcsec of the hint
        search_mask = r_px <= search_arcsec / pix_scale
        masked = np.where(search_mask, smoothed, -np.inf)
        py, px = np.unravel_index(np.nanargmax(masked), masked.shape)
        if not np.isfinite(masked[py, px]):
            return float(hint_ra), float(hint_dec)
        # Subpixel centroid (small box around the peak)
        h_box = 2
        yy0, yy1 = max(0, py-h_box), min(smoothed.shape[0], py+h_box+1)
        xx0, xx1 = max(0, px-h_box), min(smoothed.shape[1], px+h_box+1)
        p = smoothed[yy0:yy1, xx0:xx1].copy()
        p = np.where(p > 0, p, 0)
        if p.sum() > 0:
            ys, xs = np.indices(p.shape)
            cx = (xs * p).sum() / p.sum() + xx0
            cy = (ys * p).sum() / p.sum() + yy0
        else:
            cx, cy = float(px), float(py)
        cx_full = cx + x0
        cy_full = cy + y0
        sn_ra, sn_dec = wcs.all_pix2world(cx_full, cy_full, 0)
        return float(sn_ra), float(sn_dec)
    except Exception as e:
        print(f"      (find_sn_point_source failed on {sci_path}: {e})", flush=True)
        return float(hint_ra), float(hint_dec)


def _lephare_z(cid):
    """LEPHARE zfinal for catalog id `cid`.  Lazy-loads a dict on first call
    (just the `id` column of MASTER + the `zfinal` column of LEPHARE — much
    cheaper than reading the full Tables).  Returns NaN if not found."""
    global _Z_LOOKUP
    if _Z_LOOKUP is None:
        with fits.open(MASTER_FITS, memmap=True) as h:
            ids = np.asarray(h[1].data["id"]).astype(int)
        with fits.open(LEPHARE_FITS, memmap=True) as h:
            zs = np.asarray(h[1].data["zfinal"]).astype(float)
        _Z_LOOKUP = dict(zip(ids.tolist(), zs.tolist()))
    return float(_Z_LOOKUP.get(int(cid), float("nan")))


def aper_photometry(sci_path, err_path, kind, sn_ra, sn_dec):
    """Aperture photometry at (sn_ra, sn_dec).
    Returns (mag_AB, snr).  Uses the same constants as measure_sn_5.py.
    Re-uses _FITS_CACHE — OS page cache makes this near-free if cut()
    has already opened the file.
    """
    if sci_path is None:
        return None, 0.0
    try:
        h = _open_cached(sci_path)
        hdu_names = [x.name for x in h]
        if kind == "jwst" and "SCI" in hdu_names:
            sci_hdu = h["SCI"]
        else:
            sci_hdu = h[0]
        hdr = sci_hdu.header
        wcs = _wcs_cached(sci_path, sci_hdu)
    except Exception as e:
        print(f"      (open failed {sci_path}: {type(e).__name__}: {e})", flush=True)
        return None, 0.0
    try:
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    except Exception:
        pix_scale = 0.03
    aper_px  = APER_ARCSEC      / pix_scale
    ring_in  = HOST_RING_IN_AS  / pix_scale
    ring_out = HOST_RING_OUT_AS / pix_scale
    half = int(np.ceil(max(ring_out, 2.0 / pix_scale)) + 2)

    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    # Get shape from header so we never materialise the full image.
    ny = int(hdr.get("NAXIS2", 0)); nx = int(hdr.get("NAXIS1", 0))
    if ny == 0 or nx == 0:
        # Fallback: read shape from the memmap (cheap, no copy)
        ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx - half)); x1 = min(nx, int(sx + half + 1))
    y0 = max(0, int(sy - half)); y1 = min(ny, int(sy + half + 1))
    # SLICE FIRST (memory-mapped read of just the small box),
    # THEN astype.  Same fix as cut() — avoids reading 25 GB JWST tiles.
    sub_raw = sci_hdu.data[y0:y1, x0:x1]
    if sub_raw.size == 0: return None, 0.0
    sub = sub_raw.astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - sx)**2 + (yy - sy)**2)
    in_aper = rr <= aper_px; in_ring = (rr >= ring_in) & (rr <= ring_out)
    n_aper = int(in_aper.sum())
    if n_aper == 0 or in_ring.sum() == 0: return None, 0.0
    host_sb = float(np.nanmedian(sub[in_ring]))
    flux = float(np.nansum(sub[in_aper])) - host_sb * n_aper
    sigma_ring = float(np.nanstd(sub[in_ring]))
    sigma_aper = sigma_ring * np.sqrt(n_aper)
    snr = flux / (sigma_aper + 1e-30)
    bunit = (hdr.get("BUNIT") or "").strip()
    if "MJy/sr" in bunit or kind == "jwst":
        pix_sr = hdr.get("PIXAR_SR") or ((pix_scale / 206265.0) ** 2)
        f_jy = flux * float(pix_sr) * 1.0e6
        mag = (-2.5 * np.log10(f_jy / 3631.0)) if f_jy > 0 else None
    elif kind == "hst":
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
        mag = (float(zp) - 2.5 * np.log10(flux)) if flux > 0 else None
    else:
        zp = hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9
        mag = (float(zp) - 2.5 * np.log10(flux)) if flux > 0 else None
    return mag, float(snr)


def cut(path, pos, north_up_jwst=False):
    """Cut 6'' window with two efficiency tricks:
      - Slice from memory-mapped data FIRST, then astype on the small slice
        (avoids reading the full 500MB-30GB tile through .astype).
      - Cache open HDUList per path (shared Euclid tiles between sources).
    If north_up_jwst, rotate by -roll so North is up after cropping.
    """
    h = _open_cached(path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    wcs = _wcs_cached(path, sci_hdu)
    # copy=True forces Cutout2D to materialise the ~80KB small region only.
    c = Cutout2D(sci_hdu.data, pos, size=CUTOUT_SIZE, wcs=wcs, copy=True)
    # Cast the small cutout to float32 (cheap):
    c.data = np.asarray(c.data, dtype=np.float32)
    if north_up_jwst:
        angle = compute_roll_angle(wcs, pos.ra.deg, pos.dec.deg)
        data = ndi_rotate(c.data, -angle, reshape=False, order=3,
                          mode="constant", cval=0.0)
        c.data = data
    return c


def main():
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-crosshair", action="store_true",
                    help="skip the simple '+' crosshair (for downstream polish)")
    ap.add_argument("--measure", action="store_true",
                    help="also do aperture photometry per band and write CSV")
    ap.add_argument("--csv", default="/tmp/sn_3.csv",
                    help="output CSV path when --measure is set")
    ap.add_argument("--input-csv", default=None,
                    help="lookup-table CSV with columns: id, telescope, sn_ra, "
                         "sn_dec, host_ra, host_dec, hst, jwst, euclid. "
                         "Replaces the hardcoded SOURCES list when provided.")
    ap.add_argument("--find-sn", action="store_true",
                    help="run DAOStarFinder on the detection band per source "
                         "(HST->F814W, JWST->F115W, EUCLID->NIR-H) to refine "
                         "the SN position before cutout+photometry. Pure peak "
                         "detection + intensity centroid; no host-galaxy model.")
    ap.add_argument("--search-arcsec", type=float, default=2.5,
                    help="DAO search radius around the input hint (arcsec)")
    ap.add_argument("--threshold-sigma", type=float, default=5.0,
                    help="DAO detection threshold in units of local σ")
    args = ap.parse_args()
    crosshair = not args.no_crosshair

    # Lookup-table mode: read sources from CSV (for mass production)
    if args.input_csv:
        with open(args.input_csv) as fh:
            rdr = csv.DictReader(fh)
            sources = []
            for i, r in enumerate(rdr, 1):
                sources.append(dict(
                    id=int(r["id"]), seq=i,
                    sn_ra=float(r["sn_ra"]), sn_dec=float(r["sn_dec"]),
                    host_ra=float(r["host_ra"]), host_dec=float(r["host_dec"]),
                    hst=r["hst"], jwst=r["jwst"], euclid=r["euclid"],
                    telescope=r.get("telescope", "JWST").strip().upper(),
                ))
        print(f"Loaded {len(sources)} sources from {args.input_csv}", flush=True)
    else:
        sources = SOURCES

    phot_rows = [] if args.measure else None
    for src in sources:
        seqn = f"{src['seq']:04d}"
        print(f"\n==== ID={src['id']} seq={seqn} ====", flush=True)
        # --- Optional DAO point-source refinement (BEFORE pos/cut) ---
        if args.find_sn:
            tel = src.get("telescope", "JWST").strip().upper()
            entry = DETECT_BAND.get(tel)
            if entry is None:
                print(f"  DAO: no detection band defined for telescope={tel}; "
                      f"keeping hint", flush=True)
            else:
                fwhm_arcsec, path_fn = entry
                det_path = path_fn(src)
                if det_path is None or not Path(det_path).exists():
                    print(f"  DAO: detection file missing ({det_path}); "
                          f"keeping hint", flush=True)
                else:
                    r = find_sn_via_dao(det_path, src["sn_ra"], src["sn_dec"],
                                        psf_fwhm_arcsec=fwhm_arcsec,
                                        search_arcsec=args.search_arcsec,
                                        threshold_sigma=args.threshold_sigma)
                    flag = "OK " if r["success"] else "no compact source"
                    print(f"  DAO ({tel}, fwhm={fwhm_arcsec}\"): {flag}  "
                          f"moved={r['moved_arcsec']:.2f}\"  "
                          f"candidates={r['n_candidates']}  "
                          f"peak={r['peak']:.4g}", flush=True)
                    if r["success"]:
                        src["sn_ra"]  = r["sn_ra"]
                        src["sn_dec"] = r["sn_dec"]
        pos = SkyCoord(src["sn_ra"], src["sn_dec"], unit="deg", frame="icrs")
        print(f"  SN @ RA={src['sn_ra']:.6f} Dec={src['sn_dec']:.6f}", flush=True)

        # HST
        try:
            hst_path = f"{HST_DIR}/acs_I_030mas_{src['hst']}_sci.fits"
            if not Path(hst_path).exists():
                hst_path += ".gz"
            print(f"  HST F814W ({Path(hst_path).name}) ...", flush=True)
            c = cut(hst_path, pos)
            render_gray(c.data, OUT_HST / f"sn_{ID_TAG}_{seqn}_hst.png",
                        crosshair=crosshair)
        except Exception as e:
            print(f"    HST FAILED: {e}", flush=True)

        # Euclid VIS
        try:
            p = resolve_euclid_path(src["euclid"], "VIS")
            print(f"  Euclid VIS ({Path(p).name}) ...", flush=True)
            c = cut(p, pos)
            render_gray(c.data, OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_vis.png",
                        crosshair=crosshair)
        except Exception as e:
            print(f"    Euclid VIS FAILED: {e}", flush=True)

        # Euclid NISP Y/J/H
        try:
            cy_d = cut(resolve_euclid_path(src["euclid"], "NIR-Y"), pos).data
            cj_d = cut(resolve_euclid_path(src["euclid"], "NIR-J"), pos).data
            ch_d = cut(resolve_euclid_path(src["euclid"], "NIR-H"), pos).data
            print(f"  Euclid NISP Y/J/H ...", flush=True)
            render_rgb_nisp(cy_d, cj_d, ch_d,
                            OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_nisp.png",
                            crosshair=crosshair)
        except Exception as e:
            print(f"    Euclid NISP FAILED: {e}", flush=True)

        # JWST jwst1 (F115/F150/F277) N-up
        try:
            c115 = cut(resolve_jwst_path(src["jwst"], "f115w"), pos, north_up_jwst=True).data
            c150 = cut(resolve_jwst_path(src["jwst"], "f150w"), pos, north_up_jwst=True).data
            c277 = cut(resolve_jwst_path(src["jwst"], "f277w"), pos, north_up_jwst=True).data
            print(f"  JWST F115/F150/F277 (N-up) ...", flush=True)
            render_rgb_jwst(c115, c150, c277,
                            OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst1.png",
                            crosshair=crosshair)
            c444 = cut(resolve_jwst_path(src["jwst"], "f444w"), pos, north_up_jwst=True).data
            print(f"  JWST F150/F277/F444 (N-up) ...", flush=True)
            render_rgb_jwst(c150, c277, c444,
                            OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst2.png",
                            crosshair=crosshair)
        except Exception as e:
            print(f"    JWST FAILED: {e}", flush=True)

        # Aperture photometry (--measure).  Re-uses the same FITS_CACHE so
        # the per-band file is already open from the cut() calls above.
        if args.measure:
            host_ra_s  = float(src.get("host_ra", src["sn_ra"]))
            host_dec_s = float(src.get("host_dec", src["sn_dec"]))
            sep_arcsec = float(
                SkyCoord(src["sn_ra"], src["sn_dec"], unit="deg").separation(
                SkyCoord(host_ra_s, host_dec_s, unit="deg")).to(u.arcsec).value
            )
            row = dict(id=src["id"],
                       telescope=src.get("telescope", "HST"),
                       hst_tile=src.get("hst", ""),
                       jwst_tile=src.get("jwst", ""),
                       euclid_tile=src.get("euclid", ""),
                       host_ra=host_ra_s,
                       host_dec=host_dec_s,
                       sn_ra=src["sn_ra"],
                       sn_dec=src["sn_dec"],
                       sn_host_sep_arcsec=round(sep_arcsec, 3),
                       host_z=round(_lephare_z(src["id"]), 4))
            best_band = "F814W"; best_snr = -999.0
            for label, kind, paths_fn in BANDS_PHOT:
                sci_p, err_p = paths_fn(src)
                mag, snr = aper_photometry(sci_p, err_p, kind,
                                           src["sn_ra"], src["sn_dec"])
                row[f"mag_{label}"] = -1 if mag is None else round(mag, 2)
                row[f"snr_{label}"] = round(snr, 2)
                if snr > best_snr:
                    best_snr = snr; best_band = label
            row["best_band"] = best_band
            row["best_snr"]  = round(best_snr, 2)
            row["combined_sigma"] = round(best_snr, 2)  # for HST SN, dominated by F814W
            phot_rows.append(row)
            print(f"  photometry: best={best_band}@{best_snr:.1f}σ", flush=True)

    # Write CSV with extended schema (added 2026-05-26):
    #   * hst/jwst/euclid tile numbers (self-documenting — no external
    #     lookup needed to know which FITS each measurement came from)
    #   * sn_host_sep_arcsec (computed from SkyCoord.separation)
    #   * host_z = LEPHARE zfinal (cached lookup)
    if args.measure and phot_rows:
        cols = ["id","telescope",
                "hst_tile","jwst_tile","euclid_tile",
                "host_ra","host_dec","sn_ra","sn_dec","sn_host_sep_arcsec",
                "host_z",
                "best_band","best_snr","combined_sigma",
                "mag_F814W","mag_VIS","mag_Y","mag_J","mag_H",
                "mag_F115W","mag_F150W","mag_F277W","mag_F444W",
                "snr_F814W","snr_VIS","snr_Y","snr_J","snr_H",
                "snr_F115W","snr_F150W","snr_F277W","snr_F444W"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in phot_rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nWrote {args.csv}  ({len(phot_rows)} rows)", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
