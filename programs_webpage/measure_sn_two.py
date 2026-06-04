"""Detect SN, compute SN RA/Dec, and measure aperture mags for 2 test sources.

Two sources: ID=318858 (seq 23) and ID=320233 (seq 24) in known34.

For each:
  1. Load JWST1 PNG, detect SN via blue-anomaly inside the central box
  2. Convert PNG dx,dy -> native tile px -> SN RA/Dec using the JWST tile WCS
  3. Run aperture photometry on each band's FITS at the SN position:
       HST  F814W (gray)            -> SN faded -> typically -1
       Euclid VIS, Y, J, H (gray)   -> SN faded -> typically -1
       JWST F115/F150/F277/F444     -> SN present -> measured AB mag
     Aperture = 0.20", background annulus 0.5"-1.0", subtract local median.
     If aperture flux is <= 0 or < 3*sigma above bg -> mag = -1.

Output: csv with id, host_ra, host_dec, sn_ra, sn_dec, dx_pil, dy_pil,
        and one column per band: mag_<band>.
"""
import warnings
warnings.filterwarnings("ignore")

import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

# ----------------------------------------------------------------------
# Sources to process
SOURCES = [
    dict(seq=23, id=318858, host_ra=150.185272, host_dec=1.842447,
         hst_tile="041", jwst_tile="A9", euclid_tile="101541377"),
    dict(seq=24, id=320233, host_ra=150.146358, host_dec=1.866410,
         hst_tile="041", jwst_tile="A9", euclid_tile="101541377"),
]

# PNG geometry
PNG_SIZE          = 480
NATIVE_JWST_PX    = 200        # 6" at 30 mas
SEARCH_FRAC       = 0.15
LABEL_MASK_FRAC_Y = 0.20
LABEL_MASK_FRAC_X = 0.40

# Aperture photometry
APER_ARCSEC   = 0.20
ANNULUS_IN_AS = 0.50
ANNULUS_OUT_AS = 1.00
DETECT_SNR    = 3.0   # below this -> mag = -1

# Paths
HST_DIR    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
JWST_DIR   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir")
EU_DIR     = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")
JWST_PNG   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
OUT_CSV    = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_test_2sources.csv")

# Bands to measure, in column order:
BANDS = [
    # (label, kind, file-template-or-pattern, png_basename_suffix for SN-finding visual)
    ("F814W",   "hst",        "acs_I_030mas_{tile}_sci.fits.gz"),
    ("VIS",     "euclid",     "EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile}-*.fits"),
    ("Y",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE{tile}-*.fits"),
    ("J",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE{tile}-*.fits"),
    ("H",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE{tile}-*.fits"),
    ("F115W",   "jwst",       "mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits"),
    ("F150W",   "jwst",       "mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits"),
    ("F277W",   "jwst",       "mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits"),
    ("F444W",   "jwst",       "mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits"),
]

# ----------------------------------------------------------------------
# SN detection in JWST PNG (blue-anomaly method, same as test_sn_crosshair.py)

def detect_sn_in_png(jwst1_png_path):
    """Return (sx, sy) in PIL coords of the brightest blue-anomaly peak
    inside the central 30%-box, or None."""
    img = np.asarray(Image.open(jwst1_png_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    b, g, r = img[..., 2], img[..., 1], img[..., 0]
    blueness = b - 0.5*(r + g)
    # Mask label region (upper-left)
    blueness[:int(LABEL_MASK_FRAC_Y * h), :int(LABEL_MASK_FRAC_X * w)] = -10
    # Confine to central box
    cx, cy = w // 2, h // 2
    half = int(SEARCH_FRAC * min(w, h))
    box = np.full_like(blueness, -10)
    box[cy-half:cy+half, cx-half:cx+half] = blueness[cy-half:cy+half, cx-half:cx+half]
    yi, xi = np.unravel_index(np.argmax(box), box.shape)
    return int(xi), int(yi)


# ----------------------------------------------------------------------
# RA/Dec computation via tile WCS

def png_to_sky(host_ra, host_dec, sx_pil, sy_pil, tile_fits_path, native_size):
    """Convert SN PIL pixel offset to RA/Dec via the tile WCS."""
    cx_pil = PNG_SIZE // 2
    cy_pil = PNG_SIZE // 2
    # In native tile px: PIL +x = array +x; PIL +y is FLIPPED relative to array
    dx_native = (sx_pil - cx_pil) * (native_size / PNG_SIZE)
    dy_native = -(sy_pil - cy_pil) * (native_size / PNG_SIZE)
    with fits.open(tile_fits_path) as h:
        wcs = WCS(h[0].header)
    hx, hy = wcs.all_world2pix(host_ra, host_dec, 0)
    sn_ra, sn_dec = wcs.all_pix2world(hx + dx_native, hy + dy_native, 0)
    return float(sn_ra), float(sn_dec), float(dx_native), float(dy_native)


# ----------------------------------------------------------------------
# Aperture photometry

def aperture_phot_ab(fits_path, sn_ra, sn_dec, unit="auto"):
    """Aperture phot at (sn_ra, sn_dec).  Return (mag_ab, snr) or (None, snr).

    Handles three unit conventions:
      'MJy/sr'        : JWST sci.    flux_Jy = sum(data) * pixel_sr * 1e6
      'electron/s'    : HST acs.     flux_Jy = sum(data) * PHOTFLAM * lambda_pivot...
                                     too messy; we use ABZP from header (mag = ABZP - 2.5 log10(sum))
      'unknown'       : Euclid MER.  data may be in some calibrated unit; treat as ABZP=30 baseline
                                     (rough; output is upper-limit-only).
    """
    with fits.open(fits_path, memmap=True) as h:
        hdr = h[0].header
        data = h[0].data.astype(np.float64)
        wcs = WCS(hdr)

    # SN pixel in this tile
    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    # CD-matrix pixel scale (arcsec/px) - take the median of diagonal magnitudes
    try:
        cd = wcs.pixel_scale_matrix
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(cd)))) * 3600.0
    except Exception:
        pix_scale = float(abs(hdr.get("CD1_1", hdr.get("CDELT1", 1.0/30.0/3600.0)))) * 3600.0

    aper_px       = APER_ARCSEC   / pix_scale
    ann_in_px     = ANNULUS_IN_AS / pix_scale
    ann_out_px    = ANNULUS_OUT_AS / pix_scale
    half = int(np.ceil(ann_out_px + 2))

    # Slice a small window for speed
    ny, nx = data.shape
    x0 = int(max(0, sx - half));  x1 = int(min(nx, sx + half + 1))
    y0 = int(max(0, sy - half));  y1 = int(min(ny, sy + half + 1))
    sub = data[y0:y1, x0:x1]
    if sub.size == 0:
        return None, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - sx)**2 + (yy - sy)**2)

    in_aper = (rr <= aper_px)
    in_ann  = (rr >= ann_in_px) & (rr <= ann_out_px)
    if in_aper.sum() == 0 or in_ann.sum() == 0:
        return None, 0.0

    bg_med = float(np.nanmedian(sub[in_ann]))
    bg_std = float(np.nanstd(sub[in_ann]))
    flux_sum = float(np.nansum(sub[in_aper] - bg_med))
    snr = flux_sum / (bg_std * np.sqrt(in_aper.sum()) + 1e-12)

    # Convert flux to AB mag
    bunit = (hdr.get("BUNIT") or "").strip()
    if "MJy/sr" in bunit or unit == "MJy/sr":
        # JWST: pixel solid angle from PIXAR_SR if available, else from pix_scale
        pix_sr = hdr.get("PIXAR_SR") or ((pix_scale / 206265.0) ** 2)
        flux_jy = flux_sum * float(pix_sr) * 1.0e6
        if flux_jy <= 0:
            return None, snr
        mag = -2.5 * np.log10(flux_jy / 3631.0)
    elif "electron" in bunit.lower() or hdr.get("PHOTZPT") or hdr.get("ABMAG_ZP"):
        # HST: use ABMAG zeropoint from header if present, else fallback
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937   # ACS F814W default
        if flux_sum <= 0:
            return None, snr
        mag = float(zp) - 2.5 * np.log10(flux_sum)
    else:
        # Euclid MER: try ZP from header keywords; fall back to ZP=23.9 (nJy convention)
        zp = (hdr.get("ZP") or hdr.get("MAGZP")
              or hdr.get("ZEROPNT") or hdr.get("PHOTZP") or 23.9)
        if flux_sum <= 0:
            return None, snr
        mag = float(zp) - 2.5 * np.log10(flux_sum)
    return float(mag), snr


# ----------------------------------------------------------------------
def get_tile_path(kind, tmpl, tile):
    """Resolve the FITS path for a given band on a given tile."""
    if kind == "jwst":
        return JWST_DIR / tmpl.format(tile=tile)
    if kind == "hst":
        return HST_DIR / tmpl.format(tile=tile)
    if kind == "euclid":
        matches = sorted(glob.glob(str(EU_DIR / tmpl.format(tile=tile))))
        if not matches:
            return None
        return Path(matches[-1])
    raise RuntimeError(f"unknown kind: {kind}")


def main():
    rows = []
    for src in SOURCES:
        seq = src["seq"];  cid = src["id"]
        host_ra = src["host_ra"];  host_dec = src["host_dec"]
        print(f"\n==== ID={cid} (seq {seq}) ====")
        print(f"  host: RA={host_ra:.6f}  Dec={host_dec:.6f}")

        # 1) Detect SN
        png = JWST_PNG / f"sn_known34_{seq:04d}_jwst1.png"
        sx_pil, sy_pil = detect_sn_in_png(png)
        print(f"  SN detected at PIL ({sx_pil},{sy_pil})  dx={sx_pil-PNG_SIZE//2:+d} dy={sy_pil-PNG_SIZE//2:+d}")

        # 2) Convert -> SN RA/Dec via JWST tile WCS
        jwst_tile_path = JWST_DIR / f"mosaic_nircam_f150w_COSMOS-Web_30mas_{src['jwst_tile']}_v0_8_sci.fits"
        sn_ra, sn_dec, dx_nat, dy_nat = png_to_sky(
            host_ra, host_dec, sx_pil, sy_pil, jwst_tile_path, NATIVE_JWST_PX)
        print(f"  SN: RA={sn_ra:.6f}  Dec={sn_dec:.6f}")

        row = dict(
            id=cid, seq=seq,
            host_ra=host_ra, host_dec=host_dec,
            sn_ra=sn_ra, sn_dec=sn_dec,
            sn_dx_pil=sx_pil - PNG_SIZE // 2,
            sn_dy_pil=sy_pil - PNG_SIZE // 2,
        )

        # 3) Aperture phot for each band
        tiles_by_kind = dict(
            hst=src["hst_tile"], jwst=src["jwst_tile"], euclid=src["euclid_tile"])
        for (label, kind, tmpl) in BANDS:
            tile = tiles_by_kind[kind]
            path = get_tile_path(kind, tmpl, tile)
            if path is None or not Path(path).exists():
                row[f"mag_{label}"] = -1
                row[f"snr_{label}"] = 0
                print(f"    {label}: tile not found -> mag=-1")
                continue
            mag, snr = aperture_phot_ab(path, sn_ra, sn_dec)
            if mag is None or snr < DETECT_SNR:
                row[f"mag_{label}"] = -1
                row[f"snr_{label}"] = round(snr, 2)
                print(f"    {label}: SNR={snr:.2f} -> mag=-1")
            else:
                row[f"mag_{label}"] = round(mag, 2)
                row[f"snr_{label}"] = round(snr, 2)
                print(f"    {label}: mag={mag:.2f}  SNR={snr:.2f}")

        rows.append(row)

    # Write CSV
    cols = (
        ["id", "seq", "host_ra", "host_dec", "sn_ra", "sn_dec", "sn_dx_pil", "sn_dy_pil"]
        + [f"mag_{b[0]}" for b in BANDS]
        + [f"snr_{b[0]}" for b in BANDS]
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, -1) for c in cols})
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
