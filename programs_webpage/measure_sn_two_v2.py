"""SN detection + proper SNR using JWST v1.0 i2d ERR maps and HST wht maps.

For each of the 2 test sources (ID=318858, ID=320233):
  - Detect SN in the existing host-centered JWST1 PNG (blue-anomaly)
  - Convert PNG offset -> SN RA/Dec via the JWST tile WCS
  - Aperture photometry per band:
        JWST v1.0 i2d.fits.gz  : SCI ext + ERR ext  -> proper SNR
        HST sci.fits.gz + wht.fits.gz                -> proper SNR
        Euclid (no rms map)                          -> local-bg-rms fallback
    Aperture = 0.2"; background annulus 0.5-1.0"; subtract local median bg.
  - Magnitude conversion to AB:
        JWST MJy/sr  : flux_Jy = sum(SCI) * PIXAR_SR * 1e6
        HST e/s      : mag = PHOTZP - 2.5 log10(sum(SCI))
                       (ACS F814W default ABZP=25.937 if header missing)
        Euclid       : mag = ZP_guess - 2.5 log10(sum(SCI))  (rough)
  - combined sigma  = sqrt(sum(snr_i^2)) over detection bands

Output: /Users/suzuki/github/projects_cosmos/csvfiles/sn_test_2sources_v2.csv
"""
import warnings
warnings.filterwarnings("ignore")

import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS

# ----------------------------------------------------------------------
SOURCES = [
    dict(seq=23, id=318858, host_ra=150.185272, host_dec=1.842447,
         hst_tile="041", jwst_tile="A9", euclid_tile="101541377"),
    dict(seq=24, id=320233, host_ra=150.146358, host_dec=1.866410,
         hst_tile="041", jwst_tile="A9", euclid_tile="101541377"),
]

PNG_SIZE          = 480
NATIVE_JWST_PX    = 200
SEARCH_FRAC       = 0.15
LABEL_MASK_FRAC_Y = 0.20
LABEL_MASK_FRAC_X = 0.40
APER_ARCSEC       = 0.20    # SN aperture
# Tight ring immediately outside the aperture: estimates host SB locally
# (better than a wide annulus that averages SB across a steep host gradient)
HOST_RING_IN_AS   = 0.25
HOST_RING_OUT_AS  = 0.40
DETECT_SNR        = 3.0

HST_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
JWST_DIR = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8")    # NEW v1.0 i2d location
EU_DIR   = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")
JWST_PNG = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
OUT_CSV  = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_test_2sources_v2.csv")

# Bands: (label, kind, sci_template, err_template_or_None_for_internal_HDU)
#   - JWST: single i2d.fits.gz with SCI + ERR HDUs (err_tpl=None means "use ERR ext")
#   - HST: separate _sci and _wht files (err_tpl is the wht template)
#   - Euclid: only sci available
BANDS = [
    ("F814W",   "hst",        "acs_I_030mas_{tile}_sci.fits.gz",      "acs_I_030mas_{tile}_wht.fits.gz"),
    ("VIS",     "euclid",     "EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile}-*.fits",   None),
    ("Y",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE{tile}-*.fits", None),
    ("J",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE{tile}-*.fits", None),
    ("H",       "euclid",     "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE{tile}-*.fits", None),
    ("F115W",   "jwst",       "mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits.gz", None),
    ("F150W",   "jwst",       "mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits.gz", None),
    ("F277W",   "jwst",       "mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits.gz", None),
    ("F444W",   "jwst",       "mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits.gz", None),
]


# ----------------------------------------------------------------------
def detect_sn_in_png(jwst1_png_path):
    img = np.asarray(Image.open(jwst1_png_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    b, g, r = img[..., 2], img[..., 1], img[..., 0]
    blueness = b - 0.5*(r + g)
    blueness[:int(LABEL_MASK_FRAC_Y * h), :int(LABEL_MASK_FRAC_X * w)] = -10
    cx, cy = w // 2, h // 2
    half = int(SEARCH_FRAC * min(w, h))
    box = np.full_like(blueness, -10)
    box[cy-half:cy+half, cx-half:cx+half] = blueness[cy-half:cy+half, cx-half:cx+half]
    yi, xi = np.unravel_index(np.argmax(box), box.shape)
    return int(xi), int(yi)


def png_to_sky(host_ra, host_dec, sx_pil, sy_pil, tile_fits_path, native_size):
    cx_pil = PNG_SIZE // 2;  cy_pil = PNG_SIZE // 2
    dx_native = (sx_pil - cx_pil) * (native_size / PNG_SIZE)
    dy_native = -(sy_pil - cy_pil) * (native_size / PNG_SIZE)
    with fits.open(tile_fits_path) as h:
        # JWST i2d: WCS is on the SCI HDU; HST: PRIMARY
        hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(hdu.header)
    hx, hy = wcs.all_world2pix(host_ra, host_dec, 0)
    sn_ra, sn_dec = wcs.all_pix2world(hx + dx_native, hy + dy_native, 0)
    return float(sn_ra), float(sn_dec)


# ----------------------------------------------------------------------
def aperture_phot(sci_path, err_path_or_None, sn_ra, sn_dec, kind):
    """Return (mag, snr, flux_sum, err_aper) for an aperture phot measurement.

    sci_path : SCI file (HST/Euclid) or i2d file (JWST; ERR in same file)
    err_path_or_None : HST wht file, or None if err is internal (JWST) or
                      unavailable (Euclid)
    kind: 'hst' | 'jwst' | 'euclid'
    """
    with fits.open(sci_path, memmap=True) as h:
        if kind == "jwst":
            sci_hdu = h["SCI"]
            err_hdu = h["ERR"]
            err = err_hdu.data.astype(np.float64)
        else:
            sci_hdu = h[0]
            err = None  # filled below from wht or local bg
        sci = sci_hdu.data.astype(np.float64)
        hdr = sci_hdu.header
        wcs = WCS(hdr)

    # Get HST wht -> per-pixel sigma if available
    if kind == "hst" and err_path_or_None is not None:
        with fits.open(err_path_or_None, memmap=True) as h:
            wht = h[0].data.astype(np.float64)
        # err per px = 1/sqrt(wht); guard against zeros
        safe = (wht > 0) & np.isfinite(wht)
        err = np.full_like(sci, np.inf)
        err[safe] = 1.0 / np.sqrt(wht[safe])

    # Pixel scale (arcsec/px)
    try:
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    except Exception:
        pix_scale = abs(hdr.get("CD1_1", 1.0/3600.0/30.0)) * 3600.0

    aper_px       = APER_ARCSEC      / pix_scale
    ring_in_px    = HOST_RING_IN_AS  / pix_scale
    ring_out_px   = HOST_RING_OUT_AS / pix_scale
    # Expand the window: aperture + host ring AND a far-out sky region for noise
    sky_half_as = 2.0          # sample sky pixels out to 2 arcsec away from SN
    half = int(np.ceil(max(ring_out_px, sky_half_as / pix_scale)) + 2)

    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci.shape
    x0 = max(0, int(sx - half)); x1 = min(nx, int(sx + half + 1))
    y0 = max(0, int(sy - half)); y1 = min(ny, int(sy + half + 1))
    sub_sci = sci[y0:y1, x0:x1]
    if sub_sci.size == 0:
        return None, 0.0, 0.0, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - sx)**2 + (yy - sy)**2)

    in_aper = (rr <= aper_px)
    in_host = (rr >= ring_in_px) & (rr <= ring_out_px)
    n_aper = int(in_aper.sum())
    if n_aper == 0 or in_host.sum() == 0:
        return None, 0.0, 0.0, 0.0

    # --- Host: ONE NUMBER (median surface brightness in tight ring just
    # outside the aperture) -- subtract host SB * N_aper from aperture sum.
    host_sb = float(np.nanmedian(sub_sci[in_host]))
    flux_aper_total = float(np.nansum(sub_sci[in_aper]))
    flux_sum = flux_aper_total - host_sb * n_aper

    # --- Noise: use the RING's pixel sigma (captures sky + local host
    # clumpiness on the relevant spatial scale).  This is what genuinely
    # limits SN detection when the SN sits on a host galaxy.
    # sigma_aper = sigma_ring_per_px * sqrt(N_aper)
    ring_vals = sub_sci[in_host]
    valid = np.isfinite(ring_vals)
    if valid.sum() > 5:
        sigma_ring_per_px = float(np.nanstd(ring_vals[valid]))
    else:
        sigma_ring_per_px = 1.0
    sigma_aper = sigma_ring_per_px * np.sqrt(n_aper)
    snr = flux_sum / (sigma_aper + 1e-30)

    # -> AB mag
    bunit = (hdr.get("BUNIT") or "").strip()
    if "MJy/sr" in bunit or kind == "jwst":
        pix_sr = hdr.get("PIXAR_SR") or ((pix_scale / 206265.0) ** 2)
        flux_jy = flux_sum * float(pix_sr) * 1.0e6
        mag = (-2.5 * np.log10(flux_jy / 3631.0)) if flux_jy > 0 else None
    elif kind == "hst":
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
        mag = (float(zp) - 2.5 * np.log10(flux_sum)) if flux_sum > 0 else None
    else:  # euclid
        zp = (hdr.get("ZP") or hdr.get("MAGZP")
              or hdr.get("ZEROPNT") or hdr.get("PHOTZP") or 23.9)
        mag = (float(zp) - 2.5 * np.log10(flux_sum)) if flux_sum > 0 else None

    return mag, float(snr), flux_sum, sigma_aper


def get_paths(kind, sci_tpl, err_tpl, tile):
    """Resolve sci_path and (optional) err_path."""
    if kind == "jwst":
        return JWST_DIR / sci_tpl.format(tile=tile), None
    if kind == "hst":
        return HST_DIR / sci_tpl.format(tile=tile), HST_DIR / err_tpl.format(tile=tile)
    if kind == "euclid":
        matches = sorted(glob.glob(str(EU_DIR / sci_tpl.format(tile=tile))))
        return (Path(matches[-1]) if matches else None), None
    raise RuntimeError(kind)


def main():
    rows = []
    for src in SOURCES:
        seq = src["seq"];  cid = src["id"]
        host_ra = src["host_ra"];  host_dec = src["host_dec"]
        print(f"\n==== ID={cid} (seq {seq}) ====")
        png = JWST_PNG / f"sn_known34_{seq:04d}_jwst1.png"
        sx_pil, sy_pil = detect_sn_in_png(png)
        print(f"  SN at PIL ({sx_pil},{sy_pil}); dx={sx_pil-PNG_SIZE//2:+d} dy={sy_pil-PNG_SIZE//2:+d}")
        jwst_tile_path = JWST_DIR / f"mosaic_nircam_f150w_COSMOS-Web_30mas_{src['jwst_tile']}_v1.0_i2d.fits.gz"
        sn_ra, sn_dec = png_to_sky(host_ra, host_dec, sx_pil, sy_pil,
                                   jwst_tile_path, NATIVE_JWST_PX)
        print(f"  SN: RA={sn_ra:.6f}  Dec={sn_dec:.6f}")

        row = dict(id=cid, seq=seq,
                   host_ra=host_ra, host_dec=host_dec,
                   sn_ra=sn_ra, sn_dec=sn_dec,
                   sn_dx_pil=sx_pil - PNG_SIZE // 2,
                   sn_dy_pil=sy_pil - PNG_SIZE // 2)

        tiles_by_kind = dict(hst=src["hst_tile"], jwst=src["jwst_tile"], euclid=src["euclid_tile"])
        detection_snrs = []
        for (label, kind, sci_tpl, err_tpl) in BANDS:
            tile = tiles_by_kind[kind]
            sci_p, err_p = get_paths(kind, sci_tpl, err_tpl, tile)
            if sci_p is None or not Path(sci_p).exists():
                row[f"mag_{label}"]  = -1
                row[f"snr_{label}"]  = -1
                row[f"flux_{label}"] = -1
                row[f"sig_{label}"]  = -1
                print(f"    {label}: tile not found")
                continue
            mag, snr, flux, sigma = aperture_phot(sci_p, err_p, sn_ra, sn_dec, kind)
            if mag is None:
                row[f"mag_{label}"]  = -1
                row[f"snr_{label}"]  = round(snr, 2)
                row[f"flux_{label}"] = round(flux, 6)
                row[f"sig_{label}"]  = round(sigma, 6)
                print(f"    {label}: snr={snr:.2f} flux<=0 -> mag=-1")
            else:
                row[f"mag_{label}"]  = round(mag, 2)
                row[f"snr_{label}"]  = round(snr, 2)
                row[f"flux_{label}"] = round(flux, 6)
                row[f"sig_{label}"]  = round(sigma, 6)
                tag = "DET" if snr >= DETECT_SNR else "non-det"
                print(f"    {label}: mag={mag:.2f}  snr={snr:.2f}  [{tag}]")
                if snr >= DETECT_SNR:
                    detection_snrs.append(snr)

        # Combined significance: sqrt(sum(snr_i^2)) over detected bands
        if detection_snrs:
            combined = float(np.sqrt(np.sum(np.array(detection_snrs) ** 2)))
            row["combined_sigma"] = round(combined, 2)
            print(f"  combined sigma (over {len(detection_snrs)} det bands): {combined:.2f}")
        else:
            row["combined_sigma"] = -1
            print(f"  no detections -> combined_sigma = -1")
        rows.append(row)

    cols = (["id", "seq", "host_ra", "host_dec", "sn_ra", "sn_dec",
             "sn_dx_pil", "sn_dy_pil"]
            + [f"mag_{b[0]}"  for b in BANDS]
            + [f"snr_{b[0]}"  for b in BANDS]
            + [f"flux_{b[0]}" for b in BANDS]
            + [f"sig_{b[0]}"  for b in BANDS]
            + ["combined_sigma"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, -1) for c in cols})
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
