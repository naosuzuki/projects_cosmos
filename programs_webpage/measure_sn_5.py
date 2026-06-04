"""SN photometry for the 5-source test (2 JWST + 3 HST detections).

For JWST-detected SN (telescope='JWST' in known34c_single.fits):
    Detect blue-anomaly in JWST1 PNG (existing logic), convert to RA/Dec.
For HST-detected SN (telescope='HST'):
    Use the catalog RA/Dec directly (the cutout center already IS the SN).

Then aperture photometry on all 9 bands using the production tile paths
(prefers uncompressed .fits over .fits.gz, JWST i2d SCI/ERR HDUs).

Output: sn_test_5sources.csv  with id, telescope, sn_ra/dec, per-band mag/snr,
combined_sigma, best_snr_band.
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
from astropy.table import Table

# ----------------------------------------------------------------------
CAND_FITS = Path("/Users/suzuki/github/projects_euclid/fits_tbl/known34c_single.fits")
OUT_CSV   = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_test_5sources.csv")
JWST_PNG  = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
JWST_DIR  = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8")
HST_DIR   = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
EU_DIR    = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")

# Map ID -> tile assignment (looked up earlier from sn_known34_*_index.csv)
TILES = {
    318858: dict(hst="041", jwst="A9",  euclid="101541377", jwst_seq_in_known34=23),
    320233: dict(hst="041", jwst="A9",  euclid="101541377", jwst_seq_in_known34=24),
    130972: dict(hst="052", jwst="A4",  euclid="101542818", jwst_seq_in_known34=15),
    371996: dict(hst="040", jwst="A10", euclid="101542818", jwst_seq_in_known34=26),
    471959: dict(hst="076", jwst="B3",  euclid="101545698", jwst_seq_in_known34=34),
}

PNG_SIZE = 480; NATIVE_JWST_PX = 200
SEARCH_FRAC = 0.15; LABEL_MASK_FRAC_Y = 0.20; LABEL_MASK_FRAC_X = 0.40
APER_ARCSEC   = 0.20
HOST_RING_IN_AS = 0.25; HOST_RING_OUT_AS = 0.40
DETECT_SNR    = 3.0

BANDS = [  # (label, kind, sci pattern, err pattern or None for JWST internal/Euclid)
    ("F814W", "hst",    "acs_I_030mas_{tile}_sci.fits",          "acs_I_030mas_{tile}_wht.fits"),
    ("VIS",   "euclid", "EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile}-*.fits",   None),
    ("Y",     "euclid", "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE{tile}-*.fits", None),
    ("J",     "euclid", "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE{tile}-*.fits", None),
    ("H",     "euclid", "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE{tile}-*.fits", None),
    ("F115W", "jwst",   "mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits", None),
    ("F150W", "jwst",   "mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits", None),
    ("F277W", "jwst",   "mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits", None),
    ("F444W", "jwst",   "mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits", None),
]


def detect_sn_in_jwst_png(jwst1_png_path):
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


def png_to_sky(host_ra, host_dec, sx_pil, sy_pil, jwst_tile_path):
    """Convert PIL offset to RA/Dec via JWST tile WCS."""
    cx_pil = PNG_SIZE // 2;  cy_pil = PNG_SIZE // 2
    dx_native = (sx_pil - cx_pil) * (NATIVE_JWST_PX / PNG_SIZE)
    dy_native = -(sy_pil - cy_pil) * (NATIVE_JWST_PX / PNG_SIZE)
    with fits.open(jwst_tile_path) as h:
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(sci_hdu.header)
    hx, hy = wcs.all_world2pix(host_ra, host_dec, 0)
    sn_ra, sn_dec = wcs.all_pix2world(hx + dx_native, hy + dy_native, 0)
    return float(sn_ra), float(sn_dec)


def jwst_tile_path_for(jwst_tile, band):
    """Return path to JWST i2d file (prefers uncompressed v1.0; falls back
    to .gz, then to v0.8 sci-only)."""
    name = f"mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v1.0_i2d.fits"
    if (JWST_DIR / name).exists():
        return JWST_DIR / name
    if (JWST_DIR / f"{name}.gz").exists():
        return JWST_DIR / f"{name}.gz"
    v08 = JWST_DIR / "scidir" / f"mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v0_8_sci.fits"
    return v08


def resolve_path(kind, sci_tpl, err_tpl, tile):
    if kind == "jwst":
        return jwst_tile_path_for(tile, sci_tpl.split("_")[2]), None
    if kind == "hst":
        sci = HST_DIR / sci_tpl.format(tile=tile)
        if not sci.exists():
            sci = HST_DIR / (sci_tpl.format(tile=tile) + ".gz")
        err = HST_DIR / err_tpl.format(tile=tile)
        if not err.exists():
            err = HST_DIR / (err_tpl.format(tile=tile) + ".gz")
        return sci, err
    if kind == "euclid":
        matches = sorted(glob.glob(str(EU_DIR / sci_tpl.format(tile=tile))))
        return (Path(matches[-1]) if matches else None), None
    raise RuntimeError(kind)


def aperture_phot(sci_path, err_path, sn_ra, sn_dec, kind):
    """Return (mag_AB, snr).  See measure_sn_two_v2.py for derivation.
    Returns (None, 0) if the file cannot be read or HDUs missing."""
    try:
        with fits.open(sci_path, memmap=True) as h:
            hdu_names = [x.name for x in h]
            if kind == "jwst":
                if "SCI" in hdu_names:
                    sci_hdu = h["SCI"]
                    err = (h["ERR"].data.astype(np.float64)
                           if "ERR" in hdu_names else None)
                else:
                    # v0.8 sci-only fallback: just PRIMARY
                    sci_hdu = h[0]; err = None
            else:
                sci_hdu = h[0]; err = None
            sci = sci_hdu.data.astype(np.float64)
            hdr = sci_hdu.header
            wcs = WCS(hdr)
    except Exception as e:
        print(f"      (open failed: {type(e).__name__}: {e})", flush=True)
        return None, 0.0
    if kind == "hst" and err_path is not None:
        with fits.open(err_path, memmap=True) as h:
            wht = h[0].data.astype(np.float64)
        safe = (wht > 0) & np.isfinite(wht)
        err = np.full_like(sci, np.inf)
        err[safe] = 1.0 / np.sqrt(wht[safe])
    try:
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    except Exception:
        pix_scale = 0.03
    aper_px    = APER_ARCSEC      / pix_scale
    ring_in    = HOST_RING_IN_AS  / pix_scale
    ring_out   = HOST_RING_OUT_AS / pix_scale
    half = int(np.ceil(max(ring_out, 2.0 / pix_scale)) + 2)

    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci.shape
    x0 = max(0, int(sx - half)); x1 = min(nx, int(sx + half + 1))
    y0 = max(0, int(sy - half)); y1 = min(ny, int(sy + half + 1))
    sub = sci[y0:y1, x0:x1]
    if sub.size == 0:
        return None, 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - sx)**2 + (yy - sy)**2)
    in_aper = (rr <= aper_px); in_ring = (rr >= ring_in) & (rr <= ring_out)
    n_aper = int(in_aper.sum())
    if n_aper == 0 or in_ring.sum() == 0:
        return None, 0.0
    host_sb = float(np.nanmedian(sub[in_ring]))
    flux = float(np.nansum(sub[in_aper])) - host_sb * n_aper
    sigma_ring_per_px = float(np.nanstd(sub[in_ring]))
    sigma_aper = sigma_ring_per_px * np.sqrt(n_aper)
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


def main():
    cat = Table.read(CAND_FITS)
    rows = []
    for entry in cat:
        cid = int(entry["id"])
        host_ra = float(entry["host_ra"]); host_dec = float(entry["host_dec"])
        telescope = str(entry["telescope"]).strip().upper()
        tiles = TILES[cid]
        print(f"\n==== ID={cid}  ({telescope} SN, tiles HST={tiles['hst']} JWST={tiles['jwst']}) ====")

        if telescope == "JWST":
            # blue-anomaly detection in JWST PNG
            seq = tiles["jwst_seq_in_known34"]
            png = JWST_PNG / f"sn_known34_{seq:04d}_jwst1.png"
            sx, sy = detect_sn_in_jwst_png(png)
            sn_ra, sn_dec = png_to_sky(host_ra, host_dec, sx, sy,
                                       jwst_tile_path_for(tiles["jwst"], "f150w"))
            print(f"  JWST blue-anomaly SN @ PIL({sx},{sy}) -> RA={sn_ra:.6f} Dec={sn_dec:.6f}")
        else:  # HST SN
            sn_ra, sn_dec = host_ra, host_dec
            print(f"  HST SN at catalog position: RA={sn_ra:.6f} Dec={sn_dec:.6f}")

        row = dict(id=cid, telescope=telescope,
                   host_ra=host_ra, host_dec=host_dec,
                   sn_ra=sn_ra, sn_dec=sn_dec)

        det_snrs = []
        for (lbl, kind, sci_tpl, err_tpl) in BANDS:
            tile = tiles[kind]
            sci, err = resolve_path(kind, sci_tpl, err_tpl, tile)
            if sci is None or not Path(sci).exists():
                row[f"mag_{lbl}"] = -1; row[f"snr_{lbl}"] = -1
                print(f"    {lbl}: tile MISSING ({sci})")
                continue
            mag, snr = aperture_phot(sci, err, sn_ra, sn_dec, kind)
            if mag is None:
                row[f"mag_{lbl}"] = -1; row[f"snr_{lbl}"] = round(snr, 2)
                print(f"    {lbl}: snr={snr:.2f} (flux<=0) -> mag=-1")
            else:
                row[f"mag_{lbl}"] = round(mag, 2)
                row[f"snr_{lbl}"] = round(snr, 2)
                tag = "DET" if snr >= DETECT_SNR else "non-det"
                print(f"    {lbl}: mag={mag:.2f}  snr={snr:.2f}  [{tag}]")
                if snr >= DETECT_SNR:
                    det_snrs.append((lbl, snr, kind))

        # Combined sigma (over DETECTION-survey bands only, to keep it meaningful)
        # If telescope == JWST -> combine JWST bands; if HST -> just HST
        if telescope == "JWST":
            keep = [s for (lbl, s, k) in det_snrs if k == "jwst"]
        else:  # HST
            keep = [s for (lbl, s, k) in det_snrs if k == "hst"]
        combined = float(np.sqrt(np.sum(np.array(keep) ** 2))) if keep else -1.0
        row["combined_sigma"] = round(combined, 2) if combined > 0 else -1

        # Best band for the upper-right "SN=X.Xσ (band)" label
        if telescope == "JWST":
            jwst_bands = [(lbl, s) for (lbl, s, k) in det_snrs if k == "jwst"]
            if jwst_bands:
                lbl, s = max(jwst_bands, key=lambda x: x[1])
                row["best_band"] = lbl; row["best_snr"] = round(s, 2)
        else:
            hst_bands = [(lbl, s) for (lbl, s, k) in det_snrs if k == "hst"]
            if hst_bands:
                lbl, s = max(hst_bands, key=lambda x: x[1])
                row["best_band"] = lbl; row["best_snr"] = round(s, 2)
        row.setdefault("best_band", ""); row.setdefault("best_snr", -1)
        print(f"  combined sigma = {row['combined_sigma']}   best = {row['best_band']}={row['best_snr']}")

        rows.append(row)

    cols = (["id", "telescope", "host_ra", "host_dec", "sn_ra", "sn_dec",
             "best_band", "best_snr", "combined_sigma"]
            + [f"mag_{b[0]}" for b in BANDS]
            + [f"snr_{b[0]}" for b in BANDS])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, -1) for c in cols})
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
